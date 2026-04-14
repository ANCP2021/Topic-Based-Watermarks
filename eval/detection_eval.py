import torch
import sys
import os
import json
src_path = os.path.abspath(os.path.join('..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
from semantic_topic_extension import EmbeddingMapper
from model import (
    load_model, 
    load_sentence_model, 
)
from topic_watermark_processor import WatermarkBase
from keybert import KeyBERT
import collections
from nltk.util import ngrams
import scipy.stats
from torch import Tensor
from math import sqrt
from collections import Counter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
#     'model_name_or_path': 'facebook/opt-6.7b',
    'model_name_or_path': 'google/gemma-7b',
    'load_fp16' : True,
    'prompt_max_length': None, 
    'max_new_tokens': 200, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'delta': 3.0, 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 4.75, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'topic_token_mapping': {},
    'granularity': 'kmeans',
}

topic_list = ["animals", "technology", "sports", "medicine"]

model, tokenizer = load_model(args)
print("Model loaded")
sentence_model = load_sentence_model()
print("Sentence model loaded")
args['topic_token_mapping'] = topic_token_mapping

topic_model = KeyBERT()
def load_topic_model(input_text, n_topics=7):
    keywords = topic_model.extract_keywords(
        input_text,
        keyphrase_ngram_range=(1, 1),
        stop_words='english',
        top_n=n_topics
    )
    topics = [kw[0].lower() for kw in keywords]
    return topics

class TopicWatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer = None,
        sentence_model = None,
        z_threshold: float = 4.0,
        ignore_repeated_bigrams: bool = True,
        detected_topics: list[str] = [],
        granularity: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        # Instance attributes
        self.tokenizer = tokenizer
        self.device = device
        self.sentence_model = sentence_model
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        # Seeding scheme 
        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        # Repeated bigrams handling
        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

        self.detected_topics=detected_topics
        self.granularity=granularity

    # Computes z-scores for the observed green token count
    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        numer = observed_count - self.expected_count * T
        denom = sqrt(T * self.expected_count * (1 - self.expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        # Computes p-value from the z-score
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_sequence(self, input_ids: Tensor):
        green_token_count, num_tokens_scored = 0, 0

        if self.ignore_repeated_bigrams:
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq)
            for bigram in freq:
                prefix = torch.tensor([bigram[0]], device=self.device)
                greenlist_ids = self._get_greenlist_ids(prefix)
                if bigram[1] in greenlist_ids:
                    green_token_count += 1
        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            for idx in range(self.min_prefix_len, len(input_ids)):
                prefix = input_ids[:idx]
                curr_token = input_ids[idx].item()
                greenlist_ids = set(self._get_greenlist_ids(prefix))
                if curr_token in greenlist_ids:
                    green_token_count += 1

        z = self._compute_z_score(green_token_count, num_tokens_scored)
        p = self._compute_p_value(z)
        return {
            "num_tokens_scored": num_tokens_scored,
            "num_green_tokens": green_token_count,
            "green_fraction": green_token_count / num_tokens_scored,
            "z_score": z,
            "p_value": p
        }

    def _select_topic(self, detected_topics):
        embedding_mapper = EmbeddingMapper(self.tokenizer, self.sentence_model)

        """Select the first detected topic that is recognized in the topic mappings."""
        for topic in detected_topics:
            if topic.lower() in self.topic_token_mapping:
                return topic.lower()
            
        if self.granularity == 'kmeans':
            detected_topic, _ = embedding_mapper.kmeans_detected_topics_to_embeddings(
                list(self.topic_token_mapping.keys()), 
                detected_topics,
            )
        else:
            detected_topic, _ = embedding_mapper.detected_topics_to_embeddings(
                list(self.topic_token_mapping.keys()), 
                detected_topics
            )
        return detected_topic

    def detect(self, text: str):
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
        if input_ids[0] == self.tokenizer.bos_token_id:
            input_ids = input_ids[1:]

        self.detected_topic = self._select_topic(self.detected_topics)
        topic_tokens = self.topic_token_mapping.get(self.detected_topic, self.vocab)
        self.expected_count = len(topic_tokens) / self.vocab_size

        scores = self._score_sequence(input_ids)
        scores["prediction"] = scores["z_score"] > self.z_threshold
        scores["confidence"] = 1 - scores["p_value"]
        return scores
    
    
def resolve_true_topic(detected_topics, args, tokenizer, sentence_model):
    for topic in detected_topics:
        if topic.lower() in args['topic_token_mapping']:
            return topic.lower()

    embedding_mapper = EmbeddingMapper(tokenizer, sentence_model)
    if args['granularity'] == 'kmeans':
        detected_topic, _ = embedding_mapper.kmeans_detected_topics_to_embeddings(
            list(args['topic_token_mapping'].keys()), detected_topics)
    else:
        detected_topic, _ = embedding_mapper.detected_topics_to_embeddings(
            list(args['topic_token_mapping'].keys()), detected_topics)
    return detected_topic
    
def detect_strict(sample, args, tokenizer, sentence_model, device):
    generated_text = sample["generated"]
    raw_topic_keywords = sample["topic"] 
    
    true_topic = resolve_true_topic(raw_topic_keywords, args, tokenizer, sentence_model)
    detected_topics = load_topic_model(generated_text)

    detector = TopicWatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        seeding_scheme=args['seeding_scheme'],
        device=device,
        tokenizer=tokenizer,
        sentence_model=sentence_model,
        z_threshold=args['detection_z_threshold'],
        ignore_repeated_bigrams=args['ignore_repeated_bigrams'],
        select_green_tokens=args['select_green_tokens'],
        topic_token_mapping=args['topic_token_mapping'],
        detected_topics=detected_topics,
        granularity=args['granularity']
    )

    try:
        output = detector.detect(text=generated_text)
        predicted_topic = detector.detected_topic
        z_score = output["z_score"]
        is_watermarked = output["prediction"]
    except Exception as e:
        print(f"Detection error for sample {sample.get('id', 'N/A')}: {e}")
        predicted_topic = "error"
        z_score = None
        is_watermarked = False

    return {
        "id": sample.get("id"),
        "true_topic": true_topic,
        "predicted_topic": predicted_topic,
        "z_score": z_score,
        "is_watermarked": is_watermarked
    }

def detect_sliding_window(sample, args, tokenizer, sentence_model, device):
    generated_text = sample["generated"]
    raw_topic_keywords = sample["topic"]
    
    true_topic = resolve_true_topic(raw_topic_keywords, args, tokenizer, sentence_model)

    input_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    window_size = 50
    window_topics = []

    for i in range(0, len(input_ids), window_size):
        window_ids = input_ids[i:i+window_size]
        if len(window_ids) < 5:
            continue 

        window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
        detected_keywords = load_topic_model(window_text, n_topics=3)

        try:
            embedding_mapper = EmbeddingMapper(tokenizer, sentence_model)
            if any(k.lower() in args['topic_token_mapping'] for k in detected_keywords):
                for k in detected_keywords:
                    if k.lower() in args['topic_token_mapping']:
                        window_topics.append(k.lower())
                        break
            else:
#                 if args["granularity"] == "kmeans":
#                     mapped_topic, _ = embedding_mapper.kmeans_detected_topics_to_embeddings(
#                         list(args['topic_token_mapping'].keys()), detected_keywords)
#                 else:
#                     mapped_topic, _ = embedding_mapper.detected_topics_to_embeddings(
#                         list(args['topic_token_mapping'].keys()), detected_keywords)
                if len(detected_keywords) < 3:
                    mapped_topic, _ = embedding_mapper.detected_topics_to_embeddings(
                        list(args['topic_token_mapping'].keys()), detected_keywords)
                elif args["granularity"] == "kmeans":
                    mapped_topic, _ = embedding_mapper.kmeans_detected_topics_to_embeddings(
                        list(args['topic_token_mapping'].keys()), detected_keywords)
                else:
                    mapped_topic, _ = embedding_mapper.detected_topics_to_embeddings(
                        list(args['topic_token_mapping'].keys()), detected_keywords)

                window_topics.append(mapped_topic)
        except Exception as e:
            print(f"Window topic extraction failed: {e}")
            continue

    if not window_topics:
        return {
            "id": sample.get("id"),
            "true_topic": true_topic,
            "predicted_topic": "error",
            "z_score": None,
            "is_watermarked": False
        }

    topic_counts = Counter(window_topics)
    predicted_topic = topic_counts.most_common(1)[0][0]

    detector = TopicWatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        seeding_scheme=args['seeding_scheme'],
        device=device,
        tokenizer=tokenizer,
        sentence_model=sentence_model,
        z_threshold=args['detection_z_threshold'],
        ignore_repeated_bigrams=args['ignore_repeated_bigrams'],
        select_green_tokens=args['select_green_tokens'],
        topic_token_mapping=args['topic_token_mapping'],
        detected_topics=[predicted_topic],
        granularity=args['granularity']
    )

    try:
        output = detector.detect(text=generated_text)
        z_score = output["z_score"]
        is_watermarked = output["prediction"]
    except Exception as e:
        print(f"Detection failed on sample {sample.get('id')}: {e}")
        z_score = None
        is_watermarked = False

    return {
        "id": sample.get("id"),
        "true_topic": true_topic,
        "predicted_topic": predicted_topic,
        "z_score": z_score,
        "is_watermarked": is_watermarked
    }


def detect_max_z_via_topic_detector(sample, args, tokenizer, sentence_model, device):
    start_time = time.time() 
    generated_text = sample["generated"]
    topic_list = list(args['topic_token_mapping'].keys())
    raw_topic_keywords = sample["topic"]
    
    true_topic = resolve_true_topic(raw_topic_keywords, args, tokenizer, sentence_model)

    results = []

    for topic in topic_list:
        detector = TopicWatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            seeding_scheme=args['seeding_scheme'],
            device=device,
            tokenizer=tokenizer,
            sentence_model=sentence_model,
            z_threshold=args['detection_z_threshold'],
            ignore_repeated_bigrams=args['ignore_repeated_bigrams'],
            select_green_tokens=args['select_green_tokens'],
            topic_token_mapping=args['topic_token_mapping'],
            detected_topics=[topic], 
            granularity=args['granularity']
        )

        try:
            output = detector.detect(text=generated_text)
            z_score = output["z_score"]
            is_watermarked = output["prediction"]
            results.append((topic, z_score, is_watermarked))
        except Exception as e:
            print(f"Detection failed on sample {sample.get('id')}: {e}")
            z_score = None
            is_watermarked = False

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if not results:
        return {
            "id": sample.get("id"),
            "elapsed_time": elapsed_time
        }

    highest = max(results, key=lambda x: x[1])
    predicted_topic, z_score, watermarked = highest
    
    if predicted_topic != true_topic:
        watermarked = False

    return {
        "id": sample.get("id"),
        "elapsed_time": elapsed_time
    }

    
results = []
input_file = ""

with open(input_file, "r") as f:
    for line in f:
        sample = json.loads(line)
        result = detect_strict(sample, args, tokenizer, sentence_model, device)
        result = detect_sliding_window(sample, args, tokenizer, sentence_model, device)
        result = detect_max_z_via_topic_detector(sample, args, tokenizer, sentence_model, device)
        results.append(result)

