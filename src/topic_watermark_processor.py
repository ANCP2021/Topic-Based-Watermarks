from __future__ import annotations
import collections
from math import sqrt

import scipy.stats

import torch
from torch import Tensor
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from transformers import LogitsProcessor

from nltk.util import ngrams
from semantic_topic_extension import EmbeddingMapper

"""
    - Managing watermarking parameters and operations given a vocabulary V.
    - Allows for seeding the random number generator based on input tokens and selecting a subset of 
    green listed tokens from V. 
    - Provides methods to seed the random number generator using a specific scheme and to generate the list
    of green IDs based on the seeded random number generator/other paramters.
"""
class WatermarkBase:

    def __init__(
        self,
        vocab: list[int] = None,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # mostly unused/always default
        hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        select_green_tokens: bool = True,
        topic_token_mapping: dict = None,
        detected_topic: str = "",
    ):

        # watermarking parameters
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        self.topic_token_mapping = topic_token_mapping or {}
        self.detected_topic = detected_topic

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None: 
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            # There needs to be at least one token in the input for seeding 
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            # Seed the random number generator using the last token in the input sequence and hash key
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else: # error for unimplmented seeding 
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens according to the seeding_scheme
        self._seed_rng(input_ids)

        if self.detected_topic and self.detected_topic in self.topic_token_mapping:
            # Use the entire set of tokens for topic
            topic_tokens = self.topic_token_mapping[self.detected_topic]
            greenlist_ids = topic_tokens
        else:
            raise ValueError(f"Incorrectly mapped topic: {self.detected_topic}")

        return greenlist_ids

"""
    - Extends WatermarkBase and HuggingFace LogitsProcessor modifying the logits of a model bassed on
    the watermarking scheme
    - Calculates a mask for green listed tokens, bias of logtis of green listed tokens, and applies
    the modifications during the forward pass
"""
class TopicWatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        # Initializes the parent class with the given arguements 
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # Creates the mask for the green listed tokens
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            # Sets the positions of green list tokens to 1
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        # Converts the mask to a boolean tensor
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        # Increases the scores if the green listed tokens by a green list bias
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Initialize the random number generator if not set, colocating on the same device as input ids
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        # List to store the green list IDs for each sequence in the batch
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            # Get green list IDs for the current sequence
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        # Calculates the mast for the green listed tokens
        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        # Bias the logits of the green listed tokens
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)
        return scores

"""
    - Inherits from the WatermarkBase
    - Detectes watermarks in the token sequence
    - Initilizes device, tokenizer, and normalization strategies
    - Computes z-scores and p-values for sequences whether tokens fall into the green list or not
    - Considers bigrams (two consecutive tokens) or standard token sequences and returns 
    metrics about the watermark detection (number of green tokens, z-scores, p-values)
    - Detectin of watermarks in either raw or pre-tokenized input, normalizing the text and performing
    the detection process returning results
# """
class TopicWatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        sentence_model: SentenceTransformer = None,
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
        print(f"oberserved: {observed_count}")

        numer = observed_count - self.expected_count * T
        denom = sqrt(T * self.expected_count * (1 - self.expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        # Computes p-value from the z-score
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        if self.ignore_repeated_bigrams:
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())
        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len
            if num_tokens_scored < 1:
                raise ValueError(
                    (
                        f"Must have at least {1} token to score after "
                        f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                    )
                )
            # Standard method.
            # Since we generally need at least 1 token (for the simplest scheme)
            # we start the iteration over the token sequence with a minimum
            # num tokens as the first prefix for the seeding scheme,
            # and at each step, compute the greenlist induced by the
            # current prefix and check if the current token falls in the greenlist.
            green_token_count, green_token_mask = 0, []
            for idx in range(self.min_prefix_len, len(input_ids)):
                curr_token = input_ids[idx].item()
                greenlist_ids = set(self._get_greenlist_ids(input_ids[:idx]))
                if curr_token in greenlist_ids:
                    green_token_count += 1
                    green_token_mask.append(True)
                else:
                    green_token_mask.append(False)

        # Results dictionary
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

    def _select_topic(self, detected_topics):
        embedding_mapper = EmbeddingMapper(self.tokenizer, self.sentence_model)

        """Select the first detected topic that is recognized in the topic mappings."""
        for topic in detected_topics:
            if topic.lower() in self.topic_token_mapping:
                print(f"Detected topic from list: {topic.lower()}")
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
        print(f"Detected topic from list: {detected_topic}")
        return detected_topic

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:

        # Look at either raw or tokenized text is provided, not both
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        if tokenized_text is None: # Ensure tokenizer is available for raw text processing
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            # Tokenize the text input
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to("cuda")
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the beginning of sequence token if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        self.detected_topic = self._select_topic(self.detected_topics)
        topic_tokens = self.topic_token_mapping.get(self.detected_topic, self.vocab)
        topic_vocab_size = len(topic_tokens)
        self.expected_count = topic_vocab_size / self.vocab_size

        # call score method to evaluate the sequence
        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict
