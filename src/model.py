from topic_watermark_processor import TopicWatermarkLogitsProcessor, TopicWatermarkDetector
from semantic_topic_extension import EmbeddingMapper
from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        LogitsProcessorList
    )
import torch
from functools import partial
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

device = "cuda" if torch.cuda.is_available() else "cpu"
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

def load_sentence_model(model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model

# Loads and returns the model
def load_model(args):

    is_seq2seq_model = any([(model_type in args['model_name_or_path']) for model_type in ["t5","T0"]])
    args['seq2seq'] = is_seq2seq_model

    is_decoder_only_model = any([(model_type in args['model_name_or_path']) for model_type in ["gpt","opt","bloom","llama","gemma","starcoder"]])
    args['decoder'] = is_decoder_only_model
    
    if is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args['model_name_or_path'])
    elif is_decoder_only_model:
        if args['load_fp16']:
            model = AutoModelForCausalLM.from_pretrained(args['model_name_or_path'],torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args['model_name_or_path'], device_map='auto')
    else:
        raise ValueError(f"Unknown model type: {args['model_name_or_path']}")

    tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'])

    return model, tokenizer

# Instatiate the WatermarkLogitsProcessor according to the watermark parameters
# and generate watermarked text by passing it to the generate method of the model
# as a logits processor.
def generate(prompt, detected_topics, args, model=None, tokenizer=None, sentence_model=None):
    
    # if set to is_topic, use topic based watermarks
    detected_topic = ''
    for topic in detected_topics:
        if topic.lower() in args['topic_token_mapping']:
            detected_topic = topic.lower()
            print(f"Topic detected in one to one mapping: {detected_topic}")
            break
    if detected_topic == '':
        embedding_mapper = EmbeddingMapper(tokenizer, sentence_model)

        if args['granularity'] == 'kmeans':
            print(f"Mapping topic with granularity Kmeans.")
            detected_topic, _ = embedding_mapper.kmeans_detected_topics_to_embeddings(
                list(args['topic_token_mapping'].keys()), 
                detected_topics,
            )
        else:
            print(f"Mapping topic with granularity averaging")
            detected_topic, _ = embedding_mapper.detected_topics_to_embeddings(
                list(args['topic_token_mapping'].keys()), 
                detected_topics
            )

    print(f"Detected topic: {detected_topic}")
    watermark_processor = TopicWatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        delta=args['delta'],
        seeding_scheme=args['seeding_scheme'],
        select_green_tokens=args['select_green_tokens'],
        topic_token_mapping=args['topic_token_mapping'],
        detected_topic=detected_topic,
    )
    
  
    gen_kwargs = dict(max_new_tokens=args['max_new_tokens'])

    if args['use_sampling']:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args['sampling_temp'],
            repetition_penalty=1.1,
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args['n_beams']
        ))

    # generate without the watermark 
    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    ) 

    # generate with watermark using LogitsProcessorList
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )

    if args['prompt_max_length']:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args['prompt_max_length'] = model.config.max_position_embeddings - args['max_new_tokens']
    else:
        args['prompt_max_length'] = 2048 - args['max_new_tokens']

    tokenized_input = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=True, 
        truncation=True, 
        max_length=args['prompt_max_length']
    ).to(device)

    truncation_warning = True if tokenized_input["input_ids"].shape[-1] == args['prompt_max_length'] else False
    redecoded_input = tokenizer.batch_decode(tokenized_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args['generation_seed'])

    output_without_watermark = generate_without_watermark(**tokenized_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args['seed_separately']: 
        torch.manual_seed(args['generation_seed'])
    output_with_watermark = generate_with_watermark(**tokenized_input)

    if args['decoder']:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokenized_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokenized_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
    ) 

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dicts, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    if isinstance(score_dicts, dict):  # For backward compatibility if a single dict is passed
        score_dicts = [score_dicts]

    for score_dict in score_dicts:
        topic_scores = []    
        for k,v in score_dict.items():
            if k=='green_fraction': 
                topic_scores.append([format_names(k), f"{v:.1%}"])
            elif k=='confidence': 
                topic_scores.append([format_names(k), f"{v:.3%}"])
            elif isinstance(v, float): 
                topic_scores.append([format_names(k), f"{v:.3g}"])
            elif isinstance(v, bool):
                topic_scores.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
            else: 
                topic_scores.append([format_names(k), f"{v}"])

        if "confidence" in score_dict:
            topic_scores.insert(-2,["z-score Threshold", f"{detection_threshold}"])
        else:
            topic_scores.insert(-1,["z-score Threshold", f"{detection_threshold}"])

    lst_2d.extend(topic_scores)
    lst_2d.append([])

    return lst_2d

def detect(topic_text, input_text, args, device=None, tokenizer=None, sentence_model=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    detected_topics = load_topic_model(topic_text)
    print(f"Detected topic: {detected_topics}")
    watermark_detector = TopicWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
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

    if len(input_text)-1 > watermark_detector.min_prefix_len:
        score_dict = watermark_detector.detect(input_text)

        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output