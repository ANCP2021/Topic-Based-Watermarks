import sys
import os
src_path = os.path.abspath(os.path.join('..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
import torch
import copy
from functools import partial
from datasets import load_dataset
from transformers import (
    LogitsProcessorList,
) 
from topic_watermark_processor import TopicWatermarkLogitsProcessor
from semantic_topic_extension import EmbeddingMapper
from model import (
    load_model, 
    load_sentence_model, 
    load_topic_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    # 'model_name_or_path': 'facebook/opt-2.7b',
    # 'model_name_or_path': 'facebook/opt-1.3b',
#     'model_name_or_path': 'facebook/opt-125m',
#     'model_name_or_path': 'facebook/opt-6.7b',
    'model_name_or_path': 'google/gemma-7b',
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 200,
    'min_new_tokens': 195,
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'delta': 2.0, 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 4.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'topic_token_mapping': {},
    'granularity': 'kmeans',
}

topic_list = ["animals", "technology", "sports", "medicine"]
# topic_list = ["animals", "technology", "sports", "medicine", "politics", "entertainment", "education", "finance"]
# topic_list = ["animals", "technology", "sports", "medicine", "politics", "entertainment", "education", "finance", "science", "law", "food", "travel", "environment", "religion", "fashion", "history"]
# topic_list = ["animals", "technology", "sports", "medicine", "politics", "entertainment", "education", "finance", "science", "law", "food", "travel", "environment", "religion", "fashion", "history", "art", "military", "gaming", "literature", "parenting", "space", "transportation", "psychology", "agriculture", "housing", "cryptocurrency", "architecture", "economics", "fitness", "relationships", "mythology"]


model, tokenizer = load_model(args)
print("Model loaded")

sentence_model = load_sentence_model()
print("Sentence model loaded")
embedding_mapper = EmbeddingMapper(tokenizer, sentence_model)
total_tokens, vocab_embeddings = embedding_mapper.get_model_vocab_embeddings()
topic_embeddings = embedding_mapper.get_defined_topic_list_embeddings(topic_list)
topic_token_mapping = embedding_mapper.map_tokens_to_topics(total_tokens, vocab_embeddings, topic_list, topic_embeddings)
args['topic_token_mapping'] = topic_token_mapping


c4_dataset = load_dataset('json', data_files='./c4/c4.json', split='train[:2500]')
c4_iter = iter(c4_dataset.select(range(2000, 2500)))

samples_list = []
n = 500  # number of samples to create
for _ in range(n):
    entry = next(c4_iter)
    text = entry["text"]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    prompt_ids = token_ids[:100]
    # Decode the token IDs back to strings
    prompt = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=True)
    samples_list.append({'prompt': prompt})


def generate_no_watermark(prompt, args, model=None, tokenizer=None):
    gen_kwargs = {
        'max_new_tokens': args['max_new_tokens'],
        'min_new_tokens': args['min_new_tokens']
    }

    if args['use_sampling']:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args['sampling_temp'],
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

    if args['prompt_max_length']:
        pass
    elif hasattr(model.config,"max_position_embeddings"):
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

    torch.manual_seed(args['generation_seed'])

    output_without_watermark = generate_without_watermark(**tokenized_input)

    if args['decoder']:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokenized_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]

    return decoded_output_without_watermark
    
def generate_TBW(prompt, detected_topics, args, model=None, tokenizer=None, sentence_model=None):
  
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
  
    gen_kwargs = {
        'max_new_tokens': args['max_new_tokens'],
        'min_new_tokens': args['min_new_tokens']
    }
    
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

    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )

    if args['prompt_max_length']:
        pass
    elif hasattr(model.config,"max_position_embeddings"):
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

    output_with_watermark = generate_with_watermark(**tokenized_input)

    if args['decoder']:
        # need to isolate the newly generated tokens
        output_with_watermark = output_with_watermark[:,tokenized_input["input_ids"].shape[-1]:]

    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return decoded_output_with_watermark

warmup_length = 10
schemes = {
#     "TBW": generate_TBW,
    "NW": generate_no_watermark
}

warmup_length = 10
for scheme, func in schemes.items():
    input_text = samples_list[0]['prompt']
    
    if scheme == 'TBW':
        detected_topics = load_topic_model(input_text)
        args_warmup = copy.deepcopy(args)
        args_warmup['max_new_tokens'] = warmup_length
        args_warmup['min_new_tokens'] = warmup_length
        generated_text = func(input_text, detected_topics, args_warmup, model=model, tokenizer=tokenizer, sentence_model=sentence_model)
    else: 
        args_warmup = copy.deepcopy(args)
        args_warmup['max_new_tokens'] = warmup_length
        args_warmup['min_new_tokens'] = warmup_length
        generated_text = func(input_text, args_warmup, model=model, tokenizer=tokenizer)

    print(f"Done generating for scheme {scheme}")
print("Warmup done")

gen_tokens = 200
results = {scheme: [] for scheme in schemes}
for scheme, func in schemes.items():
    gen_texts = []
    print(f"Running generation for scheme: {scheme}")
    for i, sample in enumerate(samples_list):
        input_text = sample['prompt']
        detected_topics = load_topic_model(input_text)

       
        if scheme == 'TBW':
            args_iter = copy.deepcopy(args)
            args_iter['max_new_tokens'] = gen_tokens + 5
            args_iter['min_new_tokens'] = gen_tokens - 5
            generated_text = func(
                input_text,
                detected_topics,
                args_iter,
                model=model,
                tokenizer=tokenizer,
                sentence_model=sentence_model
            )
        else:
            args_iter = copy.deepcopy(args)
            args_iter['max_new_tokens'] = gen_tokens + 5
            args_iter['min_new_tokens'] = gen_tokens - 5
            generated_text = func(input_text, args_iter, model=model, tokenizer=tokenizer)

        print(i)

        gen_texts.append({
            "id": i,
            "prompt": input_text,
            "topic": detected_topics,
            "generated": generated_text
        })

    results[scheme] = gen_texts
