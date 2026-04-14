import torch
from model import (
    load_model, 
    load_sentence_model, 
    load_topic_model,
    generate, 
    detect
)
from semantic_topic_extension import EmbeddingMapper
from inputs import sports_input
from pprint import pprint

DEBUG = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    # 'model_name_or_path': 'facebook/opt-2.7b',
    # 'model_name_or_path': 'facebook/opt-1.3b',
    'model_name_or_path': 'facebook/opt-125m', 
    # 'model_name_or_path': 'meta-llama/Llama-3.2-1B',
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 300, 
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

if __name__ == '__main__':
    input_text = sports_input()
    topic_list = ["animals", "technology", "sports", "medicine"]

    model, tokenizer = load_model(args)
    if DEBUG: print('model loaded')
    sentence_model = load_sentence_model()
    if DEBUG: print('sentence model loaded')

    embedding_mapper = EmbeddingMapper(tokenizer, sentence_model)
    total_tokens, vocab_embeddings = embedding_mapper.get_model_vocab_embeddings()
    topic_embeddings = embedding_mapper.get_defined_topic_list_embeddings(topic_list)
    topic_token_mapping = embedding_mapper.map_tokens_to_topics(total_tokens, vocab_embeddings, topic_list, topic_embeddings)
    
    args['topic_token_mapping'] = topic_token_mapping

    detected_topics = load_topic_model(input_text)

    if DEBUG: print(f"Topic extraction is finished for watermarking: {detected_topics}")

    print(f"Prompt:\n {input_text}")

    redecoded_input, truncation_warning, decoded_output_without_watermark, decoded_output_with_watermark = generate(
        input_text, 
        detected_topics,
        args, 
        model=model, 
        tokenizer=tokenizer,
        sentence_model=sentence_model,
    )

    if DEBUG: print("Decoding with and without watermarkings are finished")

    topic_text = input_text + decoded_output_without_watermark
    without_watermark_detection_result = detect(topic_text, decoded_output_without_watermark, 
                                                args, 
                                                device=device, 
                                                tokenizer=tokenizer,
                                                sentence_model=sentence_model)

    topic_text = input_text + decoded_output_with_watermark
    with_watermark_detection_result = detect(topic_text, decoded_output_with_watermark, 
                                                args, 
                                                device=device, 
                                                tokenizer=tokenizer,
                                                sentence_model=sentence_model)
    if DEBUG: print("Finished with watermark detection")


    print("#########################################")
    print("Output without watermark:")
    print(decoded_output_without_watermark)
    print(("#########################################"))
    print(f"Detection result @ {args['detection_z_threshold']}:")
    pprint(without_watermark_detection_result)
    print(("#########################################"))

    print(("#########################################"))
    print("Output with watermark:")
    print(decoded_output_with_watermark)
    print(("#########################################"))
    print(f"Detection result @ {args['detection_z_threshold']}:")
    pprint(with_watermark_detection_result)
    print(("#########################################"))
