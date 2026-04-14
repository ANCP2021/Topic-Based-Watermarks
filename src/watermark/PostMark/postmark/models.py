import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import json
import re
import spacy
import nltk
import time
import random
from collections import Counter
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from scipy.stats import kendalltau
import numpy as np
import pickle
import tiktoken
import pdb

torch.manual_seed(42)
random.seed(42)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class EmbeddingModel():
    def __init__(self, ratio=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ratio = ratio
        self.nlp = spacy.load("en_core_web_sm")
        self.word2idx = {}
        self.idx2word = []
        self.embedding_table = torch.zeros(1, 300).to(self.device)
        self.freq_dict = json.load(open("wikitext_freq.json", 'r'))
        self.tagger = nltk.PerceptronTagger()
    
    def get_embedding(self, word):
        if word in self.word2idx:
            return self.embedding_table[self.word2idx[word]]
        else:
            return None

    def get_embeddings(self, words):
        indices = []
        none_indices = []
        for i, word in enumerate(words):
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                none_indices.append(i)
        embeddings = self.embedding_table[indices, :]
        if none_indices:
            for i in none_indices:
                t = torch.zeros(1, self.embedding_table.shape[1]).to(self.device)
                embeddings = torch.cat((embeddings[:i], t, embeddings[i:]), dim=0)
        assert embeddings.shape[0] == len(words), f"embeddings shape mismatch with words size: {embeddings.shape[0]} != {len(words)}"
        return embeddings
    
    def get_doc_embedding(self, text):
        doc = self.nlp(text)
        words = [token.text for token in doc if not token.is_stop and token.text in self.word2idx]
        embeddings = self.get_embeddings(words)
        return embeddings.mean(dim=0).unsqueeze(0).to(self.device)
    
    def get_word(self, embedding):
        idx = (self.embedding_table == embedding).all(dim=1).nonzero(as_tuple=True)[0]
        return self.idx_to_word(idx)
    
    def idx_to_word(self, idx):
        return self.idx2word[idx]
    
    def word_to_idx(self, word):
        return self.word2idx[word]

    def get_words(self, text):
        k = int(len(text.split()) * self.ratio)
        response_vec = self.get_doc_embedding(text)
        if isinstance(response_vec, list):
            response_vec = torch.tensor(response_vec).to(self.device)

        scores = cosine_similarity(self.embedding_table, response_vec, dim=1)
        assert scores.shape[0] == len(self.idx2word), f"scores shape mismatch with idx2word size: {scores.shape[0]} != {len(self.idx2word)}"
        top_k_scores, top_k_indices = torch.topk(scores, k * 3)
        
        top_k_words = [self.idx_to_word(index.item()) for index in top_k_indices]
        words = top_k_words
        try:
            word_embs = self.get_doc_embedding(words)
        except:
            return words
        # Avoid re-wrapping tensors to silence torch warning
        if not isinstance(word_embs, torch.Tensor):
            word_embs = torch.tensor(word_embs)
        word_embs = word_embs.to(self.device)
        text_emb = response_vec.unsqueeze(0).to(self.device)
        scores = cosine_similarity(text_emb, word_embs, dim=1)

        top_k_scores, top_k_indices = torch.topk(scores, k)
        words = [words[idx] for idx in top_k_indices]
        words = [word.lower() for word in words]
        words = sorted(list(set(words)))
        return words

class NomicEmbed(EmbeddingModel):
    def __init__(self, model="nomic-embed-text-v1", word=False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.word = word
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.embedder = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True).to(self.device)
        self.embedder.eval()
        if not self.word:
            wpath = "valid_wtmk_words_in_wiki_base-only-f1000.pkl"  # only use base forms of nouns, verbs, adjectives, adverbs
            words = pickle.load(open(wpath, 'rb'))
            redpj_embs = pickle.load(open("filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl", 'rb'))
            indices = random.sample(range(len(redpj_embs)), len(words))
            # Avoid tensor-from-tensor warning by checking type
            emb_list = [
                (redpj_embs[i].clone().detach() if isinstance(redpj_embs[i], torch.Tensor) else torch.tensor(redpj_embs[i]))
                for i in indices
            ]
            random.shuffle(emb_list)
            emb_table = torch.stack(emb_list)
            self.embedding_table = emb_table.to(self.device)
            print(f"Embedding table shape: {self.embedding_table.shape}")
            self.word2idx = {word: idx for idx, word in enumerate(words)}
            self.idx2word = words
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_doc_embedding(self, text):
        if type(text) == str:
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.embedder(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)[0]
        elif type(text) == list:
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.embedder(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class Paragram(EmbeddingModel):
    def __init__(self, filter_vocab=0, **kwargs):
        super().__init__(**kwargs)
        words={}
        We = []
        We = pickle.load(open("paragram_xxl.pkl", 'rb'))
        words = json.load(open("paragram_xxl_words.json", 'r'))
        self.word2idx = words
        self.idx2word = list(self.word2idx.keys())
        self.embedding_table = We.to(self.device)
        indices = list(self.word2idx.values())


class Watermarker():
    def __init__(self, llm, embedder, inserter, ratio=0.1, iterate=None):
        self.llm = LLM(llm)
        self.inserter = LLM(inserter)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoding = tiktoken.get_encoding('cl100k_base')
        print(f"Loading {embedder} embedder...")
        if embedder == 'nomic':
            self.embedder = NomicEmbed(ratio=ratio)
        else:
            raise ValueError("Only 'nomic' embedder is enabled in this setup.")
        print("Loaded embedding model.")
        with open(f"prompts/insert.txt", 'r') as f:
            self.watermark_template = f.read()
        self.iterate = iterate

    def get_words(self, text):
        words = self.embedder.get_words(text)
        return words
    
    def insert_watermark(self, text1, max_tokens=600):
        list1 = self.get_words(text1)
        if len(list1) == 0:
            print("No words found in text, returning...")
            return {"text1": text1, "list1": list1, "text2": text1, "list2": list1}
        
        if self.iterate:
            if self.iterate == "v2":
                sublists = [list1[i:i+10] for i in range(0, len(list1), 10)]
                input_res = text1
                for sublist in sublists:
                    init_words_str = ", ".join(sublist)
                    new_prompt = self.watermark_template.format(input_res, init_words_str)
                    sub_presence = 0
                    n_attempts = 0
                    while sub_presence < 0.5:
                        if n_attempts == 3:
                            print(f"Exceeded 3 tries, breaking...sub_presence: {sub_presence}")
                            break
                        input_res = self.inserter.generate(new_prompt, max_tokens=max_tokens, temperature=0)
                        sub_presence = sum([1 for word in sublist if word.lower() in input_res.lower()]) / len(sublist)
                        n_attempts += 1
                text2 = input_res
                presence = sum([1 for word in list1 if word.lower() in text2.lower()]) / len(list1)
            else:
                raise ValueError
        else:
            init_words_str = ", ".join(list1)
            new_prompt = self.watermark_template.format(text1, init_words_str)
            text2 = self.inserter.generate(new_prompt, max_tokens=max_tokens, temperature=0)
        
        list2 = self.get_words(text2)
        res = {
            "text1": text1,
            "list1": list1,
            "text2": text2,
            "list2": list2
        }
        return res


class LLM():
    def __init__(self, model):
        if model == "llama-3-8b":
            self.model = Llama3_8B()
        elif model == "llama-3-8b-chat":
            self.model = Llama3_8B_Chat()
        elif model == "mistral-7b-inst":
            self.model = Mistral_7B_Inst()
        else:
            raise ValueError("Supported models: 'llama-3-8b', 'llama-3-8b-chat', 'mistral-7b-inst'")
    
    def generate(self, prompt, max_tokens=600, temperature=1):
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature)

class Llama3_8B():
    def __init__(self, half=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto", torch_dtype=torch.float16)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1,
        num_return_sequences=1,
        logits_processor=None
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        generation_args = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id,
        }
        # Handle deterministic decoding when temperature <= 0
        if temperature is not None and temperature > 0:
            generation_args["temperature"] = temperature
            generation_args["do_sample"] = True if do_sample is None else do_sample
        else:
            generation_args["do_sample"] = False
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return output_text


class Llama3_8B_Chat():
    def __init__(self, half=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.float16)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1,
        num_return_sequences=1,
        logits_processor=None
    ):
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        generation_args = {
            "eos_token_id": eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id
        }
        # Handle deterministic decoding when temperature <= 0
        if temperature is not None and temperature > 0:
            generation_args["temperature"] = temperature
            generation_args["do_sample"] = True if do_sample is None else do_sample
        else:
            generation_args["do_sample"] = False
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        # Build an explicit attention mask (no padding is applied here)
        attention_mask = torch.ones_like(inputs).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(inputs, attention_mask=attention_mask, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return output_text


class Mistral_7B_Inst():
    def __init__(self, half=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", torch_dtype=torch.float16)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
            self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1,
        num_return_sequences=1,
        logits_processor=None
    ):
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        generation_args = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id
        }
        # Handle deterministic decoding when temperature <= 0
        if temperature is not None and temperature > 0:
            generation_args["temperature"] = temperature
            generation_args["do_sample"] = True if do_sample is None else do_sample
        else:
            generation_args["do_sample"] = False
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        # Build an explicit attention mask (no padding is applied here)
        attention_mask = torch.ones_like(inputs).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(inputs, attention_mask=attention_mask, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return output_text
