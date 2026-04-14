import json
import os
import spacy
import torch
from torch.nn.functional import cosine_similarity
from postmark.models import Paragram

nlp = spacy.load("en_core_web_sm")
paragram = Paragram(ratio=0.1)

def get_similarities(list_words, text_words):
    list_words_embs = paragram.get_embeddings(list_words)
    text_words_embs = paragram.get_embeddings(text_words)
    sims = []
    for list_word_emb in list_words_embs:
        sim = cosine_similarity(text_words_embs, list_word_emb.unsqueeze(0))
        if sim.shape[0] != len(text_words):
            assert text_words_embs.shape[0] != len(text_words), f"{sim.shape[0]} != {len(text_words)}"
        sims.append(sim)
    sims = torch.stack(sims, dim=0).cpu()
    topk_scores, topk_indices = torch.topk(sims, 1, dim=1)
    topk_words = [[text_words[i] for i in indices] for indices in topk_indices]
    return topk_words, topk_scores


def compute_presence(text, words, threshold=0.7):
    text_words = [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_space]
    topk_words, topk_scores = get_similarities(words, text_words)
    present = 0
    for w, s in zip(words, topk_scores):
        if w.lower() in text_words or s[0] >= threshold:
            present += 1
    presence = present / len(words)
    return presence