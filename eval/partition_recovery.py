#!/usr/bin/env python3
"""
Partition recovery evaluation for Topic-Based Watermarking (TBW).
Measures how accurately an attacker can reconstruct TBW's token-to-topic partition
under various assumptions (embedding model, threshold).
Does not load full LLM weights; uses tokenizers and sentence embedding models only.

Alignment with src/semantic_topic_extension.py:
- Vocab order: list(tokenizer.get_vocab().keys()) (dict insertion order).
- Encoding: all tokens encoded as-is (no placeholder for empty/whitespace).
- Semantic pass: cosine_similarity(token_emb, topic_embs), assign if max_sim >= threshold.
- Residual: shuffle(residual_tokens) then round-robin over topics (defender seed=42, attacker=99999).
- topic_token_mapping in src is {topic: [token_id, ...]}; we track same partition plus assignment type.
"""

import argparse
import os
import random
import sys

# Allow importing from src if needed (eval is self-contained; no src imports used)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

TOPIC_LIST = ["animals", "technology", "sports", "medicine"]
DEFENDER_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFENDER_TAU = 0.7

LLM_CONFIGS = {
    "opt": "facebook/opt-6.7b",
    "gemma": "google/gemma-7b",
}

ATTACKER_EMBED_MODELS = [
    "all-MiniLM-L6-v2",   # oracle
    "all-mpnet-base-v2",  # same family, different model
    "BAAI/bge-small-en-v1.5",  # different architecture
]
ATTACKER_TAUS = [0.3, 0.5, 0.7, 0.9]

EMBED_BATCH_SIZE = 4096

def _get_vocab_tokens(tokenizer):
    """Return (token_strings, token_ids) in tokenizer dict insertion order (matches src/)."""
    vocab = tokenizer.get_vocab()
    token_strings = list(vocab.keys())
    token_ids = [vocab[t] for t in token_strings]
    return token_strings, token_ids

def _encode_vocab_in_batches(sentence_model, token_strings, batch_size=EMBED_BATCH_SIZE, progress_msg="Encoding"):
    """Encode token strings in batches; return numpy array (n_tokens, dim). Matches src: encode as-is."""
    n = len(token_strings)
    embeddings_list = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = token_strings[start:end]
        emb = sentence_model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        embeddings_list.append(emb)
        if progress_msg and (end % (batch_size * 4) == 0 or end == n):
            print(f"  {progress_msg}: {end}/{n} tokens", flush=True)
    return np.vstack(embeddings_list)

def build_partition(
    tokenizer,
    sentence_model,
    topic_list,
    threshold,
    random_seed=42,
    progress_msg="Defender partition",
):
    """
    Build token-to-topic partition with semantic vs round-robin tracking.

    NOTE: The real TBW implementation (src/semantic_topic_extension.py) randomly
    shuffles residual tokens before round-robin assignment. This means even an
    attacker with identical embedding model and threshold cannot recover residual
    assignments without knowing the random state. We simulate this by using
    different random seeds for defender (42) and attacker (99999).

    Returns:
        topic_to_tokens: dict[str, set[int]]  topic -> set of token ids
        token_to_topic: dict[int, str]        token_id -> topic
        token_assignment_type: dict[int, str] token_id -> 'semantic' | 'round_robin'
        residual_pct: float                   fraction of vocab assigned via round-robin
    """
    token_strings, token_ids = _get_vocab_tokens(tokenizer)
    n_tokens = len(token_ids)

    # Encode topics
    topic_embeddings = sentence_model.encode(topic_list, convert_to_tensor=False)
    topic_embeddings_np = np.array(topic_embeddings)

    # Encode vocabulary in batches
    vocab_embeddings_np = _encode_vocab_in_batches(
        sentence_model, token_strings, EMBED_BATCH_SIZE, progress_msg
    )

    topic_to_tokens = {t: [] for t in topic_list}
    token_to_topic = {}
    token_assignment_type = {}
    residual_token_ids = []

    # Semantic assignment pass (matches src: every token gets similarity, no special-case for empty)
    for idx in range(n_tokens):
        tid = token_ids[idx]
        vec = vocab_embeddings_np[idx : idx + 1]
        sims = cosine_similarity(vec, topic_embeddings_np).flatten()
        max_sim = sims.max()
        best_idx = sims.argmax()
        if max_sim >= threshold:
            topic = topic_list[best_idx]
            topic_to_tokens[topic].append(tid)
            token_to_topic[tid] = topic
            token_assignment_type[tid] = "semantic"
        else:
            residual_token_ids.append(tid)

    # Shuffle residual tokens (matches src/semantic_topic_extension.py) then round-robin
    rng = random.Random(random_seed)
    rng.shuffle(residual_token_ids)
    num_topics = len(topic_list)
    for i, tid in enumerate(residual_token_ids):
        topic = topic_list[i % num_topics]
        topic_to_tokens[topic].append(tid)
        token_to_topic[tid] = topic
        token_assignment_type[tid] = "round_robin"

    # Convert lists to sets for set operations later
    topic_to_tokens = {t: set(ids) for t, ids in topic_to_tokens.items()}
    n_residual = len(residual_token_ids)
    residual_pct = n_residual / n_tokens if n_tokens else 0.0

    return topic_to_tokens, token_to_topic, token_assignment_type, residual_pct

def _safe_f1(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def per_topic_metrics(defender_set, attacker_set):
    """Jaccard, Precision, Recall, F1 for one topic."""
    inter = len(defender_set & attacker_set)
    union = len(defender_set | attacker_set)
    jaccard = inter / union if union else 0.0
    precision = inter / len(attacker_set) if attacker_set else 0.0
    recall = inter / len(defender_set) if defender_set else 0.0
    f1 = _safe_f1(precision, recall)
    return jaccard, precision, recall, f1

def compute_all_metrics(
    defender_topic_to_tokens,
    defender_token_to_topic,
    defender_token_assignment_type,
    defender_residual_pct,
    attacker_topic_to_tokens,
    attacker_token_to_topic,
    topic_list,
):
    """
    Compute per-topic metrics and global residual/semantic recovery.
    Returns dict with per-topic and macro metrics, plus residual_recovery and semantic_recovery.
    """
    vocab_size = sum(len(s) for s in defender_topic_to_tokens.values())
    semantic_ids = [tid for tid, t in defender_token_assignment_type.items() if t == "semantic"]
    round_robin_ids = [tid for tid, t in defender_token_assignment_type.items() if t == "round_robin"]
    n_semantic = len(semantic_ids)
    n_residual = len(round_robin_ids)

    per_topic = []
    for topic in topic_list:
        def_set = defender_topic_to_tokens[topic]
        att_set = attacker_topic_to_tokens.get(topic, set())
        j, p, r, f1 = per_topic_metrics(def_set, att_set)
        per_topic.append({
            "topic": topic,
            "jaccard": j,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    # Macro averages
    macro_jaccard = np.mean([x["jaccard"] for x in per_topic])
    macro_precision = np.mean([x["precision"] for x in per_topic])
    macro_recall = np.mean([x["recall"] for x in per_topic])
    macro_f1 = np.mean([x["f1"] for x in per_topic])

    # Residual recovery: among defender round_robin tokens, fraction attacker puts in same topic
    if n_residual == 0:
        residual_recovery = 1.0
    else:
        same = sum(
            1 for tid in round_robin_ids
            if attacker_token_to_topic.get(tid) == defender_token_to_topic.get(tid)
        )
        residual_recovery = same / n_residual

    # Semantic recovery: among defender semantic tokens, fraction attacker puts in same topic
    if n_semantic == 0:
        semantic_recovery = 1.0
    else:
        same = sum(
            1 for tid in semantic_ids
            if attacker_token_to_topic.get(tid) == defender_token_to_topic.get(tid)
        )
        semantic_recovery = same / n_semantic

    return {
        "per_topic": per_topic,
        "macro_jaccard": macro_jaccard,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "residual_pct": defender_residual_pct,
        "residual_recovery": residual_recovery,
        "semantic_recovery": semantic_recovery,
    }

def run_for_llm(llm_name: str, model_key: str):
    """Run partition recovery evaluation for one LLM (opt or gemma)."""
    model_id = LLM_CONFIGS[model_key]
    print(f"LLM: {llm_name} ({model_id})")
    print("Loading tokenizer (no model weights)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size}", flush=True)

    # 1) Ground-truth (defender) partition
    print(f"\nBuilding defender partition (embed={DEFENDER_EMBED_MODEL}, tau={DEFENDER_TAU})...", flush=True)
    defender_model = SentenceTransformer(DEFENDER_EMBED_MODEL)
    (
        defender_topic_to_tokens,
        defender_token_to_topic,
        defender_token_assignment_type,
        defender_residual_pct,
    ) = build_partition(
        tokenizer,
        defender_model,
        TOPIC_LIST,
        DEFENDER_TAU,
        random_seed=42,
        progress_msg="Defender vocab",
    )
    print(f"  Residual token % (defender): {defender_residual_pct:.1%}", flush=True)
    del defender_model

    # 2) Attacker partitions for all configs
    configs = []
    for embed_model in ATTACKER_EMBED_MODELS:
        for tau in ATTACKER_TAUS:
            configs.append((embed_model, tau))

    all_results = []
    csv_rows = []

    for i, (att_embed, att_tau) in enumerate(configs):
        print(f"\nAttacker config {i+1}/{len(configs)}: embed={att_embed}, tau={att_tau}...", flush=True)
        att_model = SentenceTransformer(att_embed)
        (
            att_topic_to_tokens,
            att_token_to_topic,
            _,
            _,
        ) = build_partition(
            tokenizer,
            att_model,
            TOPIC_LIST,
            att_tau,
            random_seed=99999,
            progress_msg=f"Attacker {att_embed} tau={att_tau}",
        )
        del att_model

        metrics = compute_all_metrics(
            defender_topic_to_tokens,
            defender_token_to_topic,
            defender_token_assignment_type,
            defender_residual_pct,
            att_topic_to_tokens,
            att_token_to_topic,
            TOPIC_LIST,
        )

        # Oracle sanity check: same model + same tau -> semantic recovery 1.0, residual ~ 1/K
        if att_embed == DEFENDER_EMBED_MODEL and att_tau == DEFENDER_TAU:
            assert abs(metrics["semantic_recovery"] - 1.0) < 1e-9, (
                f"Oracle semantic recovery sanity check failed: {metrics['semantic_recovery']}"
            )
            print(
                f"  Oracle residual recovery: {metrics['residual_recovery']:.4f} "
                f"(expected ~{1/len(TOPIC_LIST):.4f})"
            )

        all_results.append({
            "attacker_embed_model": att_embed,
            "attacker_tau": att_tau,
            **metrics,
        })

        for pt in metrics["per_topic"]:
            csv_rows.append({
                "attacker_embed_model": att_embed,
                "attacker_tau": att_tau,
                "topic": pt["topic"],
                "jaccard": pt["jaccard"],
                "precision": pt["precision"],
                "recall": pt["recall"],
                "f1": pt["f1"],
                "residual_pct": defender_residual_pct,
                "residual_recovery": metrics["residual_recovery"],
                "semantic_recovery": metrics["semantic_recovery"],
            })

    # 3) Print table to stdout
    print(f"Results table: {llm_name} ({model_id})")
    header = (
        "attacker_embed_model          attacker_tau  "
        "macro_jaccard  macro_prec  macro_rec  macro_f1   "
        "residual_pct  resid_rec  sem_rec"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        embed_short = (r["attacker_embed_model"].split("/")[-1])[:28]
        line = (
            f"{embed_short:<28} {r['attacker_tau']:<13.2f} "
            f"{r['macro_jaccard']:.4f}     {r['macro_precision']:.4f}   "
            f"{r['macro_recall']:.4f}   {r['macro_f1']:.4f}   "
            f"{r['residual_pct']:.2%}    {r['residual_recovery']:.4f}  "
            f"{r['semantic_recovery']:.4f}"
        )
        print(line)
    print()

    # 4) Write CSV
    results_dir = os.path.join(_SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"partition_recovery_{model_key}.csv")
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}", flush=True)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Partition recovery evaluation for Topic-Based Watermarking"
    )
    parser.add_argument(
        "--model",
        choices=["opt", "gemma", "both"],
        default="both",
        help="LLM to evaluate: opt (OPT-6.7B), gemma (GEMMA-7B), or both",
    )
    args = parser.parse_args()

    if args.model == "both":
        models_to_run = [("OPT-6.7B", "opt"), ("GEMMA-7B", "gemma")]
    elif args.model == "opt":
        models_to_run = [("OPT-6.7B", "opt")]
    else:
        models_to_run = [("GEMMA-7B", "gemma")]

    for llm_name, model_key in models_to_run:
        run_for_llm(llm_name, model_key)

    print("\nDone.")


if __name__ == "__main__":
    main()
