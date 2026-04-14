import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from evaluate import load as load_metric
from bert_score import score as bertscore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TOKENIZER_NAME = "google/gemma-7b"
prompt_tok = AutoTokenizer.from_pretrained(PROMPT_TOKENIZER_NAME)
bertscore = load_metric("bertscore")

def norm(s: str) -> str:
    return " ".join((s or "").strip().split())

c4 = load_dataset("json", data_files="../c4/c4.json", split="train")
print(f"Loaded C4 with {len(c4)} rows")

prompt2ref = {}
for row in c4:
    full_text = row["text"]
    tok_ids = prompt_tok.encode(full_text, add_special_tokens=False)
    p_ids = tok_ids[:100]
    r_ids = tok_ids[100:]
    p_txt = prompt_tok.decode(p_ids, clean_up_tokenization_spaces=True)
    r_txt = prompt_tok.decode(r_ids, clean_up_tokenization_spaces=True) if r_ids else ""
    key = norm(p_txt)
    if key and key not in prompt2ref:
        prompt2ref[key] = r_txt

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

SIZES = [4, 8, 16, 32]
IN_FMT = ""
OUT_FMT = ""

for size in SIZES:
    infile = IN_FMT.format(size=size)
    outfile = OUT_FMT.format(size=size)

    if not os.path.exists(infile):
        print(f"Missing {infile}, skipping.")
        continue

    entries = load_jsonl(infile)

    preds, refs, ids = [], [], []
    not_found = 0
    for e in entries:
        pr = norm(e.get("prompt", ""))
        ref = prompt2ref.get(pr, "")
        if ref == "":
            not_found += 1
            ref = "."
        preds.append(e.get("generated", "") or "")
        refs.append(ref)
        ids.append(e.get("id", len(ids)))
    bs = bertscore.compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type="microsoft/deberta-base-mnli",
        rescale_with_baseline=False,
        device=DEVICE,
        batch_size=16
    )

    P = bs["precision"]
    R = bs["recall"]
    F = bs["f1"]

    per_sample = []
    for i, (p, r, f) in enumerate(zip(P, R, F)):
        per_sample.append({
            "id": ids[i],
            "bertscore_precision": float(p),
            "bertscore_recall":    float(r),
            "bertscore_f1":        float(f),
        })

    summary = {
        "mean_bertscore_precision": float(sum(P)/len(P)) if P else 0.0,
        "mean_bertscore_recall":    float(sum(R)/len(R)) if R else 0.0,
        "mean_bertscore_f1":        float(sum(F)/len(F)) if F else 0.0,
        "prompts_not_found":        not_found,
    }
