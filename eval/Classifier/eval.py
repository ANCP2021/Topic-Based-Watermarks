import argparse
import json
import re
from pathlib import Path
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
from sklearn.metrics import accuracy_score, f1_score

def parse_test_filename(p: Path):
    m = re.match(r"bert_(?:test|testing)_(gemma|opt)_s([23])\.jsonl$", p.name)
    if not m:
        raise ValueError(
            f"Test file must look like 'bert_test_(gemma|opt)_s(2|3).jsonl', got: {p.name}"
        )
    return m.group(1), m.group(2)

def parse_model_dir(p: Path):
    m = re.search(r"bert_base_(gemma|opt)_s([23])$", p.as_posix())
    if not m:
        raise ValueError(
            f"Model dir must end with 'bert_base_(gemma|opt)_s(2|3)', got: {p}"
        )
    return m.group(1), m.group(2)

def load_jsonl(file_path: Path):
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append({"text": item["text"], "label": int(item["label"])})
    if not records:
        raise ValueError(f"No records found in {file_path}")
    return records

def tokenize_examples(examples, tokenizer, max_length=200):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned BERT watermark detector (minimal).")
    parser.add_argument("--test_file", type=str, required=True, help="Path to JSONL test file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model dir.")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    model_dir = Path(args.model_dir)

    test_src, test_strength = parse_test_filename(test_path)
    model_src, model_strength = parse_model_dir(model_dir)
    if (test_src, test_strength) != (model_src, model_strength):
        print(f"WARNING: mismatch test={test_src},s{test_strength} vs model={model_src},s{model_strength}")

    records = load_jsonl(test_path)
    ds = Dataset.from_list(records)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding=False)

    ds = ds.map(tok, batched=True).remove_columns(["text"]).with_format("torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=collator)

    outputs = trainer.predict(ds)
    preds = np.argmax(outputs.predictions, axis=-1)
    labels = outputs.label_ids

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")

    print(f"accuracy: {acc:.4f}")
    print(f"f1:       {f1:.4f}")
