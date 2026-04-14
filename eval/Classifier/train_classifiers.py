import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

def parse_filename(train_file: Path):
    m = re.match(r"bert_train_(gemma|opt)_s([23])\.jsonl$", train_file.name)
    if not m:
        raise ValueError(
            f"Train file name must look like 'bert_train_(gemma|opt)_s(2|3).jsonl', got: {train_file.name}"
        )
    return m.group(1), m.group(2)

def load_jsonl(file_path: Path):
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Expect {"text": "...", "label": 0/1}
            records.append({"text": item["text"], "label": int(item["label"])})
    return records

def tokenize_examples(examples, tokenizer, max_length=200):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT-base for watermark detection (minimal).")
    parser.add_argument("--train_file", type=str, required=True, help="Path to JSONL train file.")
    parser.add_argument("--output_root", type=str, default="models", help="Root dir for saved models.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    train_path = Path(args.train_file)
    source_model, strength = parse_filename(train_path)

    output_dir = Path(args.output_root) / f"bert_base_{source_model}_s{strength}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(train_path)
    if len(records) == 0:
        raise ValueError(f"No records found in {train_path}")

    ds_train = Dataset.from_list(records)

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding=False)

    ds_train = ds_train.map(tok, batched=True)
    ds_train = ds_train.remove_columns(["text"]).with_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        report_to="none",
        seed=args.seed,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved model to: {output_dir}")
