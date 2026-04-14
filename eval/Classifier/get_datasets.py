import json
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer
from random import shuffle, seed

TOKENIZER_NAME = "bert-base-uncased"
MAX_LEN = 200
RANDOM_SEED = 42
seed(RANDOM_SEED)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def process_file(filename: Path) -> Tuple[str, List[dict]]:
    label = 0 if "NW" in filename.name else 1
    dataset_type = "test" if "testing" in filename.name else "train"

    samples = []
    with filename.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item["generated"]
            tokens = tokenizer.encode(text, truncation=True, max_length=MAX_LEN)
            truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            samples.append({"text": truncated_text, "label": label})
    return dataset_type, samples

def expected_files(strength: int) -> List[str]:
    s = str(strength)
    return [
        f"bias_{s}/generated_texts_gemma_NW_{s}_1000_2000.jsonl",
        f"bias_{s}/generated_texts_gemma_TBW_{s}_0_1000.jsonl",
        f"bias_{s}/generated_texts_gemma_NW_{s}_500_testing.jsonl",
        f"bias_{s}/generated_texts_gemma_TBW_{s}_500_testing.jsonl",
    ]

def load_and_save_for_strength(strength: int, out_prefix: str = "bert"):
    files = [Path(p) for p in expected_files(strength)]
    # existence check helps catch typos
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for strength {strength}: {missing}")

    train_data, test_data = [], []

    for file in files:
        dataset_type, samples = process_file(file)
        if dataset_type == "train":
            train_data.extend(samples)
        else:
            test_data.extend(samples)

    shuffle(train_data)
    shuffle(test_data)

    out_train = f"{out_prefix}_train_gemmma_s{strength}.jsonl"
    out_test  = f"{out_prefix}_test_gemma_s{strength}.jsonl"

    with open(out_train, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample) + "\n")

    with open(out_test, "w", encoding="utf-8") as f:
        for sample in test_data:
            f.write(json.dumps(sample) + "\n")

    def counts(ds):
        n0 = sum(1 for x in ds if x["label"] == 0)
        n1 = len(ds) - n0
        return n0, n1

    tr0, tr1 = counts(train_data)
    te0, te1 = counts(test_data)

    print(f"[s={strength}] Training set size: {len(train_data)} (NW={tr0}, TBW={tr1})")
    print(f"[s={strength}] Testing  set size: {len(test_data)} (NW={te0}, TBW={te1})")
    print(f"Saved: {out_train}, {out_test}")

    
load_and_save_for_strength(2, out_prefix="bert")
load_and_save_for_strength(3, out_prefix="bert")
