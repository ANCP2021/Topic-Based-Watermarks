import json
from collections import Counter
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud

INPUT_FILE = ""
LABEL = "TBW GEMMA-7B δ=3.0"
TOP_N = 40
OUTPUT_PDF = ""

def iter_generated_texts(jsonl_path):
    path = Path(jsonl_path)
    if not path.exists():
        print(f"[WARN] File not found: {jsonl_path}")
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("generated", "")
            if isinstance(text, str) and text:
                yield text

def tokenize_space_lower_nopunct(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def word_counts_from_file(jsonl_path):
    c = Counter()
    for generated in iter_generated_texts(jsonl_path):
        tokens = tokenize_space_lower_nopunct(generated)
        c.update(tokens)
    return c

def top_n_normalized_frequencies(counter, n):
    top = counter.most_common(n)
    total = sum(count for _, count in top)
    if total == 0:
        return {}
    return {word: count / total for word, count in top}


def process_file_to_pdf(fname, label, top_n, output_pdf):
    counts = word_counts_from_file(fname)
    freqs = top_n_normalized_frequencies(counts, top_n)

    print(f"\nFile: {fname}")
    print(f"Unique words: {len(counts)}")
    print(f"Top {top_n} preview:")
    for w, c in counts.most_common(top_n)[:10]:
        print(f"  {w}: {c}")

    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(18, 9))

        if not freqs:
            ax.axis("off")
            ax.set_title(f"{label}\n(no words found)", fontsize=18, fontweight="bold", pad=20)
        else:
            wc = WordCloud(
                width=1200,
                height=900,
                background_color="white",
                prefer_horizontal=1.0,
                normalize_plurals=False,
                collocations=False
            ).generate_from_frequencies(freqs)

            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(label, fontsize=24, fontweight="bold", pad=20)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

process_file_to_pdf(INPUT_FILE, LABEL, TOP_N, OUTPUT_PDF)

