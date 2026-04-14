# Topic-Based Watermarks

Official implementation for our ACL Findings 2026 paper on topic-based watermarks for large language models.

## TL;DR

This repository studies watermarking for LLM-generated text, with a focus on a **Topic-Based Watermark (TBW)** scheme. TBW maps vocabulary tokens to semantic topics, biases generation toward topic-specific token sets, and detects watermark presence using a z-score test.

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2404.02138)


## Repository Overview

Core TBW implementation:

- `src/model.py` - model loading, topic extraction wrapper, generation and detection entry functions
- `src/semantic_topic_extension.py` - semantic token-to-topic mapping using sentence embeddings
- `src/topic_watermark_processor.py` - TBW logits processor and detector logic
- `src/main.py` - end-to-end demo script

Evaluation and experiments:

- `eval/generate_text.py` - dataset generation pipeline for evaluation
- `eval/detection_eval.py` - detection evaluation variants
- `eval/partition_recovery.py` - partition recovery analysis
- `eval/attacks/` - attack scripts (paraphrase/tokenization/baseline/discrete alteration)
- `eval/*.ipynb` - notebooks for perplexity, efficiency, false positives, topic shift, and z-score analyses

Additional watermark baselines:

- `src/watermark/` and `src/config/` contain implementations/configs for multiple watermarking methods.

## Quickstart (TBW End-to-End)

Run the demo:

```bash
python src/main.py
```

This script:

1. Loads an LLM and sentence embedding model
2. Builds a topic-token partition
3. Generates text with and without TBW
4. Runs watermark detection and prints z-score-based outputs

## Citation

```bibtex
@article{nemecek2024topic,
  title={Topic-based watermarks for large language models},
  author={Nemecek, Alexander and Jiang, Yuzhou and Ayday, Erman},
  journal={arXiv preprint arXiv:2404.02138},
  year={2024}
}
```
