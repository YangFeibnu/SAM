# SAM: LLM-Based Semantic-Augmented Adaptive Multi-Interest Model for Sequential Recommendation

This repository contains the official PyTorch implementation for the paper: **SAM: LLM-Based Semantic-Augmented Adaptive Multi-Interest Model for Sequential Recommendation**.

Our paper has been submitted to The 2026 ACM Web Conference. We will update the paper link and BibTeX citation upon acceptance.

**[Paper]** (Link to be updated) | **[Code]** (https://github.com/tsinghua-fib-lab/SAM)

## Introduction

Sequential recommendation is crucial for capturing users' dynamic preferences. However, existing methods often struggle with two main challenges: effectively disentangling rich semantic interests from sparse user behaviors, and seamlessly integrating these semantic interests with traditional ID-based models.

To address these challenges, we propose **SAM**, a novel two-stage framework that leverages Large Language Models (LLMs) for semantic-augmented multi-interest learning. SAM excels at semantic understanding, applying them directly to multi-interest learning by generating a semantically-rich interest representation and employing a sophisticated fusion method to preserve the structural integrity of the ID embedding space.

<p align="center">
  <img src="https://i.imgur.com/examp.png" width="800">
  <br>
  <em>Figure 1: The overall framework of SAM.</em>
</p>

### Key Contributions:

*   **Adaptive Interest Extraction**: We introduce an Interest Group Identification Module that adaptively determines the optimal number of interests for each user, using this as a data-driven constraint to guide LLM-based semantic interest generation.
*   **Interest-Guided Representation Enhancement**: A dual-attention mechanism, including an Interest-Aware Attention and a Cross-Interest Attention mechanism, effectively fuses the generated semantic information with ID-based embeddings while preserving embedding space integrity and modeling complex interest inter-dependencies.
*   **State-of-the-Art Performance**: Extensive experiments on six benchmark datasets demonstrate that SAM significantly outperforms existing state-of-the-art baselines, especially for cold-start users and long-tail items.

## Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tsinghua-fib-lab/SAM.git
    cd SAM
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Our code is built with Python 3.11 and PyTorch 2.2.1. Install all required packages using:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

We use five sub-datasets from the **Amazon Review Data (2018)** and a large-scale **Alibaba** dataset.

### Amazon Datasets

1.  **Download**: Download the 5-core review data (`reviews_*_5.json`) and metadata (`meta_*.json`) for the following datasets:
    *   [All Beauty](https://nijianmo.github.io/amazon/index.html)
    *   [Arts, Crafts & Sewing](https://nijianmo.github.io/amazon/index.html)
    *   [Industrial & Scientific](https://nijianmo.github.io/amazon/index.html)
    *   [Musical Instruments](https://nijianmo.github.io/amazon/index.html)
    *   [Office Products](https://nijianmo.github.io/amazon/index.html)

2.  **Preprocess**: Place the downloaded files in the `Data/` directory and run the preprocessing script. This script will filter users/items with less than 5 interactions and generate a unified data format.

    For example, to process the 'All Beauty' dataset:
    ```bash
    # Make sure reviews_Beauty_5.json and meta_Beauty.json are in Data/
    python Data/DataProcessing.py --dataset Musical_Instruments
    ```
    This will generate `Data/Musical_Instruments.txt` which will be used in the training process.

## How to Run

The training process of SAM consists of two stages:

### Stage 1: Multi-Interest Extraction

In this stage, we use the Interest Group Identification Module to determine the optimal number of interests (K) for each user and then leverage a pre-trained BERT model to generate semantic interest representations.

Important: We now use a local BERT model only. Please place `bert-base-uncased` under `MyModel/LLMs/bert-base-uncased` (it should contain `config.json`, `pytorch_model.bin`, `vocab.txt`, etc.). The script loads it with `local_files_only=True` and will not download from the internet.

LLM (Qwen-Turbo) API requirements:
- We call DashScope Qwen-Turbo to generate interest texts. You must provide a valid API key.
- Preferred: export an environment variable and pass it to the script.

Setup and run (example for Beauty):
```bash
# 1) Prepare API key (replace with your own key)
export DASHSCOPE_API_KEY='sk-xxxxxxxxxxxxxxxx'

# 2) Generate interests with local BERT and Qwen-Turbo
python stage1_generate_interests.py \
  --dataset Beauty \
  --api_key "$DASHSCOPE_API_KEY" \
  --debug  # optional: prints prompts and HTTP snippets for troubleshooting
```
This will produce:
- `Interests/interests_Beauty.pt` (semantic interest vectors, dict: {user_id -> Tensor[k, 768]})
- `Interests/interests_Beauty.csv` (generated interest texts per user)

Troubleshooting:
- HTTP 401 InvalidApiKey: the key is invalid/expired or contains whitespace. Re-check and retry.
- Empty interests (K=0 printed): usually due to LLM call failure; use `--debug` to see HTTP messages.

### Stage 2: Interest-Guided Representation Enhancement

This is the main training stage. The model learns to fuse the semantic interests from Stage 1 with the user's behavior sequence.

Run (example for Beauty):
```bash
python main.py \
  --dataset Beauty \
  --epochs 400 \
  --batch_size 1024 \
  --learning_rate 0.0005 \
  --item_dim 256 \
  --max_seq_len 20
```
The script will automatically read:
- `Data/Beauty.txt`
- `Interests/interests_Beauty.pt`

Key Notes:
- `--item_dim`: ID embedding dimension (default 256)
- The BERT output dimension used in Stage1 is 768 (bert-base-uncased)
- Evaluation: sampled ranking (1 positive + 100 negatives), reporting HR@20/NDCG@20/MRR@20

## Performance

SAM achieves state-of-the-art performance across all six datasets. The table below summarizes the main results (MRR@20, NDCG@20).

| Dataset                | Metric  | STOSA | BSARec  | SINE   | EIMF   | **SAM (Ours)** | **Improvement** |
| ---------------------- | ------- | ------ | ------ | ---------- | ------ | ------ | -------------- | --------------- |
| **All Beauty**         | MRR@20   |0.2180 |  0.2026 |  0.2463 |  0.2535 | **0.2615**     | **+3.14%**      |
|                        | NDCG@20 |  0.2853 |  0.2764 | 0.2985 | 0.3085| **0.3266**     | **+5.85%**      |
| **Arts Crafts & Sewing** | MRR@20   |  0.2885 |  0.3221 | 0.3159 |  0.3238| **0.3251**     | **+0.40%**      |
|                        | NDCG@20 |  0.3495|  0.3911 |  0.3907 | 0.3858 | **0.3936**     | **+0.64%**      |
| **Industrial & Scientific** | MRR@20   |  0.2136 |  0.1933 |  0.2195 | 0.1833 | **0.2225**     | **+1.37%**      |
|                        | NDCG@20 |  0.2689 |  0.2587 |  0.2858 | 0.2512 | ** 0.2873**     | **+0.52%**      |
| **Musical Instruments**  | MRR@20   |  0.2823 |  0.2814 |  0.2794 | 0.2811 | ** 0.2831**     | **+0.28%**      |
|                        | NDCG@20 | 0.3767 | 0.3512 | 0.3314 | 0.3447 | **0.3575**     | **+1.79%**      |
| **Office Products**      | MRR@20   | 0.2887 | 0.2864 | 0.2842 |  0.2883 | **0.2921**     | **+1.18%**      |
|                        | NDCG@20 | 0.3597 | 0.3582 | 0.3562 | 0.3521 | **0.3609**     | **+0.33%**      |
| **Alibaba**              | MRR@20   | 0.5327 | 0.4695 | 0.4031 | 0.4720 | **0.5709**     | **+7.17%**      |
|                        | NDCG@20 | 0.5982 | 0.5209 | 0.4416 | 0.5251 | **0.6024**     | **+0.70%**      |

## Citation

If you find our work useful for your research, please consider citing our paper. The BibTeX entry will be provided upon publication.

```bibtex
@inproceedings{yang2026sam,
  title={SAM: LLM-Based Semantic-Augmented Adaptive Multi-Interest Model for Sequential Recommendation},
  author={Yang, Fei and Liu, Bin and Xu, Ziru and Zhu, Han and Gao, Chen and Li, Yongli},
  booktitle={Proceedings of the ACM web conference 2026},
  year={2026}
}
```

## Acknowledgements

Our implementation for the dynamic routing mechanism is inspired by the official code of [MIND](https://github.com/alibaba/MIND).

