"""
Retrieval Evaluation for Image–Text Models
=========================================

Supports:
- Text → Image retrieval
- Image → Text retrieval

Metrics:
- Recall@K
- mAP
- nDCG@K

Assumption:
- 1-to-1 ground truth alignment
  (image_i ↔ text_i)
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List


# =========================================================
# METRICS
# =========================================================

def recall_at_k(sim: np.ndarray, k: int) -> float:
    """
    sim[i, j]: similarity(query_i, target_j)
    """
    ranks = np.argsort(-sim, axis=1)
    gt = np.arange(sim.shape[0])
    hits = [gt[i] in ranks[i, :k] for i in range(len(gt))]
    return float(np.mean(hits))


def average_precision(sim_row: np.ndarray, gt_index: int) -> float:
    ranked = np.argsort(-sim_row)
    for r, idx in enumerate(ranked, start=1):
        if idx == gt_index:
            return 1.0 / r
    return 0.0


def mean_average_precision(sim: np.ndarray) -> float:
    aps = [
        average_precision(sim[i], i)
        for i in range(sim.shape[0])
    ]
    return float(np.mean(aps))


def ndcg_at_k(sim: np.ndarray, k: int) -> float:
    scores = []
    for i in range(sim.shape[0]):
        ranked = np.argsort(-sim[i])[:k]
        if i in ranked:
            rank = np.where(ranked == i)[0][0] + 1
            scores.append(1.0 / np.log2(rank + 1))
        else:
            scores.append(0.0)

    ideal = 1.0 / np.log2(2)
    return float(np.mean(scores) / ideal)


# =========================================================
# EMBEDDING & SIMILARITY
# =========================================================

@torch.no_grad()
def encode_dataset(
    model,
    dataloader,
    tokenizer,
    device
):
    """
    Encode all images and texts into embeddings
    """
    model.eval()

    img_embs = []
    txt_embs = []

    for batch in tqdm(dataloader, desc="Encoding"):
        images = batch["image"].to(device)

        tok = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        img_e, txt_e = model(
            images,
            tok["input_ids"],
            tok["attention_mask"]
        )

        img_embs.append(img_e.cpu())
        txt_embs.append(txt_e.cpu())

    img_embs = torch.cat(img_embs).numpy()
    txt_embs = torch.cat(txt_embs).numpy()

    return img_embs, txt_embs


def compute_similarity(
    img_embs: np.ndarray,
    txt_embs: np.ndarray,
    direction: str
) -> np.ndarray:
    """
    direction:
        - "t2i": text → image
        - "i2t": image → text
    """
    if direction == "t2i":
        return txt_embs @ img_embs.T
    elif direction == "i2t":
        return img_embs @ txt_embs.T
    else:
        raise ValueError("direction must be 't2i' or 'i2t'")


# =========================================================
# MAIN EVALUATION
# =========================================================

def evaluate_retrieval(
    model,
    dataloader,
    tokenizer,
    device,
    ks: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    End-to-end retrieval evaluation
    """

    img_embs, txt_embs = encode_dataset(
        model, dataloader, tokenizer, device
    )

    results = {}

    for direction in ["t2i", "i2t"]:
        sim = compute_similarity(img_embs, txt_embs, direction)

        for k in ks:
            results[f"{direction.upper()}_R@{k}"] = recall_at_k(sim, k)

        results[f"{direction.upper()}_mAP"] = mean_average_precision(sim)
        results[f"{direction.upper()}_nDCG@10"] = ndcg_at_k(sim, 10)

    return results


# =========================================================
# PRETTY PRINT
# =========================================================

def print_results(results: Dict[str, float]):
    print("\n==== Retrieval Evaluation ====\n")
    for k, v in results.items():
        print(f"{k:15s}: {v:.4f}")
