import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets.entrep import ENTRepDataset
from models.nanoclip.nanoclip import NanoCLIPModel
from evaluation.retrieval_evaluation import evaluate_retrieval, print_results

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
dataset = ENTRepDataset(
    "data/processed/annotations.json",
    "data/processed/images"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

# Model
model = NanoCLIPModel.from_pretrained(
    "checkpoints/nanoclip.ckpt"   # hoặc load theo cách bạn đang dùng
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Eval
results = evaluate_retrieval(
    model=model,
    dataloader=loader,
    tokenizer=tokenizer,
    device=device
)

print_results(results)
