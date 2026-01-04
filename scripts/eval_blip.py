import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets.entrep import ENTRepDataset
from models.blip import BLIPModel
from evaluation.retrieval_evaluation import evaluate_retrieval, print_results

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ENTRepDataset(
    "data/processed/annotations.json",
    "data/processed/images"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)

model = BLIPModel(
    model_name="Salesforce/blip-itm-base-coco",
    embed_dim=256
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/blip-itm-base-coco"
)

results = evaluate_retrieval(
    model=model,
    dataloader=loader,
    tokenizer=tokenizer,
    device=device
)

print_results(results)
