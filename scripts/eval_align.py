import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets.entrep import ENTRepDataset
from models.align import ALIGNModel
from evaluation.retrieval_evaluation import evaluate_retrieval, print_results

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ENTRepDataset(
    "data/processed/annotations.json",
    "data/processed/images"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

model = ALIGNModel(
    image_backbone="resnet50",
    text_backbone="bert-base-uncased",
    embed_dim=256
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

results = evaluate_retrieval(
    model=model,
    dataloader=loader,
    tokenizer=tokenizer,
    device=device
)

print_results(results)
