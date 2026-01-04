import torch
import open_clip
from torch.utils.data import DataLoader

from datasets.entrep import ENTRepDataset
from models.biomedclip import BioMedCLIPWrapper
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

model = BioMedCLIPWrapper(
    model_name="BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
).to(device)

tokenizer = open_clip.get_tokenizer(
    "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

results = evaluate_retrieval(
    model=model,
    dataloader=loader,
    tokenizer=tokenizer,
    device=device
)

print_results(results)
