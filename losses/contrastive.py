import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_emb, txt_emb):
        logits = img_emb @ txt_emb.T / self.temperature
        labels = torch.arange(len(img_emb)).to(img_emb.device)
        return (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        ) / 2
