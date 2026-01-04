import torch.nn as nn
from transformers import AutoModel

class MiniLMEncoder(nn.Module):
    def __init__(self, model_name, embed_dim, unfreeze_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        for p in self.encoder.parameters():
            p.requires_grad = False

        if unfreeze_layers > 0:
            for layer in self.encoder.encoder.layer[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0]
        return self.proj(cls)
