import torch
import torch.nn as nn

class DINOv2Encoder(nn.Module):
    def __init__(self, embed_dim, unfreeze_n_blocks=0):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        if unfreeze_n_blocks > 0:
            for blk in self.backbone.blocks[-unfreeze_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

        self.proj = nn.Linear(self.backbone.embed_dim, embed_dim)

    def forward(self, x):
        feat = self.backbone(x)
        return self.proj(feat)
