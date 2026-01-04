"""
ALIGN-style dual encoder
Image encoder + Text encoder
Compatible with retrieval_evaluation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import timm


# =========================================================
# IMAGE ENCODER (ALIGN-style)
# =========================================================

class ALIGNImageEncoder(nn.Module):
    def __init__(self, backbone="resnet50", embed_dim=256):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0
        )

        self.proj = nn.Linear(
            self.backbone.num_features,
            embed_dim
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.proj(feat)


# =========================================================
# TEXT ENCODER (ALIGN-style)
# =========================================================

class ALIGNTextEncoder(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        embed_dim=256
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        self.proj = nn.Linear(
            self.encoder.config.hidden_size,
            embed_dim
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0]
        return self.proj(cls)


# =========================================================
# ALIGN MODEL (Adapter-compatible)
# =========================================================

class ALIGNModel(nn.Module):
    """
    Adapter-compatible ALIGN model

    Required by evaluation:
    - encode_image(images)
    - encode_text(input_ids, attention_mask)
    """

    def __init__(
        self,
        image_backbone="resnet50",
        text_backbone="bert-base-uncased",
        embed_dim=256
    ):
        super().__init__()

        self.image_encoder = ALIGNImageEncoder(
            image_backbone, embed_dim
        )
        self.text_encoder = ALIGNTextEncoder(
            text_backbone, embed_dim
        )

    def encode_image(self, images):
        emb = self.image_encoder(images)
        return F.normalize(emb, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        emb = self.text_encoder(input_ids, attention_mask)
        return F.normalize(emb, dim=-1)
