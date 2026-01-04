"""
BLIP Retrieval-style Dual Encoder
Compatible with retrieval_evaluation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BlipModel


class BLIPModel(nn.Module):
    """
    Adapter-compatible BLIP model

    Required methods:
    - encode_image(images)
    - encode_text(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name="Salesforce/blip-itm-base-coco",
        embed_dim=256
    ):
        super().__init__()

        self.blip = BlipModel.from_pretrained(model_name)

        self.vision_proj = nn.Linear(
            self.blip.vision_model.config.hidden_size,
            embed_dim
        )

        self.text_proj = nn.Linear(
            self.blip.text_model.config.hidden_size,
            embed_dim
        )

    def encode_image(self, images):
        vision_out = self.blip.vision_model(
            pixel_values=images
        )
        cls = vision_out.last_hidden_state[:, 0]
        emb = self.vision_proj(cls)
        return F.normalize(emb, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        text_out = self.blip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = text_out.last_hidden_state[:, 0]
        emb = self.text_proj(cls)
        return F.normalize(emb, dim=-1)
