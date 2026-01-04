"""
Official BioMedCLIP wrapper (OpenCLIP)
Compatible with retrieval_evaluation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class BioMedCLIPWrapper(nn.Module):
    """
    Adapter-compatible BioMedCLIP (OpenCLIP official)

    Required methods:
    - encode_image(images)
    - encode_text(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name="BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        pretrained="openai"
    ):
        super().__init__()

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )

    def encode_image(self, images):
        emb = self.model.encode_image(images)
        return F.normalize(emb, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        # OpenCLIP ignores attention_mask
        emb = self.model.encode_text(input_ids)
        return F.normalize(emb, dim=-1)
