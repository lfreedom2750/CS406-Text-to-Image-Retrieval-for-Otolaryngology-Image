"""
Official MedCLIP wrapper
Compatible with retrieval_evaluation.py
"""

import torch.nn as nn
import torch.nn.functional as F

from medclip import (
    MedCLIPModel,
    MedCLIPVisionModelViT,
    MedCLIPTextModel
)


class MedCLIPWrapper(nn.Module):
    """
    Adapter-compatible MedCLIP (official implementation)

    Required methods:
    - encode_image(images)
    - encode_text(input_ids, attention_mask)
    """

    def __init__(self):
        super().__init__()

        self.model = MedCLIPModel(
            vision_model=MedCLIPVisionModelViT(),
            text_model=MedCLIPTextModel()
        )

    def encode_image(self, images):
        emb = self.model.encode_image(images)
        return F.normalize(emb, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        emb = self.model.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return F.normalize(emb, dim=-1)
