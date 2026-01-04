import torch
import torch.nn.functional as F
from models.image_encoder import DINOv2Encoder
from models.text_encoder import MiniLMEncoder

class NanoCLIPModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img = DINOv2Encoder(
            cfg["model"]["embed_dim"],
            cfg["model"]["unfreeze_img_blocks"]
        )
        self.txt = MiniLMEncoder(
            cfg["model"]["text_encoder"],
            cfg["model"]["embed_dim"],
            cfg["model"]["unfreeze_txt_layers"]
        )

    def forward(self, image, ids, mask):
        img_emb = F.normalize(self.img(image), dim=-1)
        txt_emb = F.normalize(self.txt(ids, mask), dim=-1)
        return img_emb, txt_emb
