import torch
import lightning as L
from transformers import AutoTokenizer

from models.blip import BLIPModel
from losses.contrastive import ContrastiveLoss


class BLIPLightning(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.model = BLIPModel(
            model_name=cfg["model"]["pretrained"],
            embed_dim=cfg["model"]["embed_dim"]
        )

        self.loss_fn = ContrastiveLoss(
            temperature=cfg["training"]["temperature"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["pretrained"]
        )

        self.lr = cfg["training"]["lr"]

    def training_step(self, batch, batch_idx):
        images = batch["image"]

        tok = self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        img_emb = self.model.encode_image(images)
        txt_emb = self.model.encode_text(
            tok["input_ids"],
            tok["attention_mask"]
        )

        loss = self.loss_fn(img_emb, txt_emb)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr
        )
