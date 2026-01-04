import torch
import lightning as L
import open_clip

from models.biomedclip import BioMedCLIPWrapper
from losses.contrastive import ContrastiveLoss


class BioMedCLIPLightning(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.model = BioMedCLIPWrapper(
            model_name=cfg["model"]["name"],
            pretrained=cfg["model"]["pretrained"]
        )

        self.loss_fn = ContrastiveLoss(
            temperature=cfg["training"]["temperature"]
        )

        self.tokenizer = open_clip.get_tokenizer(
            cfg["model"]["name"]
        )

        self.lr = cfg["training"]["lr"]

    def training_step(self, batch, batch_idx):
        images = batch["image"]

        text_tokens = self.tokenizer(batch["text"]).to(self.device)

        img_emb = self.model.encode_image(images)
        txt_emb = self.model.encode_text(
            text_tokens, None
        )

        loss = self.loss_fn(img_emb, txt_emb)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr
        )
