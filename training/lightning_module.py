import lightning as L
from transformers import AutoTokenizer
from models.nanoclip.nanoclip import NanoCLIPModel
from losses.contrastive import ContrastiveLoss

class NanoCLIPLightning(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = NanoCLIPModel(cfg)
        self.loss_fn = ContrastiveLoss(cfg["model"]["temperature"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["text_encoder"]
        )
        self.lr = cfg["training"]["lr"]

    def training_step(self, batch, _):
        tok = self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        img_emb, txt_emb = self.model(
            batch["image"], tok["input_ids"], tok["attention_mask"]
        )
        loss = self.loss_fn(img_emb, txt_emb)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
