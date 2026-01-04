import yaml
import lightning as L
from torch.utils.data import DataLoader

from datasets.entrep import ENTRepDataset
from training.medclip_lightning import MedCLIPLightning


def main():
    cfg = yaml.safe_load(open("configs/medclip.yaml"))

    dataset = ENTRepDataset(
        cfg["data"]["annotation_file"],
        cfg["data"]["image_root"]
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"]
    )

    model = MedCLIPLightning(cfg)

    trainer = L.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto",
        precision=16    # RẤT NÊN cho MedCLIP
    )

    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
