import yaml
import lightning as L
from torch.utils.data import DataLoader

from datasets.entrep import ENTRepDataset
from training.biomedclip_lightning import BioMedCLIPLightning


def main():
    cfg = yaml.safe_load(open("configs/biomedclip.yaml"))

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

    model = BioMedCLIPLightning(cfg)

    trainer = L.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto",
        precision=16
    )

    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
