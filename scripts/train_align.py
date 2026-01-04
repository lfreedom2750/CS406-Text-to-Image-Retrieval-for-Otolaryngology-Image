import yaml
import lightning as L
from torch.utils.data import DataLoader

from datasets.entrep import ENTRepDataset
from training.align_lightning import ALIGNLightning


def main():
    cfg = yaml.safe_load(open("configs/align.yaml"))

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

    model = ALIGNLightning(cfg)

    trainer = L.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto"
    )

    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
