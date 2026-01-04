import yaml
from torch.utils.data import DataLoader
from datasets.entrep import ENTRepDataset
from training.lightning_module import NanoCLIPLightning
import lightning as L

cfg = yaml.safe_load(open("configs/nanoclip.yaml"))

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

model = NanoCLIPLightning(cfg)

trainer = L.Trainer(
    max_epochs=cfg["training"]["epochs"],
    accelerator="auto"
)

trainer.fit(model, loader)
