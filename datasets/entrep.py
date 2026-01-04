import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ENTRepDataset(Dataset):
    def __init__(self, ann_path, img_root):
        with open(ann_path, "r") as f:
            self.samples = json.load(f)

        self.img_root = img_root
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(f"{self.img_root}/{s['image']}").convert("RGB")
        return {
            "image": self.transform(img),
            "text": s["caption"]
        }
