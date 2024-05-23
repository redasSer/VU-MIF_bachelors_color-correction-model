import os
from PIL import Image
import torch
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

class ColorCorrectionDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        gt_path = os.path.join(self.gt_dir, self.filenames[idx])

        input_image = Image.open(input_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image).to(device)
            gt_image = self.transform(gt_image).to(device)

        return input_image, gt_image