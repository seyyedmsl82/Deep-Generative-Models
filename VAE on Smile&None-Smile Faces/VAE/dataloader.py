import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class FaceDataSet(Dataset):
    def __init__(self, main_path, gp, transform):
        self.main_path = main_path
        self.transform = transform
        self.image_paths = [os.path.join(main_path, image) for image in os.listdir(main_path)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image
