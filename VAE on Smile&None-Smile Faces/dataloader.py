"""
This script defines a custom dataset class `FaceDataSet` for loading face images from a specified directory.

Classes:
    - FaceDataSet: A PyTorch dataset for loading and transforming images of faces.
    
"""

import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class FaceDataSet(Dataset):
    """
    A custom dataset class for loading face images from a directory. 

    Attributes:
        main_path (str): The path to the directory containing image files.
        transform (torchvision.transforms.Compose): Transformations to apply to each image.
        image_paths (list): List of file paths for each image in the dataset.
    """

    def __init__(self, main_path, transform):
        """
        Initializes the FaceDataSet with a directory path and optional transformations.
        
        :param main_path: Path to the main directory containing images.
        :param gp: Not used in this implementation; can be ignored or used for grouping if needed.
        :param transform: Transformations to apply to the images (e.g., resizing, normalization).
        """
        self.main_path = main_path
        self.transform = transform
        self.image_paths = [os.path.join(main_path, image) for image in os.listdir(main_path)]

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        
        :return: The number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image at the specified index and applies transformations.

        :param idx: Index of the image to retrieve.
        :return: Transformed image as a tensor.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image
