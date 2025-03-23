import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, Callable, Dict, Any


class Dataset(TorchDataset):
    """
    Dataset class for image segmentation task.

    This dataset loads image and mask pairs for training a segmentation model
    to identify areas of images created with generative AI.
    """

    def __init__(
            self,
            data_dir: str,
            split_file: str,
            transforms: Optional[Callable] = None,
            return_id: bool = False
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Root directory of the dataset
            split_file: Path to the file containing image IDs for this split (train.txt or val.txt)
            transforms: Optional transforms to be applied to both image and mask
            return_id: If True, returns the image ID along with the image and mask
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.return_id = return_id

        # Read IDs from the split file
        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        # Define paths
        self.image_dir = os.path.join(data_dir, 'train', 'images')
        self.mask_dir = os.path.join(data_dir, 'train', 'masks')

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample to fetch

        Returns:
            A dictionary containing:
                - 'image': The input image as a tensor
                - 'mask': The segmentation mask as a tensor
                - 'id': The image ID (if return_id is True)
        """
        # Get the image ID
        img_id = self.ids[index]

        # Load image and mask
        img_path = os.path.join(self.image_dir, f'{img_id}')
        mask_path = os.path.join(self.mask_dir, f'{img_id}')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load as grayscale

        # Apply transforms if specified
        if self.transforms:
            transformed = self.transforms(image=np.array(image), mask=np.array(mask))
            image = transformed['image']
            mask = transformed['mask'] / 255
        else:
            # Convert to tensors by default if no transforms are provided
            image = torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0)
            mask = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0)

        # Prepare the output dictionary
        sample = {
            'image': image,
            'mask': mask,
        }

        if self.return_id:
            sample['id'] = img_id

        return sample
