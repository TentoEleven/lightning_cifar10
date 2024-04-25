from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning.pytorch as pl
import os


class DataAugmentation():
    """Data Augmentation Transformations for the datasets."""

    def __init__(self, image_size: int=224):
        self.image_size = image_size
        self.jitter = T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    # Transformations for the train set.
    def train_set(self):
        return A.Compose([
            A.HorizontalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=[0.5, 0.5, 0.5], mask_fill_value=None, p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
            ])
    
    # Transformations for the validation set.
    def val_set(self):
        return A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
            ])


class CIFAR10DataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for the CIFAR10 dataset."""
    
    def __init__(self, data_dir: str='datasets', batch_size: int = 32, num_workers: int=os.cpu_count()):
        super().__init__()
        self.data_dir = data_dir
        self.dl_dict = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        self.transform = DataAugmentation(image_size=224)

    # Download the dataset
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    # Assign train/val datasets
    def setup(self, stage: str):
        self.train_data = CIFAR10(self.data_dir, train=True, transform=self.transform.train_set())
        self.val_data = CIFAR10(self.data_dir, train=False, transform=self.transform.val_set())
        self.classes = self.train_data.classes

    # Create train dataloader
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, **self.dl_dict, shuffle=True)

    # Create validation dataloader
    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, **self.dl_dict, shuffle=False)
