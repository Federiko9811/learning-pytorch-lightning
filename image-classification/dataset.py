import lightning as pl
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch


class CIFARDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CIFAR-10 normalization values
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        # Download data (called only once and on single GPU)
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        # Setup datasets for different stages
        if stage == "fit":
            entire_dataset = datasets.CIFAR10(
                root=self.data_dir, train=True, download=False, transform=self.transform
            )

            # Calculate splits based on dataset length
            total_size = len(entire_dataset)
            train_size = int(0.9 * total_size)  # 90% for training
            val_size = total_size - train_size  # 10% for validation

            self.train_ds, self.val_ds = random_split(
                entire_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test":
            self.test_ds = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=False,
                transform=self.transform,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
