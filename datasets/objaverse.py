#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from typing import Any, Optional


"""
We assume that the dataset is in the following format:
- path/
    - train.json
    - valid.json
    - test.json
    - data/
        - 000-000-000.npz
        - 000-000-001.npz
        - ...
Each npz file contains:
    - xyz: (N, 3) coordinates of the points
    - cond_token: (576, 774) image condition tokens from DINOv2 and ray embedding
"""


class ObjaverseDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = 'train',
    ):
        self.path = path
        with open(os.path.join(path, f'{split}.json'), 'r') as f:
            self.all_files = json.load(f)
        print(f'Loaded {len(self.all_files)} files from {path} for split {split}')

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.path, 'data', self.all_files[idx]), allow_pickle=True)

        return {
            'coord': data['xyz'],
            'signal': data['xyz'],
            'label': data['cond_token'],
        }


class ObjaverseDataModule(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers,
        batch_size,
        split,
        val_set_config,
        pin_memory=True,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.path = path

        # List of all image paths
        self.split = split

    def setup(self, stage: str) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )
            self.batch_size_per_device_test = (
                self.hparams.val_set_config.batch_size // self.trainer.world_size
            )

            self.train_set = ObjaverseDataset(
                path=self.hparams.path,
                split='train',
            )
            self.val_set = ObjaverseDataset(
                path=self.hparams.path,
                split='valid',
            )
            self.test_set = ObjaverseDataset(
                path=self.hparams.path,
                split='test',
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size_per_device_test,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
