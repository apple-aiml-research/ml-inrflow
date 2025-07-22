#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import os
import torch
from PIL import Image
from einops import rearrange
from typing import Any, Optional

from torchvision import transforms
from torch.utils.data import Dataset
from lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)


"""
We assume that the dataset is in the following format:
- path/
    - image1.jpg
    - image2.jpg
    - ...
"""


def _getdata(path):
    all_files = [
        f for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")
    ]
    return all_files


def convert_to_coord_format(h, w, device="cpu"):
    x_channel = torch.linspace(0, 1, w, device=device).view(1, 1, -1).repeat(1, w, 1)
    y_channel = torch.linspace(0, 1, h, device=device).view(1, -1, 1).repeat(1, 1, h)
    return torch.cat((x_channel, y_channel), dim=0)


class ImageDataset(Dataset):
    def __init__(
        self,
        path="./data/",
        resolution=32,
        nsamples=None,
        **kwargs,
    ):
        all_files = _getdata(path)
        self.path = path

        # List of all image paths
        self.all_files = all_files
        self.img_res = resolution
        self.nsamples = min(len(all_files), nsamples) if nsamples else len(all_files)
        print(f"Total number of samples: {self.nsamples}")

        self.transform = transforms.Compose(
            [transforms.Resize(resolution), transforms.ToTensor()]
        )

    def __len__(self):
        return self.nsamples

    def __getitem__(self, index):
        imgpath = os.path.join(self.path, self.all_files[index])
        pil_image = Image.open(imgpath)
        pil_image.load()
        img = pil_image.convert("RGB")
        img = self.transform(img)

        img = (img * 2) - 1  # [-1, 1] scale
        H, W = img.shape[1], img.shape[2]
        if self.img_res == 0:
            self.img_res = H

        coord = convert_to_coord_format(H, W)
        coord_img = torch.cat([coord, img], dim=0)
        coord_img = rearrange(coord_img, "c h w -> (h w) c")

        batch = {
            "coord": coord_img[:, :2],
            "signal": coord_img[:, 2:],
            "resolution": self.img_res,
        }

        return batch


class ImageDataModule(LightningDataModule):
    """
    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        path="./data/",
        resolution=256,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 0,
        nsamples=None,
        val_set_config=None,
        **kwargs,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.img_res = resolution
        self.nsamples = nsamples

    def setup(self, stage: Optional[str] = None) -> None:
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

        # load and split datasets only if not loaded already
        if not hasattr(self, "train_set") and not hasattr(self, "val_set"):
            self.train_set = ImageDataset(
                path=self.hparams.path,
                resolution=self.hparams.resolution,
                nsamples=self.hparams.nsamples,
            )

            n_val_samples = self.hparams.val_set_config.nsamples
            self.val_set = Subset(
                self.train_set,
                indices=torch.randperm(self.train_set.__len__())[:n_val_samples],
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
