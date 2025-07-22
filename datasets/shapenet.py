#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import os
import h5py
import random
import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
from copy import copy
from typing import Any, Optional


"""
We assume that the dataset is in the following format:
- path/
    - shapenet.hdf5
    - test_data/
        - PF2_val_all.pt
        - ref_val_airplane.pt
        - ref_val_chair.pt
        - ref_val_car.pt
Checkpoints for test data are accessed from https://huggingface.co/xiaohui2022/lion_ckpt/tree/main
"""


synsetid_to_cate = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02747177": "can",
    "02942699": "camera",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "04379243": "table",
    "04401088": "telephone",
    "02946921": "tin_can",
    "04460130": "tower",
    "04468005": "train",
    "03085013": "keyboard",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04530566": "vessel",
    "04554684": "washer",
    "02992529": "cellphone",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetLION1000(Dataset):
    def __init__(self, path, category, **kwargs):
        super().__init__()

        if category[0] == "all":
            self.data = torch.load(
                os.path.join(path, "test_data/PF2_val_all.pt")
            )
        else:
            self.data = torch.load(
                os.path.join(path, f"test_data/ref_val_{category[0]}.pt")
            )

    def __getitem__(self, idx):
        pc = self.data["ref"][idx][:, :3]

        pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
        pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
        shift = ((pc_min + pc_max) / 2).view(1, 3)
        scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale

        batch = {
            "coord": pc,
            "signal": pc,
        }

        return batch

    def __len__(self):
        return self.data["ref"].shape[0]


class ShapeNetDataset(Dataset):

    GRAVITATIONAL_AXIS = 1

    def __init__(
        self,
        path,
        category,
        split,
        scale_mode,
        **kwargs,
    ):
        super().__init__()
        # assert isinstance(category, list), '`category` must be a list of cate names.'
        assert split in ("train", "val", "test")
        assert scale_mode is None or scale_mode in (
            "global_unit",
            "shape_unit",
            "shape_bbox",
            "shape_half",
            "shape_34",
        )

        self.path = path
        if "all" in category:
            category = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in category]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode

        self.pointclouds = []
        self.stats = None

        self.get_statistics()
        self.load()

    def get_statistics(self):

        path = os.path.join(self.path, "shapenet.hdf5")

        basename = os.path.basename(path)
        dsetname = basename[: basename.rfind(".")]
        stats_dir = os.path.join(os.path.dirname(path), dsetname + "_stats")
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, "stats_all.pt")
        else:
            stats_save_path = os.path.join(
                stats_dir, "stats_" + "_".join(self.cate_synsetids) + ".pt"
            )
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        with h5py.File(path, "r") as f:
            pointclouds = []
            for synsetid in self.cate_synsetids:
                for split in ("train", "val", "test"):
                    pointclouds.append(torch.from_numpy(f[synsetid][split][...]))

        all_points = torch.cat(pointclouds, dim=0)  # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B * N, -1).mean(dim=0)  # (1, 3)
        std = all_points.view(-1).std(dim=0)  # (1, )

        self.stats = {"mean": mean, "std": std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):

        path = os.path.join(self.path, "shapenet.hdf5")

        def _enumerate_pointclouds(f):
            for synsetid in self.cate_synsetids:
                cate_name = synsetid_to_cate[synsetid]
                for j, pc in enumerate(f[synsetid][self.split]):
                    yield torch.from_numpy(pc), j, cate_name

        with h5py.File(path, mode="r") as f:
            for pc, pc_id, cate_name in _enumerate_pointclouds(f):

                if self.scale_mode == "global_unit":
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats["std"].reshape(1, 1)
                elif self.scale_mode == "shape_unit":
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == "shape_half":
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == "shape_34":
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == "shape_bbox":
                    pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

                pc = (pc - shift) / scale

                self.pointclouds.append(
                    {
                        "data": pc,
                        "cate": cate_name,
                        "id": pc_id,
                        "shift": shift,
                        "scale": scale,
                    }
                )

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data["id"], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            k: v.clone() if isinstance(v, torch.Tensor) else copy(v)
            for k, v in self.pointclouds[idx].items()
        }

        batch = {
            "coord": data["data"],
            "signal": data["data"],
            "shift": data["shift"],
            "scale": data["scale"],
        }

        return batch


class ShapenetDataModule(LightningDataModule):
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
        path,
        num_workers,
        batch_size,
        split,
        category,
        scale_mode,
        val_set_config,
        pin_memory=True,
        **kwargs
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.path = path

        # List of all image paths
        self.split = split

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
            self.train_set = ShapeNetDataset(
                path=self.hparams.path,
                category=self.hparams.category,
                split=self.hparams.split,
                scale_mode=self.hparams.scale_mode,
            )
            self.val_set = ShapeNetLION1000(
                path=self.hparams.path,
                category=self.hparams.val_set_config.category,
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
