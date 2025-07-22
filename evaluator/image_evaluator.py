#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from torchvision.utils import save_image, make_grid
from einops import rearrange
import os
from lightning.pytorch.utilities import rank_zero_only
from tqdm import tqdm
from glob import glob
from torchvision.io import read_image
from torch import nn


class ImageEvaluatorPipeline:

    def __init__(self, eval_dir, viz_dir, metrics, device):

        self.eval_dir = eval_dir
        self.viz_dir = viz_dir
        self.device = device

        self.create_folders()

        for metric_name in metrics.keys():
            metrics[metric_name] = metrics[metric_name].to(self.device)

        self.metrics = metrics

    @rank_zero_only
    def create_folders(self):

        if not os.path.exists(os.path.join(self.eval_dir, "gt")):
            os.mkdir(os.path.join(self.eval_dir, "gt"))

        if not os.path.exists(os.path.join(self.eval_dir, "sampled")):
            os.mkdir(os.path.join(self.eval_dir, "sampled"))

        if not os.path.exists(self.viz_dir):
            os.mkdir(self.viz_dir)

        print()

    @rank_zero_only  # We sample in parallel and gather to master gpu
    def save_samples(self, y_gt, y_sampled, batch_idx, global_step):

        y_sampled = (y_sampled + 1) * 0.5
        y_gt = (y_gt + 1) * 0.5

        if batch_idx == 0:

            save_path = os.path.join(
                self.viz_dir,
                "step_{}_gt.jpg".format(
                    str(global_step).zfill(8),
                ),
            )
            save_image(
                make_grid(
                    rearrange(
                        y_gt,
                        "b (h w) c -> b c h w",
                        h=int(y_gt.shape[1] ** 0.5),
                        w=int(y_gt.shape[1] ** 0.5),
                    )
                ),
                save_path,
            )

            save_path = os.path.join(
                self.viz_dir,
                "step_{}_sampled.jpg".format(
                    str(global_step).zfill(8),
                ),
            )
            save_image(
                make_grid(
                    rearrange(
                        y_sampled,
                        "b (h w) c -> b c h w",
                        h=int(y_sampled.shape[1] ** 0.5),
                        w=int(y_sampled.shape[1] ** 0.5),
                    )
                ),
                save_path,
            )

        for i, sample in enumerate(y_gt):
            sample = rearrange(
                sample,
                "(h w) c -> c h w",
                h=int(sample.shape[0] ** 0.5),
                w=int(sample.shape[0] ** 0.5),
            )

            save_path = os.path.join(
                self.eval_dir,
                "gt",
                "gt_{}.jpg".format(str(i + (batch_idx * y_gt.shape[0])).zfill(6)),
            )
            save_image(sample, save_path)

        for i, sample in enumerate(y_sampled):
            sample = rearrange(
                sample,
                "(h w) c -> c h w",
                h=int(sample.shape[0] ** 0.5),
                w=int(sample.shape[0] ** 0.5),
            )

            save_path = os.path.join(
                self.eval_dir,
                "sampled",
                "sampled_{}.jpg".format(
                    str(i + (batch_idx * y_sampled.shape[0])).zfill(6)
                ),
            )

            save_image(sample, save_path)

    def compute_metrics(self, eval_dir=None):

        # In case we want to pass a directory manually
        eval_dir = self.eval_dir if eval_dir is None else eval_dir
        assert ("gt" in os.listdir(eval_dir)) and (
            "sampled" in os.listdir(eval_dir)
        ), "The evaluation directory should contain two sub dirs: ./gt and ./sampled"

        metric_dict = {}
        y_sampled_paths = glob(os.path.join(eval_dir, "sampled", "*.jpg"))
        y_gt_paths = glob(os.path.join(eval_dir, "gt", "*.jpg"))

        # Make dataloader from folders one for gt one for sampled?

        for y_sampled_path, y_gt_path in tqdm(
            zip(y_sampled_paths, y_gt_paths), desc="Computing metrics"
        ):

            y_sampled = read_image(y_sampled_path).unsqueeze(0).to(self.device)
            y_gt = read_image(y_gt_path).unsqueeze(0).to(self.device)

            for metric_name in self.metrics.keys():
                if metric_name == "is":
                    self.metrics[metric_name].update(y_sampled)
                else:
                    self.metrics[metric_name].update(y_gt, real=True)
                    self.metrics[metric_name].update(y_sampled, real=False)

        # compute metrics
        for metric_name in self.metrics.keys():
            print("Computing " "{}" " metric...".format(metric_name))
            try:
                if metric_name in ["kid", "is"]:
                    metric_dict[metric_name] = self.metrics[metric_name].compute()[0]
                else:
                    metric_dict[metric_name] = self.metrics[metric_name].compute()
            except Exception as e:
                print(
                    "Error while computing metric {} with exception: {}".format(
                        metric_name, e
                    )
                )
                continue

            self.metrics[metric_name].reset()

        return metric_dict
