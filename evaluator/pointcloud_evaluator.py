#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


from einops import rearrange
import os
from lightning.pytorch.utilities import rank_zero_only
from tqdm import tqdm
from glob import glob
import numpy as np
import torch
from utils.point_evaluation_metrics import compute_all_metrics, normalize_point_clouds
import struct


def write_pointcloud(filename, xyz_points, rgb_points=None):
    """creates a .pkl file of the point clouds generated"""

    assert xyz_points.shape[1] == 3, "Input XYZ points should be Nx3 float array"
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert (
        xyz_points.shape == rgb_points.shape
    ), "Input RGB colors should be Nx3 float array and have same size as input XYZ points"

    # Write header of .ply file
    fid = open(filename, "wb")
    fid.write(bytes("ply\n", "utf-8"))
    fid.write(bytes("format binary_little_endian 1.0\n", "utf-8"))
    fid.write(bytes("element vertex %d\n" % xyz_points.shape[0], "utf-8"))
    fid.write(bytes("property float x\n", "utf-8"))
    fid.write(bytes("property float y\n", "utf-8"))
    fid.write(bytes("property float z\n", "utf-8"))
    fid.write(bytes("property uchar red\n", "utf-8"))
    fid.write(bytes("property uchar green\n", "utf-8"))
    fid.write(bytes("property uchar blue\n", "utf-8"))
    fid.write(bytes("end_header\n", "utf-8"))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(
            bytearray(
                struct.pack(
                    "fffccc",
                    xyz_points[i, 0],
                    xyz_points[i, 1],
                    xyz_points[i, 2],
                    rgb_points[i, 0].tostring(),
                    rgb_points[i, 1].tostring(),
                    rgb_points[i, 2].tostring(),
                )
            )
        )
    fid.close()


class PointcloudEvaluatorPipeline:

    def __init__(self, eval_dir, viz_dir, device):

        self.eval_dir = eval_dir

        self.create_folders()
        self.viz_dir = viz_dir

        self.last_saved_idx = 0

    @rank_zero_only
    def create_folders(self):

        if not os.path.exists(os.path.join(self.eval_dir, "gt")):
            os.mkdir(os.path.join(self.eval_dir, "gt"))

        if not os.path.exists(os.path.join(self.eval_dir, "sampled")):
            os.mkdir(os.path.join(self.eval_dir, "sampled"))

        print()

    @rank_zero_only  # We sample in parallel and gather to master gpu
    def save_samples(self, y_gt, y_sampled, batch_idx, global_step):

        # Save pointclouds to file
        for i, sample in enumerate(y_gt):

            save_path_npy = os.path.join(
                self.eval_dir,
                "gt",
                "gt_{}.npy".format(str(i + self.last_saved_idx).zfill(6)),
            )

            save_path_ply = os.path.join(
                self.eval_dir,
                "gt",
                "gt_{}.ply".format(str(i + self.last_saved_idx).zfill(6)),
            )

            np.save(save_path_npy, sample.cpu().numpy())
            write_pointcloud(save_path_ply, sample.cpu().numpy())

        for i, sample in enumerate(y_sampled):

            save_path_npy = os.path.join(
                self.eval_dir,
                "sampled",
                "sampled_{}.npy".format(str(i + self.last_saved_idx).zfill(6)),
            )

            save_path_ply = os.path.join(
                self.eval_dir,
                "sampled",
                "sampled_{}.ply".format(str(i + self.last_saved_idx).zfill(6)),
            )

            np.save(save_path_npy, sample.cpu().numpy())
            write_pointcloud(save_path_ply, sample.cpu().numpy())

        self.last_saved_idx = self.last_saved_idx + i + 1

    def compute_metrics(self, num_samples=None, eval_dir=None):

        # In case we want to pass a directory manually
        eval_dir = self.eval_dir if eval_dir is None else eval_dir
        assert ("gt" in os.listdir(eval_dir)) and (
            "sampled" in os.listdir(eval_dir)
        ), "The evaluation directory should contain two sub dirs: ./gt and ./sampled"

        metric_dict = {}
        y_sampled_paths = glob(os.path.join(eval_dir, "sampled", "*.npy"))
        y_gt_paths = glob(os.path.join(eval_dir, "gt", "*.npy"))

        if num_samples != None:
            y_sampled_paths = y_sampled_paths[:num_samples]
            y_gt_paths = y_gt_paths[:num_samples]

        # Make dataloader from folders one for gt one for sampled?

        y_sampled_all = []
        y_gt_all = []
        for y_sampled_path, y_gt_path in tqdm(
            zip(y_sampled_paths, y_gt_paths), desc="Computing metrics"
        ):

            y_sampled = torch.from_numpy(np.load(y_sampled_path))
            y_gt = torch.from_numpy(np.load(y_gt_path))
            y_sampled_all.append(y_sampled)
            y_gt_all.append(y_gt)

        y_sampled_all = torch.stack(y_sampled_all, dim=0).cuda()
        y_gt_all = torch.stack(y_gt_all, dim=0).cuda()

        y_sampled_all = 0.5 * normalize_point_clouds(y_sampled_all, mode="shape_bbox")
        y_gt_all = 0.5 * normalize_point_clouds(y_gt_all, mode="shape_bbox")

        metric_dict = compute_all_metrics(y_sampled_all, y_gt_all, batch_size=128)

        self.last_saved_idx = 0

        return metric_dict
