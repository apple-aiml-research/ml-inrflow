#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch
import numpy as np

from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment

# from pytorch3d.loss import chamfer_distance
from torchmetrics.utilities.data import dim_zero_cat
import torchvision.models as models


import numpy as np
import torch.nn as nn
from torch import Tensor

import torchvision.transforms as transforms

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
from tqdm import trange
import torch.nn.functional as F
from typing import List

from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Geometry import Point3D
from tqdm import tqdm
from torch_fidelity.utils import create_feature_extractor
from torch_fidelity.metric_prc import prc_features_to_metric
from torch.hub import load_state_dict_from_url

# from torch_fidelity.helpers import get_kwarg, vprint
# from torch_fidelity.utils import (
#     create_feature_extractor,
#     extract_featuresdict_from_input_id_cached,
#     resolve_feature_extractor,
#     resolve_feature_layer_for_metric,
# )


def iterate_in_chunks(l, n):
    """Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def cd_in_batch(xyz1, xyz2):
    raise ("You need to install Pytorch3D to compute chamfer distances")
    # xyz1, xyz2: (B, N, 3)
    # cd_12, _ = chamfer_distance(xyz1.cuda(), xyz2.cuda(), batch_reduction=None)
    # cd_21, _ = chamfer_distance(xyz2.cuda(), xyz1.cuda(), batch_reduction=None)
    # return (cd_12.unsqueeze(1) + cd_21.unsqueeze(1)) / 2  # (B, 1)


def set_rdmol_positions(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol_ = deepcopy(mol)
    for i in range(pos.shape[0]):
        # mol_.GetConformer().SetAtomPosition(i, pos[i].tolist())
        conf = mol_.GetConformers()[0]
        x, y, z = pos[i].tolist()
        conf.SetAtomPosition(i, Point3D(x, y, z))
    return mol_


def calc_rmsd(mol, ref_mol):
    mol = Chem.RemoveHs(mol)
    ref_mol = Chem.RemoveHs(ref_mol)
    try:
        rmsd = rdMolAlign.GetBestRMS(mol, ref_mol)
    except:
        # print("Can't match molecules!", mol, ref_mol)
        rmsd = None
    return rmsd


def calc_confution_mat(sampled_mols, gt_mols, useFF=False):
    n_confs = len(gt_mols)
    n_samples = len(sampled_mols)
    rmsd_confusion_mat = np.nan * np.ones([n_confs, n_samples], dtype=np.float64)

    for i in range(n_samples):
        if useFF:
            # print('Applying FF on generated molecules...')
            MMFFOptimizeMolecule(sampled_mols[i])

        for j in range(n_confs):
            mol_ref = deepcopy(gt_mols[j])
            mol_sampled = deepcopy(sampled_mols[i])
            rmsd_confusion_mat[j, i] = calc_rmsd(mol_sampled, mol_ref)
    return rmsd_confusion_mat


def molecule_gen_metric(sample_mols_list, gt_mols_list, threshold=0.5):
    b = len(gt_mols_list)
    rmsd_confusion_matrices = []
    covr_scores = []
    matr_scores = []
    covp_scores = []
    matp_scores = []

    for i in range(b):
        gt_mols = gt_mols_list[i]
        sampled_mols = sample_mols_list[i]
        rmsd_conf_mat = calc_confution_mat(sampled_mols, gt_mols)

        if np.isnan(np.sum(rmsd_conf_mat)):
            print("NAN in RMSD confusion matrix!")

        rmsd_ref_min = np.nanmin(rmsd_conf_mat, axis=-1)  # np (num_ref, )
        rmsd_gen_min = np.nanmin(rmsd_conf_mat, axis=0)  # np (num_gen, )
        rmsd_cov_thres = rmsd_ref_min.reshape(-1, 1) <= threshold  # np (num_ref, )
        rmsd_jnk_thres = rmsd_gen_min.reshape(-1, 1) <= threshold  # np (num_gen, )

        matr_scores.append(rmsd_ref_min.mean())
        covr_scores.append(rmsd_cov_thres.mean())
        matp_scores.append(rmsd_gen_min.mean())
        covp_scores.append(rmsd_jnk_thres.mean())
        rmsd_confusion_matrices.append(rmsd_conf_mat)

    # covr_scores = np.vstack(covr_scores)  # np (num_mols, num_thres)
    covr_scores = np.array(covr_scores)  # np (num_mols, )
    matr_scores = np.array(matr_scores)  # np (num_mols, )
    # covp_scores = np.vstack(covp_scores)  # np (num_mols, num_thres)
    covp_scores = np.array(covp_scores)  # np (num_mols, )
    matp_scores = np.array(matp_scores)  # np (num_mols, )

    return rmsd_confusion_matrices, covr_scores, matr_scores, covp_scores, matp_scores


def manifold_gen_metric(sample_pcs, ref_pcs, batch_size=64, device=None):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]

    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    mmd, _ = minimum_mathing_distance(
        sample_pcs=sample_pcs,
        ref_pcs=ref_pcs,
        batch_size=batch_size,
        device=device,
        mode="rgb",
    )
    cov = coverage(
        sample_pcs=sample_pcs,
        ref_pcs=ref_pcs,
        batch_size=batch_size,
        device=device,
        mode="rgb",
    )

    return mmd, cov


def pointcloud_gen_metric(sample_pcs, ref_pcs, batch_size=64, device=None):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]

    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    mmd, _ = minimum_mathing_distance(
        sample_pcs=sample_pcs, ref_pcs=ref_pcs, batch_size=batch_size, device=device
    )
    cov = coverage(
        sample_pcs=sample_pcs, ref_pcs=ref_pcs, batch_size=batch_size, device=device
    )

    return mmd, cov


def minimum_mathing_distance(
    sample_pcs, ref_pcs, batch_size, use_EMD=False, device=None, mode=None
):
    """Computes the MMD between two sets of point-clouds.
    Args:
        sample_pcs (torch tensor SxKx3): the S point-clouds, each of K points that will be matched and
            compared to a set of "reference" point-clouds.
        ref_pcs (torch tensor RxKx3): the R point-clouds, each of K points that constitute the set of
            "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to make
            the comparisons of the sample-vs-ref point-clouds.
        test_loader: data loader from test dataset. This must extract 1 samples for each iteration.
        use_EMD (boolean: If true, the matchings are based on the EMD.
    Returns:
        A tuple containing the MMD and all the matched distances of which the MMD is their mean.
    """

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError("Incompatible size of point-clouds.")

    matched_dists = []
    for i in range(n_ref):
        best_in_all_batches = []
        ref_pc = ref_pcs[i]

        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            batch_sample_chunk = sample_chunk.shape[0]
            ref_repeat = ref_pc.repeat(batch_sample_chunk, 1, 1)

            if mode == "rgb":
                all_dist_in_batch = (
                    F.mse_loss(ref_repeat, sample_chunk, reduction="none")
                    .sum(dim=-1)
                    .mean(dim=1)
                )
            else:
                all_dist_in_batch = cd_in_batch(ref_repeat, sample_chunk)

            best_in_batch = torch.min(all_dist_in_batch).item()
            best_in_all_batches.append(best_in_batch)
        matched_dists.append(np.min(best_in_all_batches))
    mmd = np.mean(matched_dists)
    return mmd, matched_dists


def coverage(
    sample_pcs,
    ref_pcs,
    batch_size,
    use_EMD=False,
    ret_dist=False,
    device=None,
    mode=None,
):
    """Computes the Coverage between two sets of point-clouds.
    Args:
        sample_pcs (torch tensor SxKx3): the S point-clouds, each of K points that will be matched
            and compared to a set of "reference" point-clouds.
        ref_pcs    (torch tensor RxKx3): the R point-clouds, each of K points that constitute the
            set of "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to
            make the comparisons of the sample-vs-ref point-clouds.
        use_EMD (boolean): If true, the matchings are based on the EMD.
        ret_dist (boolean): If true, it will also return the distances between each sample_pcs and
            it's matched ground-truth.
        Returns: the coverage score (int),
                 the indices of the ref_pcs that are matched with each sample_pc
                 and optionally the matched distances of the samples_pcs.
    """
    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    n_sam, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError("Incompatible Point-Clouds.")

    matched_gt = []
    matched_dist = []
    for i in range(n_sam):
        best_in_all_batches = []
        loc_in_all_batches = []

        sam_pc = sample_pcs[i]

        for ref_chunk in iterate_in_chunks(ref_pcs, batch_size):
            batch_ref_chunk = ref_chunk.shape[0]
            sam_repeat = sam_pc.repeat(batch_ref_chunk, 1, 1)

            if mode == "rgb":
                all_dist_in_batch = (
                    F.mse_loss(sam_repeat, ref_chunk, reduction="none")
                    .sum(dim=-1)
                    .mean(dim=1)
                )
            else:
                all_dist_in_batch = cd_in_batch(sam_repeat, ref_chunk)

            best_in_batch = torch.min(all_dist_in_batch).item()
            location_of_best = torch.argmin(all_dist_in_batch, axis=0).item()

            best_in_all_batches.append(best_in_batch)
            loc_in_all_batches.append(location_of_best)

        best_in_all_batches = np.array(best_in_all_batches)
        b_hit = np.argmin(best_in_all_batches)  # In which batch the minimum occurred.
        matched_dist.append(np.min(best_in_all_batches))
        hit = np.array(loc_in_all_batches)[b_hit]
        matched_gt.append(batch_size * b_hit + hit)

    cov = len(np.unique(matched_gt)) / float(n_ref)
    return cov


def _pairwise_distances(U, V):
    norm_u = torch.square(U).sum(dim=-1, keepdim=True).view(-1, 1)
    norm_v = torch.square(V).sum(dim=-1, keepdim=True).view(1, -1)
    dist = (norm_u - 2 * U @ V.T + norm_v).clamp(min=0.0)
    return dist


class PrecisionRecall(Metric):
    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features: List[Tensor]
    fake_features: List[Tensor]

    def __init__(self, reset_real_features=True, k=5, **kwargs):
        super().__init__(**kwargs)
        self.reset_real_features = reset_real_features

        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)
        self.k = k
        self.row_batch_size = 10000
        self.col_batch_size = 10000

        # self.vgg16 = create_feature_extractor("vgg16", ["fc2_relu"], {})
        url_nv = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        # fe_us = FeatureExtractorVGG16("vgg16", ["fc2_relu"])
        self.vgg16 = load_state_dict_from_url(
            url_nv, map_location="cpu", progress=True
        ).cuda()

    def update(self, imgs: Tensor, real: bool):
        with torch.no_grad():
            features = self.vgg16(imgs, return_features=True)

        if real:
            if len(self.real_features):
                if dim_zero_cat(self.real_features).shape[0] < 10000:
                    self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self):
        """Calculate precision and recall score based on accumulated extracted features from the two distributions."""

        fake_features = dim_zero_cat(self.fake_features)
        real_features = dim_zero_cat(self.real_features)

        # Following https://arxiv.org/pdf/2202.00273.pdf page 12 compute prc on 10k real and 50k generated
        # (we keep the same ration if evaluated on less samples)
        real_features = real_features[0 : int(fake_features.shape[0] / 5)]

        kwargs = {}
        kwargs["prc_neighborhood"] = 3
        kwargs["batch_size"] = 10000
        kwargs["save_cpu_ram"] = False
        kwargs["verbose"] = True

        out = prc_features_to_metric(real_features, fake_features, **kwargs)

        precision = out["precision"]
        recall = out["recall"]

        # precision, recall = self.prc_features_to_metric(real_features, fake_features)

        return precision, recall

    def reset(self) -> None:
        # if not self.reset_real_features:
        #     # remove temporarily to avoid resetting
        #     value = self._defaults.pop("real_features")
        #     super().reset()
        #     self._defaults["real_features"] = value
        # else:
        super().reset()

    def calc_cdist_part(self, features_1, features_2, batch_size=10000):
        dists = []
        for feat2_batch in features_2.split(batch_size):
            dists.append(torch.cdist(features_1, feat2_batch).cpu())
        return torch.cat(dists, dim=1)

    # def calc_cdist_full(self, features_1, features_2, batch_size=10000):
    #     dists = []
    #     for feat1_batch in features_1.split(batch_size):
    #         dists_batch = []
    #         for feat2_batch in features_2.split(batch_size):
    #             dists_batch.append(torch.cdist(feat1_batch, feat2_batch).cpu())
    #         dists.append(torch.cat(dists_batch, dim=1))
    #     return torch.cat(dists, dim=0)

    def calculate_precision_recall_part(
        self, features_1, features_2, neighborhood=3, batch_size=10000
    ):
        # Precision
        dist_nn_1 = []
        for feat_1_batch in features_1.split(batch_size):
            dist_nn_1.append(
                self.calc_cdist_part(feat_1_batch, features_1, batch_size)
                .to(torch.float32)
                .kthvalue(neighborhood + 1)
                .values
            )
        dist_nn_1 = torch.cat(dist_nn_1)
        precision = []
        for feat_2_batch in features_2.split(batch_size):
            dist_2_1_batch = self.calc_cdist_part(feat_2_batch, features_1, batch_size)
            precision.append((dist_2_1_batch <= dist_nn_1).any(dim=1).float())
        precision = torch.cat(precision).mean().item()
        # Recall
        dist_nn_2 = []
        for feat_2_batch in features_2.split(batch_size):
            dist_nn_2.append(
                self.calc_cdist_part(feat_2_batch, features_2, batch_size)
                .kthvalue(neighborhood + 1)
                .values
            )
        dist_nn_2 = torch.cat(dist_nn_2)
        recall = []
        for feat_1_batch in features_1.split(batch_size):
            dist_1_2_batch = self.calc_cdist_part(feat_1_batch, features_2, batch_size)
            recall.append((dist_1_2_batch <= dist_nn_2).any(dim=1).float())
        recall = torch.cat(recall).mean().item()
        return precision, recall

    # def calculate_precision_recall_full(
    #     self, features_1, features_2, neighborhood=3, batch_size=10000
    # ):
    #     dist_nn_1 = (
    #         self.calc_cdist_full(features_1, features_1, batch_size)
    #         .kthvalue(neighborhood + 1)
    #         .values
    #     )
    #     dist_nn_2 = (
    #         self.calc_cdist_full(features_2, features_2, batch_size)
    #         .kthvalue(neighborhood + 1)
    #         .values
    #     )
    #     dist_2_1 = self.calc_cdist_full(features_2, features_1, batch_size)
    #     dist_1_2 = dist_2_1.T
    #     # Precision
    #     precision = (dist_2_1 <= dist_nn_1).any(dim=1).float().mean().item()
    #     # Recall
    #     recall = (dist_1_2 <= dist_nn_2).any(dim=1).float().mean().item()
    #     return precision, recall

    def prc_features_to_metric(self, features_1, features_2, **kwargs):
        # Convention: features_1 is REAL, features_2 is GENERATED. This important for the notion of precision/recall only.
        assert torch.is_tensor(features_1) and features_1.dim() == 2
        assert torch.is_tensor(features_2) and features_2.dim() == 2
        assert features_1.shape[1] == features_2.shape[1]

        neighborhood = self.k
        batch_size = self.col_batch_size

        precision, recall = self.calculate_precision_recall_part(
            features_1, features_2, neighborhood, batch_size
        )
        f_score = 2 * precision * recall / max(1e-5, precision + recall)

        return precision, recall


class LogLikelihood(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_nll: List[Tensor]
    sample_nll: List[Tensor]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("real_nll", [], dist_reduce_fx=None)
        self.add_state("sample_nll", [], dist_reduce_fx=None)

    def update(self, real_logp: Tensor, sample_logp: Tensor):
        self.real_nll.append(real_logp)
        self.sample_nll.append(sample_logp)

    def compute(self):
        real_nll = dim_zero_cat(self.real_nll)
        sample_nll = dim_zero_cat(self.sample_nll)
        real_nll_mean, real_nll_std = real_nll.mean(), real_nll.std()
        sample_nll_mean, sample_nll_std = sample_nll.mean(), sample_nll.std()
        return real_nll_mean, real_nll_std, sample_nll_mean, sample_nll_std


import torch
import torch.nn.functional as F
import torchvision

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert, text_to_dtype

from torch_fidelity.interpolate_compat_tensorflow import (
    interpolate_bilinear_2d_like_tensorflow1x,
)

import sys
import warnings
from contextlib import redirect_stdout


# def torchvision_load_pretrained_vgg16():
#     with redirect_stdout(sys.stderr), warnings.catch_warnings():
#         warnings.filterwarnings(
#             "ignore", message="The parameter 'pretrained' is deprecated"
#         )
#         warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
#         try:
#             out = torchvision.models.vgg16(
#                 weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
#             )
#         except Exception:
#             out = torchvision.models.vgg16(pretrained=True)
#     return out


# class FeatureExtractorVGG16(FeatureExtractorBase):
#     INPUT_IMAGE_SIZE = 224

#     def __init__(
#         self,
#         name,
#         features_list,
#         feature_extractor_weights_path=None,
#         feature_extractor_internal_dtype=None,
#         **kwargs,
#     ):
#         """
#         VGG16 feature extractor for 2D RGB 24bit images.

#         Args:

#             name (str): Unique name of the feature extractor, must be the same as used in
#                 :func:`register_feature_extractor`.

#             features_list (list): A list of the requested feature names, which will be produced for each input. This
#                 feature extractor provides the following features:

#                 - 'fc2'
#                 - 'fc2_relu'

#             feature_extractor_weights_path (str): Path to the pretrained InceptionV3 model weights in PyTorch format.
#                 Refer to `util_convert_inception_weights` for making your own. Downloads from internet if `None`.

#             feature_extractor_internal_dtype (str): dtype to use inside the feature extractor. Specifying it may improve
#                 numerical precision in some cases. Supported values are 'float32' (default), and 'float64'.
#         """
#         super(FeatureExtractorVGG16, self).__init__(name, features_list)
#         vassert(
#             feature_extractor_internal_dtype in ("float32", "float64", None),
#             "Only 32-bit floats are supported for internal dtype of this feature extractor",
#         )
#         self.feature_extractor_internal_dtype = text_to_dtype(
#             feature_extractor_internal_dtype, "float32"
#         )

#         if feature_extractor_weights_path is None:
#             self.model = torchvision_load_pretrained_vgg16()
#         else:
#             state_dict = torch.load(feature_extractor_weights_path)
#             self.model = torchvision.models.vgg16()
#             self.model.load_state_dict(state_dict)
#         for cls_tail_id in (6, 5, 4):
#             del self.model.classifier[cls_tail_id]

#         self.to(self.feature_extractor_internal_dtype)
#         self.requires_grad_(False)
#         self.eval()

#     def forward(self, x):
#         vassert(
#             torch.is_tensor(x) and x.dtype == torch.uint8,
#             "Expecting image as torch.Tensor with dtype=torch.uint8",
#         )
#         vassert(x.dim() == 4 and x.shape[1] == 3, f"Input is not Bx3xHxW: {x.shape}")
#         features = {}
#         remaining_features = self.features_list.copy()

#         x = x.to(self.feature_extractor_internal_dtype)
#         # N x 3 x ? x ?

#         x = interpolate_bilinear_2d_like_tensorflow1x(
#             x,
#             size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
#             align_corners=False,
#         )
#         # N x 3 x 224 x 224

#         x = torchvision.transforms.functional.normalize(
#             x,
#             (255 * 0.485, 255 * 0.456, 255 * 0.406),
#             (255 * 0.229, 255 * 0.224, 255 * 0.225),
#             inplace=False,
#         )
#         # N x 3 x 224 x 224

#         x = self.model(x)

#         if "fc2" in remaining_features:
#             features["fc2"] = x.to(torch.float32)
#             remaining_features.remove("fc2")
#             if len(remaining_features) == 0:
#                 return tuple(features[a] for a in self.features_list)

#         features["fc2_relu"] = F.relu(x).to(torch.float32)

#         return tuple(features[a] for a in self.features_list)

#     @staticmethod
#     def get_provided_features_list():
#         return "fc2", "fc2_relu"

#     @staticmethod
#     def get_default_feature_layer_for_metric(metric):
#         return {
#             "isc": "fc2_relu",
#             "fid": "fc2_relu",
#             "kid": "fc2_relu",
#             "prc": "fc2_relu",
#         }[metric]

#     @staticmethod
#     def can_be_compiled():
#         return True


#     @staticmethod
#     def get_dummy_input_for_compile():
#         return (torch.rand([1, 3, 4, 4]) * 255).to(torch.uint8)
class MetricsPerTimestep(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    times: List[Tensor]
    losses: List[Tensor]

    def __init__(self, num_bins=1000, **kwargs):
        super().__init__(**kwargs)

        # Decide on the number of bins
        self.num_bins = num_bins  # For example, divide the time range into 5 bins

        # Create bins and bin indices for the time data
        self.bins = np.linspace(0, 1, num_bins + 1)
        # Prepare the bin centers for plotting
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        self.add_state("binned_average_loss", torch.zeros(num_bins), dist_reduce_fx=sum)
        self.add_state(
            "binned_average_output", torch.zeros(num_bins), dist_reduce_fx=sum
        )
        self.add_state(
            "binned_average_target", torch.zeros(num_bins), dist_reduce_fx=sum
        )
        self.counter = 0.0

    def update(self, times: Tensor, losses: Tensor, outputs: Tensor, targets: Tensor):

        bin_indices = (
            np.digitize(times.detach().cpu().numpy(), self.bins) - 1
        )  # Get the bin index for each time value

        self.counter = self.counter + float(times.numel())

        # Update the binned_averages array with the actual averages for bins that have data
        for i in range(self.num_bins):
            if np.any(bin_indices == i):
                self.binned_average_loss[i] = (
                    self.binned_average_loss[i] + losses[bin_indices == i].mean()
                )
                self.binned_average_output[i] = (
                    self.binned_average_output[i] + outputs[bin_indices == i].mean()
                )
                self.binned_average_target[i] = (
                    self.binned_average_target[i] + targets[bin_indices == i].mean()
                )

    def compute(self):
        return (
            self.bin_centers,
            self.binned_average_loss / self.counter,
            self.binned_average_output / self.counter,
            self.binned_average_target / self.counter,
        )
