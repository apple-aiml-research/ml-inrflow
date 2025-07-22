#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch
import numpy as np

from utils.utils import right_pad_dims_to


"""
Notice that:
x0: noise
x1: data
In inference, we sample time from 0 -> 1
"""


class BasePath:
    """base class for flow matching path"""

    def __init__(self):
        return

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return None, None

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return None, None

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        return d_alpha_t / alpha_t

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt

    def compute_ut(self, t, x0, x1):
        """Compute the vector field corresponding to p_t"""
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def interpolant(self, t, x0, x1):
        t = right_pad_dims_to(x0, t)
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1)
        return t, xt, ut

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = right_pad_dims_to(x, t)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift_mean = alpha_ratio * x
        drift_var = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift_mean, drift_var

    def compute_diffusion(self, x, t, form="SBDM", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
            x: [batch_dim, ...], data point
            t: [batch_dim, ...] or scalar, timestep
            form: str, form of the diffusion term
            norm: float, norm of the diffusion term
        """
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion

    def compute_log_snr(self, t):
        """Compute the signal-to-noise ratio at time t"""
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        snr = alpha_t / sigma_t
        return 2 * torch.log(snr.clamp(min=1e-10))

    def compute_start_from_noise(self, noise, y_t, t):
        t = right_pad_dims_to(y_t, t)
        alpha, _ = self.compute_alpha_t(t)
        sigma, _ = self.compute_sigma_t(t)
        return (y_t - sigma * noise) / alpha

    def compute_noise_from_start(self, y_0, y_t, t):
        t = right_pad_dims_to(y_t, t)
        alpha, _ = self.compute_alpha_t(t)
        sigma, _ = self.compute_sigma_t(t)
        return (y_t - alpha * y_0) / sigma

    def compute_start_from_v(self, v, y_t, t):
        t = right_pad_dims_to(y_t, t)
        alpha, _ = self.compute_alpha_t(t)
        sigma, _ = self.compute_sigma_t(t)
        return alpha * y_t - sigma * v

    def compute_v(self, y_0, noise, t):
        t = right_pad_dims_to(y_0, t)
        alpha, _ = self.compute_alpha_t(t)
        sigma, _ = self.compute_sigma_t(t)
        return alpha * noise - sigma * y_0

    def compute_score_from_noise(self, noise, t):
        t = right_pad_dims_to(noise, t)
        sigma, _ = self.compute_sigma_t(t)
        return - noise / sigma

    def compute_noise_from_score(self, s_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        sigma, _ = self.compute_sigma_t(t)
        return - s_t * sigma

    def compute_score_from_velocity(self, v_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = y_t
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * v_t - mean) / var
        return score

    def compute_velocity_from_score(self, s_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        drift_mean, drift_var = self.compute_drift(y_t, t)
        velocity = -drift_mean + drift_var * s_t
        return velocity

    def compute_velocity_from_noise(self, noise, y_t, t):
        t = right_pad_dims_to(y_t, t)
        drift_mean, drift_var = self.compute_drift(y_t, t)
        sigma_t, _ = self.compute_sigma_t(t)
        s_t = noise / -sigma_t
        velocity = -drift_mean + drift_var * s_t
        return velocity

    def compute_noise_from_velocity(self, v_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        noise = (v_t - d_alpha_t * y_t) / d_sigma_t
        return noise


class LinearPath(BasePath):
    """Linear Coupling Plan"""

    def __init__(self):
        return

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1
    
    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t
