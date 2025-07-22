#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import torch
from tqdm import tqdm
from einops import repeat
from collections import namedtuple
from functools import partial
from utils.utils import identity


ModelPrediction = namedtuple(
    "ModelPrediction",
    ["query_y_noise", "query_y_0", "query_y_velocity", "query_y_score"],
)


class BaseSampler:
    def __init__(
        self,
        cfg_scale=1.0,
        eval_ema=True,
        clip=False,
        **kwargs,
    ):
        super().__init__()
        self.cfg_scale = cfg_scale
        self.eval_ema = eval_ema
        self.clip = clip

    def setup(self, model, model_ema, path):
        if self.cfg_scale != 1.0:
            if self.eval_ema:
                self.model_fn = model_ema.module.forward_with_cfg
            else:
                self.model_fn = model.forward_with_cfg
        else:
            if self.eval_ema:
                self.model_fn = model_ema.module.forward
            else:
                self.model_fn = model.forward
        self.path = path
        return

    def model_step(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y=None,
        label=None,
        cond_drop_ids=None,
    ):
        if self.cfg_scale != 1.0:
            model_out = self.model_fn(
                context_x=context_x,
                context_y=context_y,
                t=t,
                query_x=query_x,
                query_y=query_y,
                label=label,
                cond_drop_ids=cond_drop_ids,
                cfg_scale=self.cfg_scale,
            )
        else:
            model_out = self.model_fn(
                context_x=context_x,
                context_y=context_y,
                t=t,
                query_x=query_x,
                query_y=query_y,
                label=label,
            )
        return model_out

    def model_predictions(
        self,
        context_x=None,
        context_y=None,
        t=None,
        query_x=None,
        query_y=None,
        label=None,
        cond_drop_ids=None,
    ):
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if self.clip else identity

        model_output = self.model_step(
            context_x=context_x,
            context_y=context_y,
            t=t,
            query_x=query_x,
            query_y=query_y,
            label=label,
            cond_drop_ids=cond_drop_ids,
        )

        query_y_noise = torch.zeros_like(query_y)
        query_y_0 = torch.zeros_like(query_y)
        query_y_velocity = torch.zeros_like(query_y)
        query_y_score = torch.zeros_like(query_y)

        query_y_velocity = model_output
        query_y_score = self.path.compute_score_from_velocity(
            query_y_velocity, query_y, t
        )
        query_y_noise = self.path.compute_noise_from_velocity(
            query_y_velocity, query_y, t
        )
        query_y_0 = self.path.compute_start_from_noise(query_y_noise, query_y, t)
        query_y_0 = maybe_clip(query_y_0)

        return ModelPrediction(
            query_y_noise.float(),
            query_y_0.float(),
            query_y_velocity.float(),
            query_y_score.float(),
        )

    @torch.no_grad()
    def sample(
        self,
        query_x=None,
        query_y_sampled=None,
        label=None,
        cond_drop_ids=None,
    ):
        raise NotImplementedError


class BaseODESampler(BaseSampler):
    def __init__(
        self,
        cfg_scale=1.0,
        eval_ema=True,
        clip=False,
        diff_term="SBDM",
        diff_norm=1.0,
    ):
        super().__init__(
            cfg_scale=cfg_scale,
            eval_ema=eval_ema,
            clip=clip,
        )
        self.diff_term = diff_term
        self.diff_norm = diff_norm

    @torch.no_grad()
    def heun_step(
        self,
        context_x=None,
        context_y=None,
        t=None,
        t_next=None,
        query_x=None,
        query_y=None,
        label=None,
        cond_drop_ids=None,
        add_noise=False,
        clip=True,
        **kwargs,
    ):
        dt = t_next - t
        if add_noise:
            eps = torch.randn_like(query_y).to(query_y)
            d_eps = eps * torch.sqrt(dt)
            diff_coeff = self.path.compute_diffusion(
                query_y, t, self.diff_term, self.diff_norm
            )
            query_y_hat = query_y + torch.sqrt(2 * diff_coeff) * d_eps
        else:
            d_eps = 0.0
            query_y_hat = query_y

        batched_t = repeat(t, " -> b", b=query_x.shape[0])
        batched_t_next = repeat(t_next, " -> b", b=query_x.shape[0])

        model_out = self.model_predictions(
            context_x=context_x,
            context_y=context_y,
            t=batched_t,
            query_x=query_x,
            query_y=query_y_hat,
            label=label,
            cond_drop_ids=cond_drop_ids,
        )
        K1 = model_out.query_y_velocity

        query_y_p = query_y_hat + K1 * dt
        model_out = self.model_predictions(
            context_x=context_x,
            context_y=context_y,
            t=batched_t_next,
            query_x=query_x,
            query_y=query_y_p,
            label=label,
            cond_drop_ids=cond_drop_ids,
        )
        K2 = model_out.query_y_velocity

        query_y_sample = query_y + 0.5 * dt * (K1 + K2)
        return query_y_sample

    @torch.no_grad()
    def euler_maruyama_step(
        self,
        context_x=None,
        context_y=None,
        t=None,
        t_next=None,
        query_x=None,
        query_y=None,
        label=None,
        cond_drop_ids=None,
        add_noise=True,
        clip=True,
        **kwargs,
    ):
        dt = t_next - t
        if add_noise:
            eps = torch.randn_like(query_y).to(query_y)
            d_eps = eps * torch.sqrt(dt)
        else:
            d_eps = 0.0

        batched_t = repeat(t, " -> b", b=query_x.shape[0])
        model_out = self.model_predictions(
            context_x=context_x,
            context_y=context_y,
            t=batched_t,
            query_x=query_x,
            query_y=query_y,
            label=label,
            cond_drop_ids=cond_drop_ids,
        )
        velocity = model_out.query_y_velocity
        score = model_out.query_y_score

        diff_coeff = self.path.compute_diffusion(
            query_y, t, self.diff_term, self.diff_norm
        )
        drift = velocity + diff_coeff * score
        mean_query_y = query_y + drift * dt
        query_y_sample = mean_query_y + torch.sqrt(2.0 * diff_coeff) * d_eps

        return query_y_sample

    @torch.no_grad()
    def euler_step(
        self,
        context_x=None,
        context_y=None,
        t=None,
        t_next=None,
        query_x=None,
        query_y=None,
        label=None,
        cond_drop_ids=None,
        clip=True,
        **kwargs,
    ):
        dt = t_next - t
        batched_t = repeat(t, " -> b", b=query_x.shape[0])
        model_out = self.model_predictions(
            context_x=context_x,
            context_y=context_y,
            t=batched_t,
            query_x=query_x,
            query_y=query_y,
            label=label,
            cond_drop_ids=cond_drop_ids,
        )
        velocity = model_out.query_y_velocity
        mean_query_y = query_y + velocity * dt

        return mean_query_y


class ODESampler(BaseODESampler):
    def __init__(
        self,
        cfg_scale=1.0,
        eval_ema=True,
        clip=False,
        num_timesteps=100,
        t_eps=0.0,
        sample_step="euler",  # ["euler", "euler_maruyama", "heun"]
        diff_term="SBDM",
        diff_norm=1.0,
        **kwargs,
    ):
        super().__init__(
            cfg_scale=cfg_scale,
            eval_ema=eval_ema,
            clip=clip,
            diff_term=diff_term,
            diff_norm=diff_norm,
        )
        self.num_timesteps = num_timesteps
        self.t_eps = t_eps
        self.clip = clip

        if sample_step == "euler":
            self.sample_step = self.euler_step
        elif sample_step == "euler_maruyama":
            self.sample_step = self.euler_maruyama_step
        elif sample_step == "heun":
            self.sample_step = self.heun_step
        else:
            raise ValueError(f"Invalid sample step: {sample_step}")

    @torch.no_grad()
    def sample(
        self,
        query_x=None,
        query_y_sampled=None,
        label=None,
        cond_drop_ids=None,
        context_mask=None,
    ):
        sampling_timesteps = self.num_timesteps
        if context_mask is not None:
            context_x = query_x[:, context_mask]
        else:
            context_x = query_x
        steps = torch.linspace(self.t_eps, 1.0, steps=sampling_timesteps + 1).to(
            query_x.device
        )

        for i in tqdm(
            range(sampling_timesteps), desc="ODE Sampling", total=sampling_timesteps
        ):
            t = steps[i]
            t_next = steps[i + 1]

            if context_mask is not None:
                context_y = query_y_sampled[:, context_mask]
            else:
                context_y = query_y_sampled

            query_y_sampled = self.sample_step(
                context_x,
                context_y,
                t,
                t_next,
                query_x,
                query_y_sampled,
                label=label,
                cond_drop_ids=cond_drop_ids,
                add_noise=False,
                clip=self.clip,
            )

        return query_y_sampled


class SDESampler(BaseODESampler):
    def __init__(
        self,
        cfg_scale=1.0,
        eval_ema=True,
        clip=False,
        num_timesteps=100,
        diff_term="SBDM",
        diff_norm=1.0,
        t_eps=0.0,
        last_h=0.01,
        sample_step="euler_maruyama",  # ["euler_maruyama", "heun", "edm"]
        last_step="euler_maruyama",  # ["euler", "euler_maruyama", "heun"]
        gamma=0.1,
        s_tmin=0.0,
        s_tmax=0.5,
        **kwargs,
    ):
        super().__init__(
            cfg_scale=cfg_scale,
            eval_ema=eval_ema,
            clip=clip,
        )
        self.num_timesteps = num_timesteps
        self.diff_term = diff_term
        self.diff_norm = diff_norm
        self.t_eps = t_eps
        self.last_h = last_h
        self.gamma = gamma
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.clip = clip

        if sample_step == "euler_maruyama":
            self.sample_step = self.euler_maruyama_step
        elif sample_step == "heun":
            self.sample_step = self.heun_step
        elif sample_step == "edm":
            self.sample_step = self.edm_step
        else:
            raise ValueError(f"Invalid sample step: {sample_step}")

        if last_step == "euler":
            self.last_sample_step = self.euler_step
        elif last_step == "euler_maruyama":
            self.last_sample_step = self.euler_maruyama_step
        elif last_step == "heun":
            self.last_sample_step = self.heun_step
        else:
            raise ValueError(f"Invalid last step: {last_step}")

    @torch.no_grad()
    def sample(
        self,
        query_x=None,
        query_y_sampled=None,
        label=None,
        cond_drop_ids=None,
        verbose=True,
    ):
        sampling_timesteps = self.num_timesteps
        context_x = query_x
        steps = torch.linspace(
            self.t_eps, 1.0 - self.last_h, steps=sampling_timesteps + 1
        ).to(query_x.device)

        for i in tqdm(
            range(sampling_timesteps),
            desc="SDE Sampling",
            total=sampling_timesteps,
            disable=not verbose,
        ):
            t = steps[i]
            t_next = steps[i + 1]
            context_y = query_y_sampled

            query_y_sampled = self.sample_step(
                context_x,
                context_y,
                t,
                t_next,
                query_x,
                query_y_sampled,
                label=label,
                cond_drop_ids=cond_drop_ids,
                add_noise=True,
                clip=self.clip,
            )

        # In last sample step, we don't add noise
        if self.last_h > 0:
            context_y = query_y_sampled
            t, t_next = steps[-1], steps[-1] + self.last_h
            query_y_sampled = self.last_sample_step(
                context_x,
                context_y,
                t,
                t_next,
                query_x,
                query_y_sampled,
                label=label,
                cond_drop_ids=cond_drop_ids,
                add_noise=False,
            )

        return query_y_sampled
