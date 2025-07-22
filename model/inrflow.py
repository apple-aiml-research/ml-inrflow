#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import os
import copy
import torch
from time import time

from torch.optim.optimizer import Optimizer
from torch.optim.swa_utils import AveragedModel
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel

import lightning
import lightning.pytorch as pl
from lightning_fabric.utilities.throughput import (
    Throughput,
    get_available_flops,
)
from deepspeed.profiling.flops_profiler import get_model_profile

from utils.utils import default
from evaluator.sample_gather import SampleGather


def logit_normal_sample(n=1, m=0.0, s=1.0):
    # Logit-Normal Sampling from https://arxiv.org/pdf/2403.03206.pdf
    u = torch.randn(n) * s + m
    t = 1 / (1 + torch.exp(-u))
    return t


class INRFlow(pl.LightningModule):

    def __init__(
        self,
        architecture,
        processor,
        loss,
        path,
        sampler,
        optimizer=None,
        scheduler=None,
        evaluator=None,
        compile=False,
        ema_decay=0.999,
        num_classes=None,
        clip_grad_norm_val=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # HardCode: manually add evaluator here
        # otherwith fvd&vis metric will raise an error in the setup function
        self.evaluator = self.hparams.evaluator

        self.model = architecture
        self.model_ema = AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.hparams.ema_decay
            ),
            use_buffers=True,
        )
        self.model_ema.eval()

        self.loss = loss
        self.path = path
        self.sampler = sampler
        self.sampler.setup(self.model, self.model_ema, self.path)
        self.num_classes = num_classes

        # To collect validation samples across all gpus
        self.y_gt_gather = SampleGather()
        self.y_sampled_gather = SampleGather()

        self.nval_steps = 0

        try:
            self.t_eps = self.sampler.t_eps
        except AttributeError:
            self.t_eps = 0.0

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def training_losses(
        self,
        context_x,
        context_y,
        t,
        query_x,
        query_y,
        context_noise,
        query_noise,
        label=None,
    ):
        context_noise = default(context_noise, lambda: torch.randn_like(context_y))
        query_noise = default(query_noise, lambda: torch.randn_like(query_y))

        _, context_y_t, _ = self.path.interpolant(t, context_noise, context_y)
        _, query_y_t, query_u_t = self.path.interpolant(t, query_noise, query_y)

        if hasattr(self, "compiled_model"):
            model = self.compiled_model
        else:
            model = self.model

        model_out = model(
            context_x=context_x,
            context_y=context_y_t,
            t=t,
            query_x=query_x,
            query_y=query_y_t,
            label=label,
        )

        self.throughput.update(
            time=time() - self.t0,
            batches=self.global_step,
            samples=self.global_step * context_x.shape[0],
            flops=self.fwd_flops * 3,
        )

        throughput_metric_dict = self.throughput.compute()
        throughput_metric_dict["fwd_gflops_1_sample"] = self.fwd_flops_1_sample / (
            10**9.0
        )
        throughput_metric_dict["total_training_gflops"] = (
            self.fwd_flops * 3 * self.global_step * self.trainer.world_size / (10**9.0)
        )

        loss = self.loss(model_out, query_u_t)
        return loss.mean(), throughput_metric_dict

    def training_step(self, batch, batch_idx, noise=None):

        for k in batch:
            batch[k] = batch[k].cuda()

        (
            context_x,
            context_y,
            query_x,
            query_y,
            context_noise,
            query_noise,
            label,
        ) = self.processor.preprocess_training(batch)

        t = logit_normal_sample(context_y.shape[0], m=0.0, s=1.0).to(self.device)
        t = t * (1 - 2 * self.t_eps) + self.t_eps

        loss, throughput_metric_dict = self.training_losses(
            context_x,
            context_y,
            t,
            query_x,
            query_y,
            context_noise,
            query_noise,
            label=label,
        )

        self.log(
            "loss/mse",
            loss.item(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        self.log(
            "trainer/global_step",
            self.global_step,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
            sync_dist=True,
        )

        for k in throughput_metric_dict:
            self.log(
                "throughput_metrics/" + k,
                throughput_metric_dict[k],
                on_epoch=False,
                logger=True,
                prog_bar=False,
                rank_zero_only=True,
                sync_dist=True,
            )

        self.global_training_step = self.trainer.global_step
        self.epoch = self.trainer.current_epoch
        self.world_size = self.trainer.world_size
        return loss

    @torch.no_grad()
    def sampling(
        self,
        query_x,
        query_y,
        label=None,
    ):
        # generation with CFG
        cond_drop_ids = None
        if self.sampler.cfg_scale > 1.0:
            n = len(query_x)
            query_x = torch.cat([query_x, query_x], 0)
            query_y = torch.cat([query_y, query_y], 0)
            if self.num_classes is not None:
                label_null = torch.tensor([self.num_classes] * n, device=self.device)
                label = torch.cat([label, label_null], 0)
            else:
                label = torch.cat([label, label], 0)
                drop_ids_1 = torch.zeros(n, dtype=torch.bool, device=self.device)
                drop_ids_2 = torch.ones(n, dtype=torch.bool, device=self.device)
                cond_drop_ids = torch.cat([drop_ids_1, drop_ids_2], 0) # drop cond_tokens for the second half

        y_sampled = self.sampler.sample(
            query_x=query_x,
            query_y_sampled=query_y,
            label=label,
            cond_drop_ids=cond_drop_ids,
        )

        if self.num_classes > 0:
            y_sampled, _ = y_sampled.chunk(2, dim=0)

        return y_sampled

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        for k in batch:
            batch[k] = batch[k].cuda()

        x, y, label = self.processor.preprocess_inference(batch)
        y_noise = torch.randn_like(y, device=self.device)

        y_sampled = self.sampling(
            query_x=x,
            query_y=y_noise,
            label=label,
        )

        _, y, y_sampled = self.processor.postprocess(x, y, y_sampled)

        self.y_gt_gather.update(y)
        self.y_sampled_gather.update(y_sampled)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):

        y_gt = self.y_gt_gather.compute()
        y_sampled = self.y_sampled_gather.compute()

        self.y_gt_gather.reset()
        self.y_sampled_gather.reset()

        self.evaluator.save_samples(
            y_gt, y_sampled, batch_idx, global_step=self.global_step
        )
        self.nval_steps = self.nval_steps + 1

    def on_validation_start(self):

        global_seed = os.environ.get("PL_GLOBAL_SEED", 42)
        pl.seed_everything(int(global_seed) + self.trainer.global_rank, True)
        return super().on_validation_start()

    def on_train_start(self):

        global_seed = os.environ.get("PL_GLOBAL_SEED", 42)
        pl.seed_everything(int(global_seed) + self.trainer.global_rank, True)
        return

    def on_validation_epoch_start(self):
        # set sampler with current model
        self.sampler.setup(self.model, self.model_ema, self.path)

    def on_validation_epoch_end(self):

        # When resuming from a checkpoint PL runs a single validation step (dont know why) and it screws up evaluation
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/18110
        if self.nval_steps == 1:
            return

        dist.barrier()
        metrics_dict = self.evaluator.compute_metrics()

        for k in metrics_dict.keys():
            self.log(
                "metrics/" + k,
                metrics_dict[k],
                on_epoch=True,
                logger=True,
                prog_bar=True,
                rank_zero_only=True,
                sync_dist=True,
            )

        train_batch_size = self.trainer.datamodule.train_dataloader().batch_size

        self.log(
            "metrics/global_step",
            self.global_step,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        self.log(
            "metrics/training_num_samples",
            self.global_step * train_batch_size,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        self.nval_steps = 0

    def compute_forward_flops(
        self,
        context_x,
        context_y,
        t,
        query_x,
        query_y,
        label,
    ):

        with torch.no_grad():

            dummy_model = copy.deepcopy(self.model).cuda()
            # https://github.com/microsoft/DeepSpeed/issues/5432

            flops, macs, params = get_model_profile(
                model=dummy_model,  # model
                args=(
                    context_x,
                    context_y,
                    t,
                    query_x,
                    query_y,
                    label,
                ),  # list of positional arguments to the model.
                kwargs=None,  # dictionary of keyword arguments to the model.
                print_profile=False,  # prints the model graph with the measured profile attached to each module
                detailed=False,  # print the detailed profile
                module_depth=-1,  # depth into the nested modules, with -1 being the inner most modules
                top_modules=1,  # the number of top modules to print aggregated profile
                warm_up=0,  # the number of warm-ups before measuring the time of each module
                as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                output_file=None,  # path to the output file. If None, the profiler prints to stdout.
                ignore_modules=None,
            )

        self.fwd_flops = flops
        self.fwd_flops_1_sample = flops / context_x.shape[0]
        self.hparams["fwd_flops"] = self.fwd_flops
        self.hparams["fwd_flops_1_sample"] = self.fwd_flops_1_sample

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        self.processor = self.hparams.processor(device=self.device)

        if self.hparams.compile and stage == "fit":
            self.compiled_model = torch.compile(self.model)

        if stage == "fit":
            self.training_gpus = self.trainer.world_size
            self.hparams["training_gpus"] = self.training_gpus

            batch = next(iter(self.trainer.datamodule.train_dataloader()))

            for k in batch:
                batch[k] = batch[k].cuda()

            (
                context_x,
                context_y,
                query_x,
                query_y,
                context_noise,
                query_noise,
                label,
            ) = self.processor.preprocess_training(batch)
            t = torch.zeros((context_y.shape[0])).cuda()

            self.compute_forward_flops(
                context_x,
                context_y,
                t,
                query_x,
                query_y,
                label=label,
            )
            self.throughput = Throughput(
                available_flops=get_available_flops(
                    device=torch.device("cuda"), dtype=torch.bfloat16
                ),
                world_size=self.trainer.world_size,
            )
            self.t0 = time()

        self.evaluator = self.evaluator(device=self.device)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        optimizer = self.optimizers()
        self.log(
            "trainer/lr",
            optimizer.param_groups[0]["lr"],
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:

        if isinstance(
            self.trainer.strategy, lightning.pytorch.strategies.fsdp.FSDPStrategy
        ):

            with FullyShardedDataParallel.summon_full_params(
                self.trainer.strategy.model, with_grads=True
            ):
                clip_grad_norm_(
                    self.trainer.strategy.model.model.parameters(),
                    self.hparams.clip_grad_norm_val,
                    norm_type=2.0,
                    error_if_nonfinite=True,
                )

        else:
            clip_grad_norm_(
                self.model.parameters(),
                self.hparams.clip_grad_norm_val,
                norm_type=2.0,
                error_if_nonfinite=True,
            )
        return

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        # if self.eval_ema:
        if isinstance(
            self.trainer.strategy, lightning.pytorch.strategies.fsdp.FSDPStrategy
        ):

            with FullyShardedDataParallel.summon_full_params(
                self.trainer.strategy.model
            ):
                self.trainer.strategy.model.model_ema.update_parameters(
                    self.trainer.strategy.model.model
                )

        else:
            self.model_ema.update_parameters(self.model)
        return

    def on_save_checkpoint(self, checkpoint) -> None:

        layers_to_delete = []
        for k in checkpoint["state_dict"].keys():
            if k.startswith("compiled"):
                layers_to_delete.append(k)

        for k in layers_to_delete:
            del checkpoint["state_dict"][k]

        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint) -> None:

        self.training_gpus = checkpoint["hyper_parameters"]["training_gpus"]
        self.fwd_flops = checkpoint["hyper_parameters"]["fwd_flops"]

        return super().on_load_checkpoint(checkpoint)

    def configure_optimizers(self):

        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
