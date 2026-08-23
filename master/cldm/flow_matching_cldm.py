"""
Flow Matching variants for ControlLDM.
"""

from __future__ import annotations

from typing import Optional

import os
import torch

from ldm.util import default
from ldm.models.diffusion.flow_matching import (
    FlowMatchingSampler,
    _sample_time,
    _time_to_index,
    _expand_time,
    _register_flow_schedule,
)
from cldm.cldm import ControlLDM


class FlowMatchingControlLDM(ControlLDM):
    """ControlLDM trained with a flow-matching objective."""

    def __init__(
        self,
        control_stage_config,
        control_key,
        only_mid_control,
        trajectory_consistency_weight: float = 0.0,
        trajectory_consistency_start_step: int = 0,
        trajectory_consistency_detach_image: bool = True,
        trajectory_alignment_steps: int = 5,
        online_aug_weight: float = 0.0,
        online_aug_start_step: int = 0,
        online_aug_detach_pseudo: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        self.trajectory_consistency_weight = trajectory_consistency_weight
        self.trajectory_consistency_start_step = trajectory_consistency_start_step
        self.trajectory_consistency_detach_image = trajectory_consistency_detach_image
        self.trajectory_alignment_steps = trajectory_alignment_steps
        self.online_aug_weight = online_aug_weight
        self.online_aug_start_step = online_aug_start_step
        self.online_aug_detach_pseudo = online_aug_detach_pseudo

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        _register_flow_schedule(self, timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def flow_q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        t_view = _expand_time(t, x_start)
        return (1.0 - t_view) * x_start + t_view * noise

    def p_losses(self, x_start, cond, t: Optional[torch.Tensor] = None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if t is None:
            t = _sample_time(x_start.shape[0], x_start.device)
        else:
            if t.dtype in (torch.int32, torch.int64) or t.max() > 1:
                t = t.float() / float(self.num_timesteps - 1)

        t_idx = _time_to_index(t, self.num_timesteps)
        x_t = self.flow_q_sample(x_start=x_start, t=t, noise=noise)

        cond_mask = {
            "c_crossattn": [cond["c_crossattn"][0]],
            "c_concat": [cond["c_concat_mask"][0]],
        }
        cond_image = {
            "c_crossattn": [cond["c_crossattn"][0]],
            "c_concat": [cond["c_concat_mask"][0]],
            "c_concat_image": [cond["c_concat_image"][0]],
        }

        weights_ones = torch.ones_like(t).to(x_start.device)
        weights_thre = torch.where(t_idx <= 200, torch.tensor(1, device=x_start.device), torch.tensor(0, device=x_start.device))

        weights_mask = 1.0 * weights_ones
        weights_image = 1.0 * weights_ones
        weights_mask_2_image = 1.0 * weights_ones
        weights_mask_regularization = 1.0 * weights_thre

        model_output_mask = self.apply_model(x_t, t_idx, cond_mask)

        target = noise - x_start
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss_simple = weights_mask * self.get_loss(model_output_mask, target, mean=False).mean([1, 2, 3])
        loss_simple_mask_value = loss_simple.mean().item()
        loss_simple_image_value = None
        loss_simple_mask_2_image_value = None
        loss_simple_mask_regularization_value = None
        loss_trajectory_alignment = None
        loss_trajectory_alignment_value = None
        loss_online_aug = None
        loss_online_aug_value = None

        model_output_image = None

        if weights_image.all():
            model_output_image = self.apply_model(x_t, t_idx, cond_image)
            loss_simple_image = self.get_loss(model_output_image, target, mean=False).mean([1, 2, 3])
            loss_simple_image_value = loss_simple_image.mean().item()
            loss_simple = loss_simple + weights_image * loss_simple_image

        if weights_mask_2_image.all():
            loss_simple_mask_2_image = self.get_loss(model_output_mask, model_output_image.detach(), mean=False).mean([1, 2, 3])
            loss_simple_mask_2_image_value = loss_simple_mask_2_image.mean().item()
            loss_simple = loss_simple + weights_mask_2_image * loss_simple_mask_2_image

        trajectory_weight = float(getattr(self, "trajectory_consistency_weight", 0.0))
        trajectory_start_step = int(getattr(self, "trajectory_consistency_start_step", 0))
        trajectory_detach_image = bool(getattr(self, "trajectory_consistency_detach_image", True))
        if (
            trajectory_weight > 0.0
            and model_output_image is not None
            and int(getattr(self, "global_step", 0)) >= trajectory_start_step
        ):
            t_view = _expand_time(t, x_t)
            x0_hat_mask = x_t - t_view * model_output_mask
            x0_hat_image = x_t - t_view * model_output_image
            if trajectory_detach_image:
                x0_hat_image = x0_hat_image.detach()

            loss_trajectory_alignment = self.get_loss(
                x0_hat_mask,
                x0_hat_image,
                mean=False,
            ).mean([1, 2, 3])
            loss_trajectory_alignment_value = loss_trajectory_alignment.mean().item()
            loss_simple = loss_simple + trajectory_weight * loss_trajectory_alignment


        online_aug_weight = float(getattr(self, "online_aug_weight", 0.0))
        online_aug_detach_pseudo = bool(getattr(self, "online_aug_detach_pseudo", True))
        if (
            online_aug_weight > 0.0
            and model_output_image is not None
            and int(getattr(self, "global_step", 0)) > (self.trainer.max_steps * 1 / 3)
        ):
            t_view = _expand_time(t, x_t)
            x0_hat_image = x_t - t_view * model_output_image
            if online_aug_detach_pseudo:
                x0_hat_image = x0_hat_image.detach()

            noise_aug = torch.randn_like(x0_hat_image)
            t_aug = _sample_time(x0_hat_image.shape[0], x0_hat_image.device)
            t_idx_aug = _time_to_index(t_aug, self.num_timesteps)
            x_t_aug = self.flow_q_sample(x_start=x0_hat_image, t=t_aug, noise=noise_aug)
            model_output_mask_aug = self.apply_model(x_t_aug, t_idx_aug, cond_mask)

            target_aug = noise_aug - x0_hat_image
            loss_online_aug = self.get_loss(model_output_mask_aug, target_aug, mean=False).mean([1, 2, 3])
            loss_online_aug_value = loss_online_aug.mean().item()
            loss_simple = loss_simple + online_aug_weight * loss_online_aug

        if (self.global_step > (self.trainer.max_steps * 1 / 3)) and weights_mask_regularization.any():
            recon_output_image = x_t - _expand_time(t, x_t) * model_output_image
            noise_image_2_mask = default(noise, lambda: torch.randn_like(recon_output_image))
            x_noisy_mask_recon = self.flow_q_sample(x_start=recon_output_image, t=t, noise=noise_image_2_mask)

            model_output_mask_xt = self.apply_model(x_noisy_mask_recon.detach(), t_idx, cond_mask)
            loss_simple_mask_regularization = self.get_loss(model_output_mask_xt, noise_image_2_mask, mean=False).mean([1, 2, 3])
            loss_simple_mask_regularization_value = loss_simple_mask_regularization.mean().item()
            loss_simple = loss_simple + weights_mask_regularization * loss_simple_mask_regularization

        try:
            log_dir = self.trainer.default_root_dir if getattr(self, "trainer", None) else "."
            log_path = os.path.join(log_dir, "loss_log.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"step={int(getattr(self, 'global_step', 0))} "
                    f"loss_simple_mask={loss_simple_mask_value:.8f} "
                    f"loss_simple_image={loss_simple_image_value if loss_simple_image_value is not None else 'NA'} "
                    f"loss_simple_mask_2_image={loss_simple_mask_2_image_value if loss_simple_mask_2_image_value is not None else 'NA'} "
                    f"loss_simple_mask_regularization={loss_simple_mask_regularization_value if loss_simple_mask_regularization_value is not None else 'NA'} "
                    f"loss_trajectory_alignment={loss_trajectory_alignment_value if loss_trajectory_alignment_value is not None else 'NA'} "
                    f"loss_online_aug={loss_online_aug_value if loss_online_aug_value is not None else 'NA'}\n"
                )
        except Exception:
            pass

        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        if loss_trajectory_alignment is not None:
            loss_dict.update({f'{prefix}/loss_trajectory_alignment': loss_trajectory_alignment.mean()})
        if loss_online_aug is not None:
            loss_dict.update({f'{prefix}/loss_online_aug': loss_online_aug.mean()})

        logvar_t = self.logvar[t_idx].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        loss_dict.update({f'{prefix}/loss': loss})
        loss_dict.update({f'{prefix}/loss_vlb': torch.tensor(0.0, device=loss.device)})

        return loss, loss_dict

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        sampler = FlowMatchingSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        stochastic = bool(kwargs.pop("stochastic", False))
        noise_scale = float(kwargs.pop("noise_scale", 0.0))
        generator = kwargs.pop("generator", None)
        samples, intermediates = sampler.sample(
            ddim_steps,
            batch_size,
            shape,
            cond,
            stochastic=stochastic,
            noise_scale=noise_scale,
            generator=generator,
            verbose=False,
        )
        return samples, intermediates

