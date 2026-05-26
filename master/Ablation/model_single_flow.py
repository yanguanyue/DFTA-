"""
Ablation Model A: Single-Flow without Augmentation
- Only Mask-Flow branch (remove Image-Flow and α_img fusion weight)
- No OSEA pseudo-sample generation or L_aug loss
- Supports both deterministic and CSFS sampling in validation (controlled by kwargs)
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


class SingleFlowNoAugControlLDM(ControlLDM):
    """ControlLDM with only Mask-Flow, no OSEA augmentation."""

    def __init__(
        self,
        control_stage_config,
        control_key,
        only_mid_control,
        *args,
        **kwargs,
    ):
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)

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

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        model_output_mask = self.apply_model(x_t, t_idx, cond_mask)

        target = noise - x_start
        loss_simple = self.get_loss(model_output_mask, target, mean=False).mean([1, 2, 3])
        loss_simple_mask_value = loss_simple.mean().item()

        try:
            log_dir = self.trainer.default_root_dir if getattr(self, "trainer", None) else "."
            log_path = os.path.join(log_dir, "loss_log.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"step={int(getattr(self, 'global_step', 0))} "
                    f"loss_simple_mask={loss_simple_mask_value:.8f} "
                    f"loss_simple_image=NA "
                    f"loss_simple_mask_2_image=NA "
                    f"loss_simple_mask_regularization=NA "
                    f"loss_trajectory_alignment=NA "
                    f"loss_online_aug=NA\n"
                )
        except Exception:
            pass

        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

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
        # CSFS (Conditional Stochastic Flow Sampling)
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
