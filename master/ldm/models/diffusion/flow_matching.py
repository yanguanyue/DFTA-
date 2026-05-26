"""
Flow Matching variants for LDM-style models.

These classes adapt the existing DDPM/LatentDiffusion logic to a
flow-matching objective while keeping the original module interfaces.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ldm.util import default
from ldm.models.diffusion.ddpm import DDPM, LatentDiffusion


def _sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample continuous time t in [0, 1] for flow matching."""
    return torch.rand(batch_size, device=device)


def _expand_time(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Expand time tensor for broadcasting with image/latent tensors."""
    return t.view(-1, *([1] * (like.ndim - 1)))


def _time_to_index(t: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Map continuous time t in [0,1] to discrete timestep indices."""
    return (t * (num_timesteps - 1)).long()


def _register_flow_schedule(module, timesteps: int, linear_start: float, linear_end: float) -> None:
    """Register a lightweight, torch-only schedule for flow-matching models."""
    module.num_timesteps = int(timesteps)
    module.linear_start = linear_start
    module.linear_end = linear_end

    betas = torch.linspace(linear_start, linear_end, module.num_timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]])

    module.register_buffer('betas', betas)
    module.register_buffer('alphas_cumprod', alphas_cumprod)
    module.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

    module.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
    module.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    module.register_buffer('log_one_minus_alphas_cumprod', torch.log(torch.clamp(1.0 - alphas_cumprod, min=1e-20)))
    module.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
    module.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1.0))

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    module.register_buffer('posterior_variance', posterior_variance)
    module.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
    module.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    module.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    module.register_buffer('lvlb_weights', torch.ones_like(betas), persistent=False)


class FlowMatchingDDPM(DDPM):
    """DDPM-style wrapper trained with a flow-matching loss."""

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        _register_flow_schedule(self, timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)

    def flow_q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        t_view = _expand_time(t, x_start)
        return (1.0 - t_view) * x_start + t_view * noise

    def p_losses(self, x_start: torch.Tensor, t: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if t is None:
            t = _sample_time(x_start.shape[0], x_start.device)

        x_t = self.flow_q_sample(x_start=x_start, t=t, noise=noise)
        t_idx = _time_to_index(t, self.num_timesteps)
        model_out = self.model(x_t, t_idx)

        target = noise - x_start
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'
        loss_dict = {f'{log_prefix}/loss_simple': loss.mean()}

        logvar_t = self.logvar[t_idx].to(self.device)
        loss = loss / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{log_prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        loss_dict.update({f'{log_prefix}/loss': loss})
        loss_dict.update({f'{log_prefix}/loss_vlb': torch.tensor(0.0, device=loss.device)})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


class FlowMatchingLatentDiffusion(LatentDiffusion):
    """Latent Diffusion trained with flow matching."""

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        _register_flow_schedule(self, timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)

    def flow_q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        t_view = _expand_time(t, x_start)
        return (1.0 - t_view) * x_start + t_view * noise

    def forward(self, x, c, *args, **kwargs):
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:
                t_cond = _sample_time(x.shape[0], x.device)
                tc = _time_to_index(t_cond, self.num_timesteps)
                c = self.flow_q_sample(x_start=c, t=t_cond, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, *args, **kwargs)

    def p_losses(self, x_start: torch.Tensor, cond, t: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if t is None:
            t = _sample_time(x_start.shape[0], x_start.device)
        else:
            if t.dtype in (torch.int32, torch.int64) or t.max() > 1:
                t = t.float() / float(self.num_timesteps - 1)

        x_t = self.flow_q_sample(x_start=x_start, t=t, noise=noise)
        t_idx = _time_to_index(t, self.num_timesteps)
        model_output = self.apply_model(x_t, t_idx, cond)

        target = noise - x_start
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        log_prefix = 'train' if self.training else 'val'
        loss_dict = {f'{log_prefix}/loss_simple': loss_simple.mean()}

        logvar_t = self.logvar[t_idx].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{log_prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        loss_dict.update({f'{log_prefix}/loss': loss})
        loss_dict.update({f'{log_prefix}/loss_vlb': torch.tensor(0.0, device=loss.device)})

        return loss, loss_dict


class FlowMatchingSampler:
    """Euler sampler for flow-matching models.

    Supports CSFS (Conditional Stochastic Flow Sampling): when stochastic=True,
    injects scaled Gaussian noise at each Euler step to improve sample diversity
    and quality. The noise scale is controlled by the noise_scale parameter.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @torch.no_grad()
    def sample(
        self,
        steps: int,
        batch_size: int,
        shape: Tuple[int, int, int],
        cond: Optional[Dict[str, List[torch.Tensor]]] = None,
        stochastic: bool = False,
        noise_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        device = next(self.model.parameters()).device
        x = torch.randn((batch_size, *shape), device=device)
        t_seq = torch.linspace(1.0, 0.0, steps + 1, device=device)
        intermediates: List[torch.Tensor] = [x]

        for i in range(steps):
            t = t_seq[i].repeat(batch_size)
            t_idx = _time_to_index(t, self.model.num_timesteps)
            if hasattr(self.model, 'apply_model') and cond is not None:
                v = self.model.apply_model(x, t_idx, cond)
            else:
                v = self.model.model(x, t_idx)

            dt = (t_seq[i + 1] - t_seq[i]).view(1, 1, 1, 1)
            x = x + dt * v
            # CSFS (Conditional Stochastic Flow Sampling): inject noise
            # proportional to sqrt(|dt|) for stochastic sampling.
            if stochastic and noise_scale > 0.0:
                dt_abs = dt.abs()
                noise = torch.randn_like(x, generator=generator)
                x = x + noise_scale * torch.sqrt(dt_abs) * noise
            if verbose or i % max(steps // 10, 1) == 0 or i == steps - 1:
                intermediates.append(x)

        return x, intermediates
