import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

# ── CLIP μ, σ (OpenCLIP)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class VisTokenExtractor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        layer_ids=(5, 11, 17, 23, 31),
        k=32,
        proj_dim=1024,
        device="cuda",
    ):
        super().__init__()
        self.backbone = backbone.to(device)
        self.layer_ids = list(layer_ids)
        self.k = k
        self.device = device

        
        self._feat_dict = {}

        def _make_hook(idx):
            def _hook(_, __, output):
                # output: (B, N+1, C)  CLS + patch
                self._feat_dict[idx] = output
            return _hook

        
        for idx in self.layer_ids:
            self.backbone.transformer.resblocks[idx].register_forward_hook(_make_hook(idx))

        hidden_dim = backbone.conv1.out_channels 

      
        self.proj = nn.ModuleDict({
        str(idx): nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        for idx in self.layer_ids
    }).to(device)

       
    @torch.no_grad()   
    def forward(self, images):
        """
        images: (B, 3, H, W) , 
        returns: list[Tensor]  – concat token (B, N_i, proj_dim)
        """
        dtype  = self.backbone.conv1.weight.dtype          
        device = self.device
        images = images.to(device, dtype=dtype)
        images = images.to(self.device)
        if images.min() < 0:
            images = (images + 1) / 2

        # 1) 224×224 bicubic
        images = F.interpolate(
            images, size=224, mode="bicubic", align_corners=False
        )
        # 2) CLIP normalize
        mean = _CLIP_MEAN.to(device=images.device, dtype=images.dtype)
        std  = _CLIP_STD.to(device=images.device, dtype=images.dtype)
        images = (images - mean) / std

        self._feat_dict.clear()                 

        _ = self.backbone(images)   # forward → hook

        layer_tokens = []
        for idx in self.layer_ids:
            feat = self._feat_dict[idx]             # (B, N+1, C)
            cls_tok, patch_tok = feat[:, :1], feat[:, 1:]   # CLS / patch

            # patch-top-k by L2-norm
            if self.k and patch_tok.size(1) > self.k:
                norm = patch_tok.norm(dim=-1)        # (B, N_patch)
                topk = norm.topk(self.k, dim=1).indices
                patch_tok = patch_tok.gather(1, topk.unsqueeze(-1).expand(-1, -1, patch_tok.size(-1)))

            # CLS + patch concat → (B, 1+k, C)
            toks = torch.cat([cls_tok, patch_tok], dim=1)

            # dimension projection
            toks = self.proj[str(idx)](toks)

            layer_tokens.append(toks)               


        return layer_tokens
