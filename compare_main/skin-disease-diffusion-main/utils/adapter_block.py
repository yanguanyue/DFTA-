# skin_diffusion/models/utils/adapter_block.py
import torch
import torch.nn as nn
from einops import rearrange, repeat

class AdapterBlock(nn.Module):

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, vis: torch.Tensor):
        """
        h   : (B, H*W, C)  — UNet 특성
        vis : (B, N_vis, C) — 레이어별 CLS+patch 토큰 (이미 프로젝션 완료)
        """
        x = torch.cat([h, vis], dim=1)          # (B, L+N, C)

        # Q, K, V
        q = self.q_proj(h)                      
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, heads, seq, dim_head]
        B, L, C = q.shape
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B,h,L,L+N)
        attn = attn.softmax(dim=-1)
        out  = attn @ v                                # (B,h,L,d)

        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_proj(out)

        return self.norm(out)      # (B, L, C)
