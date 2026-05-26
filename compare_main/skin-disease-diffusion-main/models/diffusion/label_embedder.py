
import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer
import torch.nn.functional as F

class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=1024, num_classes=7, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        # self.register_buffer("alpha", torch.tensor(1.0))


    def forward(self, condition):
        emb = self.embedding(condition)
        emb = F.layer_norm(emb, emb.shape[-1:])        # 정규화
        return self.alpha * emb                        # 스케일 적용


