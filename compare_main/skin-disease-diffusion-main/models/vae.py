from pathlib import Path 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import numpy as np
import wandb
from utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from utils.lpips import LPIPS
from models.model_base import BasicModel
import lightning as L

from pytorch_msssim import ssim

class DiagonalGaussianDistribution(nn.Module):

    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        batch_size = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar)/batch_size

        return z, kl 
    

class VAE(BasicModel):
    def __init__(
        self,
        in_channels=3, 
        out_channels=3, 
        spatial_dims = 2,
        emb_channels = 4,#8
        hid_chs =    [ 64, 128,  256, 512],
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,   2],
        norm_name = ("GROUP", {'num_groups':8, "affine": True}),
        act_name=("Swish", {}),
        dropout=None,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=True,
        use_attention='none',
        embedding_loss_weight=1e-6,
        perceiver = LPIPS, 
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
        

        optimizer=torch.optim.Adam, 
        optimizer_kwargs={'lr':1e-4},
        lr_scheduler= None, 
        lr_scheduler_kwargs={},
        loss = torch.nn.L1Loss,
        loss_kwargs={'reduction': 'none'},

        sample_every_n_steps = 1000

    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )
        self.sample_every_n_steps=sample_every_n_steps
        self.loss_fct = loss(**loss_kwargs)
        # self.ssim_fct = SSIM(data_range=1, size_average=False, channel=out_channels, spatial_dims=spatial_dims, nonnegative_ssim=True)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None 
        self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.depth = len(strides)
        self.deep_supervision = deep_supervision
        downsample_kernel_sizes = kernel_sizes
        upsample_kernel_sizes = strides 

        # -------- Loss-Reg---------
        # self.logvar = nn.Parameter(torch.zeros(size=()) )

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(
            spatial_dims, 
            in_channels, 
            hid_chs[0], 
            kernel_size=kernel_sizes[0], 
            stride=strides[0],
            act_name=act_name, 
            norm_name=norm_name,
            emb_channels=None
        )

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i-1], 
                out_channels = hid_chs[i], 
                kernel_size = kernel_sizes[i], 
                stride = strides[i],
                downsample_kernel_size = downsample_kernel_sizes[i],
                norm_name = norm_name,
                act_name = act_name,
                dropout = dropout,
                use_res_block = use_res_block,
                learnable_interpolation = learnable_interpolation,
                use_attention = use_attention[i],
                emb_channels = None
            )
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], 2*emb_channels, 3),
            BasicBlock(spatial_dims, 2*emb_channels, 2*emb_channels, 1)
        )


        # ----------- Reparameterization --------------
        self.quantizer = DiagonalGaussianDistribution()    


        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name) 

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i+1], 
                out_channels = hid_chs[i],
                kernel_size=kernel_sizes[i+1], 
                stride=strides[i+1], 
                upsample_kernel_size=upsample_kernel_sizes[i+1],
                norm_name=norm_name,  
                act_name=act_name, 
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None,
                skip_channels=0
            )
            for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True) 
            for i in range(1, deep_supervision+1)
        ])

    
    def encode(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        z, _ = self.quantizer(z)
        return z 
            
    def decode(self, z):
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h)
        x = self.outc(h)
        return x 

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.inc(x_in)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        out_hor = []
        h = self.inc_dec(z_q)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(self.outc_ver) else None 
            h = self.decoders[i](h)
        out = self.outc(h)
   
        return out, out_hor[::-1], emb_loss 
    
    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth<2):
            self.perceiver.eval()
            return self.perceiver(pred, target)*self.perceptual_loss_weight
        else:
            return 0
    
    def ssim_loss(self, pred, target):
        return 1-ssim(((pred+1)/2).clamp(0,1), (target.type(pred.dtype)+1)/2, data_range=1, size_average=False, 
                        nonnegative_ssim=True).reshape(-1, *[1]*(pred.ndim-1))
    
    def rec_loss(self, pred, pred_vertical, target):
        interpolation_mode = 'nearest-exact'

        # Loss
        loss = 0
        
        # mse_loss = self.loss_fct(pred, target)
        # perc_loss = self.perception_loss(pred, target)
        # ssim_loss = self.ssim_loss(pred, target)
        
        # if isinstance(perc_loss, torch.Tensor) and perc_loss.dim() > 0:
        #     perc_loss = perc_loss.mean()
        rec_loss = self.loss_fct(pred, target)+self.perception_loss(pred, target)+self.ssim_loss(pred, target)
        
        # rec_loss = mse_loss + perc_loss + ssim_loss
        loss += torch.sum(rec_loss) / pred.shape[0]

        for i, pred_i in enumerate(pred_vertical): 
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            rec_loss_i = self.loss_fct(pred_i, target_i)+self.perception_loss(pred_i, target_i)+self.ssim_loss(pred_i, target_i)
            # rec_loss_i = rec_loss_i/ torch.exp(self.logvar_ver[i]) + self.logvar_ver[i] 
            loss += torch.sum(rec_loss_i)/pred.shape[0]

        return loss 

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch['source']
        target = x

        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss = self.rec_loss(pred, pred_vertical, target)
        loss += emb_loss*self.embedding_loss_weight
         
        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss':loss, 'emb_loss': emb_loss}
            logging_dict['L2'] = torch.nn.functional.mse_loss(pred, target)
            logging_dict['L1'] = torch.nn.functional.l1_loss(pred, target)
            logging_dict['ssim'] = ssim((pred+1)/2, (target.type(pred.dtype)+1)/2, data_range=1)
            # logging_dict['logvar'] = self.logvar

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

        # ----------------- Log Images ------------------------------
        if state == "train" and self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
            if self.trainer.is_global_zero:
                self.eval()
                
                with torch.no_grad():
                    def norm(x):
                        return (x-x.min())/(x.max()-x.min())
                    
                
                    def depth2batch(image):
                        return (image if image.ndim<5 else torch.swapaxes(image[0], 0, 1))
            
                    x_vis = depth2batch(x)[:8]
                    pred_vis = depth2batch(pred)[:8]
                    
                    # [-1,1] -> [0,1] 정규화
                    x_norm = (x_vis + 1) / 2
                    pred_norm = (pred_vis + 1) / 2
                    
                    # numpy 변환 및 채널 순서 변경 (이전 코드 스타일)
                    x_images = []
                    pred_images = []
                    captions_orig = []
                    captions_recon = []
                    
                    for i in range(min(8, x_norm.shape[0])):
                        # [C,H,W] -> [H,W,C] 변환
                        x_img = norm(torch.moveaxis(x_norm[i], 0, -1))
                        pred_img = norm(torch.moveaxis(pred_norm[i], 0, -1))
                        
                        x_images.append(x_img.cpu().numpy())
                        pred_images.append(pred_img.cpu().numpy())
                        captions_orig.append(f"Original {i}")
                        captions_recon.append(f"Reconstructed {i}")
                    
                    # WandB로 로깅 (이전 코드의 방식)
                    log_step = self.global_step // self.sample_every_n_steps
                    
                    # Original 이미지들 로깅
                    self.logger.log_image(
                        key="vae_reconstruction/original",
                        images=x_images,
                        caption=captions_orig,
                        step=log_step
                    )
                    
                    # Reconstructed 이미지들 로깅
                    self.logger.log_image(
                        key="vae_reconstruction/reconstructed", 
                        images=pred_images,
                        caption=captions_recon,
                        step=log_step
                    )
                    
                    # 비교를 위한 paired 로깅
                    paired_images = []
                    paired_captions = []
                    for i in range(len(x_images)):
                        paired_images.extend([x_images[i], pred_images[i]])
                        paired_captions.extend([f"Original {i}", f"Reconstructed {i}"])
                    
                    self.logger.log_image(
                        key="vae_reconstruction/comparison",
                        images=paired_images,
                        caption=paired_captions,
                        step=log_step
                    )
                
                    try:
                        import wandb
                        if wandb.run is not None:
                            path_out = Path(wandb.run.dir)/'vae_images'
                        else:
                            path_out = Path('./outputs/vae_images')
                        
                        path_out.mkdir(parents=True, exist_ok=True)
                        
                        # save grid
                        from torchvision.utils import save_image
                        comparison_grid = torch.cat([x_norm, pred_norm], dim=0)
                        save_image(comparison_grid, 
                                path_out/f'vae_reconstruction_{log_step}.png', 
                                normalize=True, 
                                nrow=x_norm.shape[0])
                        
                        print(f"✅ Saved VAE reconstruction images at step {self.global_step}")
                        
                    except Exception as e:
                        print(f"Error saving VAE images: {e}")
                
                # 학습 모드로 복귀
                self.train()

        return loss