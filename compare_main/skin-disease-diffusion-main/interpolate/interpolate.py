import torch
import torch.nn.functional as F
from torchvision import utils
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image

# Import your models
from interpolate.mfsnet_train import MFSNet
from models.diffusion.diffusion_pipeline import DiffusionPipeline
from simple_lama_inpainting import SimpleLama
from models.vis_token_extractor import VisTokenExtractor

class LesionInterpolationPipeline:
    def __init__(self, mfsnet_path, diffusion_ckpt_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.segmentation_model = self._load_mfsnet(mfsnet_path)
        self.vis_extractor = VisTokenExtractor()  # 초기화 필요
        self.pipeline = DiffusionPipeline.load_from_checkpoint(
            diffusion_ckpt_path, 
            vis_extractor=self.vis_extractor
        ).to(self.device)
        self.simple_lama = SimpleLama()
        
        # Transforms
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
    
    def _load_mfsnet(self, path):
        model = MFSNet().to(self.device)
        state_dict = torch.load(path)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    
    def generate_and_interpolate(self, class_id, num_samples=20, save_path="results"):
        save_path = Path(save_path) / f"class_{class_id}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            # 1. Generate lesion samples
            samples = self._generate_samples(class_id, num_samples)
            
            # 2. Extract masks
            masks = self._extract_masks(samples)
            
            # 3. Process each sample
            for i in range(len(samples)):
                self._process_single_sample(samples[i], masks[i], i, save_path)
    
    def _generate_samples(self, class_id, num_samples):
        condition = torch.tensor([class_id], device=self.device)
        return self.pipeline.sample(
            num_samples=num_samples,
            img_size=(8, 32, 32),
            guidance_scale=3.0,
            condition=condition,
            steps=1000,
            use_ddim=True
        )
    
    def _extract_masks(self, samples):
        norm_samples = torch.stack([self.normalize(s) for s in samples])
        _, _, _, lateral_map_2 = self.segmentation_model(norm_samples)
        masks = torch.sigmoid(lateral_map_2)
        return (masks > 0.5).float()
    
    def _process_single_sample(self, sample, mask, idx, save_path):
        # Save original sample and mask
        utils.save_image(sample, save_path / f'sample_{idx}.png', normalize=True)
        utils.save_image(mask, save_path / f'mask_{idx}.png', normalize=True)
        
        # Inpainting
        image = Image.open(save_path / f'sample_{idx}.png')
        mask_img = Image.open(save_path / f'mask_{idx}.png').convert('L')
        inpainted = self.simple_lama(image, mask_img)
        inpainted.save(save_path / f'inpainted_{idx}.png')
        
        # Interpolation
        self._interpolate_lesion_normal(sample, inpainted, idx, save_path)
    
    def _interpolate_lesion_normal(self, lesion_sample, inpainted_img, idx, save_path, num_steps=10):
        inpainted_tensor = self.to_tensor(inpainted_img).unsqueeze(0).to(self.device)
        lesion_tensor = lesion_sample.unsqueeze(0)
        
        with torch.no_grad():
            # Encode to latent space
            normal_latent = self.pipeline.latent_embedder.encode(inpainted_tensor)
            lesion_latent = self.pipeline.latent_embedder.encode(lesion_tensor)
            
            # Interpolate
            interpolated_images = []
            for step in range(num_steps + 1):  # 0 to num_steps inclusive
                alpha = step / num_steps
                interpolated = (1 - alpha) * normal_latent + alpha * lesion_latent
                
                # Decode and normalize
                current_image = self.pipeline.latent_embedder.decode(interpolated)
                current_image = ((current_image + 1) / 2.0).clamp(0, 1)
                
                interpolated_images.append(current_image)
            
            # Save interpolation grid
            interpolation_grid = torch.cat(interpolated_images, dim=0)
            utils.save_image(
                interpolation_grid,
                save_path / f'interpolation_{idx}_grid.png',
                nrow=num_steps + 1,
                normalize=False,
                padding=2
            )

# Usage
if __name__ == "__main__":
    pipeline = LesionInterpolationPipeline(
        mfsnet_path='your/MFSNet.pth',
        diffusion_ckpt_path='path/diffusion_ckpt/path/here'
    )
    pipeline.generate_and_interpolate(class_id=6, num_samples=20)