from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torchvision.transforms.functional as tF
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, Subset
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import json

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim

from models.vae import VAE

batch_size = 100
max_samples = None # set to None for all
target_class = None # None for no specific class

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pil2torch = lambda x: tF.resize(
    tF.center_crop(
        torch.as_tensor(np.array(x)).moveaxis(-1, 0),
        450
    ),
    (256, 256)
)

ds_full = ImageFolder('/your/data/path/here', transform=pil2torch)
all_labels = [Path(path).parent.name for path, _ in ds_full.samples]

logger.info(f"ImageFolder class_to_idx: {ds_full.class_to_idx}")


train_idx, val_idx = train_test_split(
    list(range(len(ds_full))),
    test_size=0.1,
    random_state=42,
    shuffle=True,
    stratify=all_labels  
)

used_indices = val_idx
# ---------- Limit Sample Size (optinal) ----------
if max_samples is not None and max_samples < len(val_idx):
    used_indices = val_idx[:max_samples]

    # limited_val_idx = val_idx[:max_samples]
    # ds_real = Subset(ds_full, limited_val_idx)


# --------- Select specific class ------------
if target_class is not None:
    class_idx = ds_full.class_to_idx[target_class]
    filtered_val_idx = [idx for idx in val_idx
                        if ds_full.samples[idx][1] == class_idx]
    if max_samples:
        filtered_val_idx = filtered_val_idx[:max_samples]
    used_indices = filtered_val_idx
    # ds_real = Subset(ds_full, filtered_val_idx)

ds_real = Subset(ds_full, used_indices)
dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)

class_counts = {}
for idx in used_indices:
    class_name = Path(ds_full.samples[idx][0]).parent.name
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

# for idx in (filtered_val_idx if target_class else val_idx[:len(ds_real)]):
#     class_name = Path(ds_full.samples[idx][0]).parent.name
#     class_counts[class_name] = class_counts.get(class_name, 0) + 1

logger.info(f"Samples Real: {len(ds_real)}")


model = VAE.load_from_checkpoint('/your/ckpt/path/hereckpt')

model.to(device)
model.eval()

first_batch = next(iter(dm_real))
img_shape = first_batch[0].shape
logger.info(f"Image shape: {img_shape}")

if img_shape[-2:] != (256, 256):
    logger.warning(f"Image size mismatch! Expected (224, 224), got {img_shape[-2:]}")
    
    
    
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
path_out = Path.cwd()/'results'/'ham10k'/'metrics'/f'run_{current_time}'
path_out.mkdir(parents=True, exist_ok=True)

# ----------------- Logging -----------

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))

logger.info("="*50)
logger.info(f"Dataset: HAM10k")
logger.info(f"Model: VAE 4ch")
logger.info(f"Device: {device}")
logger.info(f"Total samples: {len(ds_full)}")
logger.info(f"Train samples: {len(train_idx)} ({len(train_idx)/len(ds_full)*100:.1f}%)")
logger.info(f"Val samples: {len(val_idx)} ({len(val_idx)/len(ds_full)*100:.1f}%)")
logger.info(f"Used for evaluation: {len(ds_real)}")
logger.info(f"Class distribution: {class_counts}")
logger.info("="*50)

# ------------- Init Metrics ----------------------
calc_lpips = LPIPS().to(device)


# --------------- Start Calculation -----------------
mmssim_list, mse_list = [], []
for real_batch in tqdm(dm_real):
    imgs_real_batch = real_batch[0].to(device)

    imgs_real_batch = tF.normalize(imgs_real_batch/255, 0.5, 0.5) # [0, 255] -> [-1, 1]
    with torch.no_grad():
        imgs_fake_batch = model(imgs_real_batch)[0].clamp(-1, 1)

    # -------------- LPIP -------------------
    calc_lpips.update(imgs_real_batch, imgs_fake_batch) # expect input to be [-1, 1]

    # -------------- MS-SSIM + MSE -------------------
    for img_real, img_fake in zip(imgs_real_batch, imgs_fake_batch):
        img_real, img_fake = (img_real+1)/2, (img_fake+1)/2  # [-1, 1] -> [0, 1]
        mmssim_list.append(mmssim(img_real[None], img_fake[None], normalize='relu'))
        mse_list.append(torch.mean(torch.square(img_real-img_fake)))

# --------------- Visualization ------------------
logger.info("Generating reconstruction visualizations...")
class_to_indices = {class_name: [] for class_name in ds_full.classes}
for idx in used_indices:
    class_name = Path(ds_full.samples[idx][0]).parent.name
    class_to_indices[class_name].append(idx)


for class_name in class_to_indices:
    class_to_indices[class_name].sort()


for set_idx in range(5):
    fig, axes = plt.subplots(2, 7, figsize=(21, 6))
    fig.suptitle(f'Reconstruction Results - Set {set_idx + 1}', fontsize=16)

   
    for col_idx, class_name in enumerate(sorted(ds_full.classes)):
        
        available_indices = class_to_indices[class_name]

        if len(available_indices) <= set_idx:
            
            logger.warning(f"Not enough samples for class {class_name} in set {set_idx + 1}")
            continue

        
        selected_idx = available_indices[set_idx]

        
        img_path = ds_full.samples[selected_idx][0]
        img_original = ds_full[selected_idx][0].unsqueeze(0).to(device)

        
        img_normalized = tF.normalize(img_original/255, 0.5, 0.5)  # [0, 255] -> [-1, 1]
        with torch.no_grad():
            img_reconstructed = model(img_normalized)[0].clamp(-1, 1)

        
        img_original_vis = img_original[0].cpu() / 255.0
        img_reconstructed_vis = (img_reconstructed[0].cpu() + 1) / 2

        
        axes[0, col_idx].imshow(img_original_vis.permute(1, 2, 0))
        axes[0, col_idx].set_title(class_name.replace('_', ' '), fontsize=10)
        axes[0, col_idx].axis('off')

       
        axes[1, col_idx].imshow(img_reconstructed_vis.permute(1, 2, 0))
        axes[1, col_idx].axis('off')

        
        if col_idx == 0:
            axes[0, 0].text(-0.1, 0.5, 'Original', transform=axes[0, 0].transAxes,
                          fontsize=12, ha='right', va='center', rotation=90)
            axes[1, 0].text(-0.1, 0.5, 'Reconstructed', transform=axes[1, 0].transAxes,
                          fontsize=12, ha='right', va='center', rotation=90)

    plt.tight_layout()

    
    save_path = path_out / f'reconstruction_set_{set_idx + 1}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visualization set {set_idx + 1} to {save_path}")



# -------------- Summary -------------------
mmssim_list = torch.stack(mmssim_list)
mse_list = torch.stack(mse_list)

lpips = 1-calc_lpips.compute()

logger.info(f"LPIPS Score: {lpips}")
logger.info(f"MS-SSIM: {torch.mean(mmssim_list)} ± {torch.std(mmssim_list)}")
logger.info(f"MSE: {torch.mean(mse_list)} ± {torch.std(mse_list)}")


metrics = {
    'LPIPS': float(lpips),
    'MS-SSIM_mean': float(torch.mean(mmssim_list)),
    'MS-SSIM_std': float(torch.std(mmssim_list)),
    'MSE_mean': float(torch.mean(mse_list)),
    'MSE_std': float(torch.std(mse_list))
}

#Save JSON
metrics_path = path_out / 'reconstruction_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
logger.info(f"Saved metrics to {metrics_path}")