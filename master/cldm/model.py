import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def _load_safetensors(ckpt_path, device):
    import safetensors.torch
    return safetensors.torch.load_file(ckpt_path, device=device)


def load_checkpoint(ckpt_path, location='cpu', exclude_buffers=None, dtype=torch.float16):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = _load_safetensors(ckpt_path, device=location)
        if dtype is not None:
            state_dict = state_dict.to(dtype)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))

    if exclude_buffers:
        state_dict = {k: v for k, v in state_dict.items() if not any(buf_name in k for buf_name in exclude_buffers)}

    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def load_state_dict(ckpt_path, location='cpu', exclude_buffers=None, dtype=torch.float16):
    return load_checkpoint(ckpt_path, location=location, exclude_buffers=exclude_buffers, dtype=dtype)


def build_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model = model.cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


def create_model(config_path):
    return build_model(config_path)


def compare_weights(state_dict, layer1_name, layer2_name):
    if layer1_name not in state_dict or layer2_name not in state_dict:
        missing = [name for name in (layer1_name, layer2_name) if name not in state_dict]
        print(f"Missing layer(s): {', '.join(missing)}")
        return False
    are_equal = torch.equal(state_dict[layer1_name], state_dict[layer2_name])
    print("The weights are identical!" if are_equal else "The weights are different!")
    return are_equal
