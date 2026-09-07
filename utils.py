
import os 
import math
import torch
import random
import numpy as np
from torch import nn
from PIL import Image
from datetime import datetime
from torch.nn.functional import interpolate

def mk_logdir(path):
    now = datetime.now()
    folder_name = now.strftime("%y_%m_%d_%H_%M") 
    logdir = os.path.join(path, folder_name)
    os.makedirs(logdir, exist_ok=True)
    return logdir

def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def load_adaface_model(ckpt_path, architecture='ir_18'):
    # load model and pretrained statedict
    model = net.build_model(architecture)
    statedict = torch.load(ckpt_path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def face_verification_loss(output, target, model):
    if output.shape[-1] != 112 or target.shape[-1] != 112:
        output = interpolate(output, (112, 112), mode='bicubic', align_corners=False).clamp(0, 1)
        target = interpolate(target, (112, 112), mode='bicubic', align_corners=False).clamp(0, 1)
    out_feat, _ = model(output * 2 - 1)
    tar_feat, _ = model(target * 2 - 1)
    out_feat = out_feat.detach()
    tar_feat = tar_feat.detach()
    cos = (nn.functional.normalize(out_feat, dim=1) * nn.functional.normalize(tar_feat, dim=1)).sum(dim=1)
    return cos
    

def calculate_psnr(output, target):
    mse = nn.functional.mse_loss(output, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
    return psnr
    
def calculate_lpips(output, target, metric):
    with torch.no_grad():
        lpips = metric(output, target, normalize=True)
    return lpips.mean().item()

def get_img(x, normalize=False):
    if normalize:
        x = (x + 1) / 2    # for (-1, 1) images
    x = x.clamp(0, 1)
    output_image = x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    output_image = (output_image * 255).astype(np.uint8)
    return output_image

def save_output_images(output_tensors, target_tensors, folder, step=0, normalize=False):
    os.makedirs(folder, exist_ok=True)
    for i, (output_tensor, target_tensor) in enumerate(zip(output_tensors, target_tensors)):
        if normalize:
            output = get_img(output_tensor, normalize=True)
            target = get_img(target_tensor, normalize=True)
        else:
            output = get_img(output_tensor)
            target = get_img(target_tensor)
        result_image = Image.fromarray(np.concatenate([output, target], 1))
        save_path = f"{folder}/num_{step + i + 1}.jpg"
        result_image.save(save_path)
    return step + i + 1


