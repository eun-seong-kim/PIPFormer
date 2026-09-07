import argparse

import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high') 
except Exception:
    pass

import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader
from torch.nn.functional import interpolate

from model import Net
from dataset import ImageDataset_CelebA
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

def get_img(x, normalize=False):
    if normalize:
        x = (x + 1) / 2    # for (-1, 1) images
    x = x.clamp(0, 1)
    output_image = x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    output_image = (output_image * 255).astype(np.uint8)
    return output_image

def main(args):
    # Fix seed
    set_seed()

    pretrain_model_url = {
        'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    }   # CodeFormer ckpt

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)

    patch_dict = {32: 4, 64: 8, 128: 8, 256: 16, 512: 16, 1024: 32}
    dim_dict  = {32: 64, 64: 128, 128: 128, 256: 256, 512: 256, 1024: 512}

    test_dataset = ImageDataset_CelebA(args.lq_dir, args.gt_dir, args.mask_dir, args.target_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = []
    if args.ckpt:
        ckpt = torch.load(ckpt, map_location=device)
        for i in range(4):
            sz = args.input_size
            model = Net(
                image_size=(sz, sz),
                patch_size=(patch_dict[sz], patch_dict[sz]),
                dim=dim_dict[sz],
                depth=6,
                heads=4,
                mlp_dim=dim_dict[sz]//4 * 2,
                dropout=0.0,
            ).to(device)
            try:
                sd = ckpt[f"state_dict_{i}"]
                msd = model.state_dict()
                keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
                skip = [k for k in sd.keys() if k not in keep]
                new  = [k for k in msd.keys() if k not in sd]
                print("[load] keep:", len(keep), "skip:", len(skip), "new:", len(new))
                model.load_state_dict(keep, strict=False)
            except Exception as e:
                print(f'can not load for {sz} resolution: {e}')
            model.eval()
            Model.append(model)

    # Codeformer load
    codeformer = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=['32', '64', '128', '256']
    ).to(device)
    ckpt_path = load_file_from_url(
        url=pretrain_model_url['restoration'],
        model_dir='weights/CodeFormer',
        progress=True, file_name=None
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')['params_ema']
    codeformer.load_state_dict(ckpt)
    codeformer.eval()

    with torch.no_grad():
        
        for batch_idx, (q_img, v_img, q_img_origin, q_mask, v_mask, q_mask_origin, q_img_path, v_img_path) in tqdm(enumerate(test_loader)):
            q_img, q_mask = q_img.to(device), q_mask.to(device)
            v_img, v_mask = v_img.to(device), v_mask.to(device)
            q_img_origin, q_mask_origin = q_img_origin.to(device), q_mask_origin.to(device)
            
            outputs = []
            for i, model in enumerate(Model):
                q_img = model(q_img, v_img, v_img, q_mask, v_mask).clamp(0, 1)
                outputs.append(q_img)

            out = outputs[-1]
            out = interpolate(out, (args.output_size, args.output_size), mode='bicubic', align_corners=False).clamp(0, 1)
            out = out * 2 -1 

            # Refinement 진행
            restored = codeformer(out, w=0.7, adain=True)[0]  
            restored = (restored.clamp(-1, 1) + 1) / 2   
                

            for out, q_path in zip(restored, q_img_path):
                out = get_img(out)
                result_img = Image.fromarray(out)
                save_path, _ = os.path.splitext(os.path.basename(q_path))
                result_img.save(f"{save_path}/{save_path}.png")
            
    print(f"Restoration is completed. Check {save_path}!")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="", help="Checkpoint path")
    parser.add_argument('--lq_dir', type=str, default="", help="LQ images directory path")
    parser.add_argument('--gt_dir', type=str, default="", help="GT directory path")
    parser.add_argument('--mask_dir', type=str, default="", help="LQ mask directory path")
    parser.add_argument('--save_dir', type=str, default="", help="Results path")
    parser.add_argument('--input_size', type=int, default=256, help="Input resolution")
    parser.add_argument('--target_size', type=int, default=256, help="Target resolution. It's same with input resolution.")
    parser.add_argument('--output_size', type=int, default=512, help="Output resolution." \
                        "Before pass the restored output to refinement Network (CodeFormer), we upscle the output (256->512).")
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args() 
    main(args)