import argparse

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

from lpips import LPIPS
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import *            
from models.model import Net
from dataset.dataset import ImageDataset_FFHQ
from models.model_dis import Discriminator

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def main(args):
    set_seed()
    save_path = mk_logdir(args.save_dir)
    os.makedirs(save_path, exist_ok=True)
    tensorboard_log_dir = os.path.join(save_path, 'log')
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    target_size = args.target_size
    input_size = args.input_size
    divides = {1024: 1, 512: 1, 256: 2, 128: 2, 64: 4, 32: 8}
    divide = divides[target_size]
    de_subnet_scale = {0: 0.7, 1: 0.5, 2: 0.3}  # degradation scale
    batch_size = args.batch_size


    patch_dict = {32: 4, 64: 8, 128: 8, 256: 16, 512: 16, 1024: 32} # 해상도별 ViT patch 크기
    dim_dict  = {32: 64, 64: 128, 128: 128, 256: 256, 512: 256, 1024: 512} # 해상도별 Transformer embedding dimension
    patch_size = (patch_dict[target_size], patch_dict[target_size])

    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    global_step = 0

    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ImageDataset_FFHQ(args.lq_dir, args.mask_dir, target_size, subnet=de_subnet_scale)
    val_dataset = ImageDataset_FFHQ(args.lq_dir.replace('Train', 'Test'), args.mask_dir.replace('Train', 'Test'), target_size)

    # Check dataset lodaing
    print("[DBG] len(train) =", len(dataset))
    print("[DBG] len(val)   =", len(val_dataset))


    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    Model = []
    Dis   = []

    for i in range(4):
        sz = input_size
        model = Net(
            image_size=(sz, sz),
            patch_size=(patch_dict[sz], patch_dict[sz]),
            dim=dim_dict[sz],
            depth=6,
            heads=4,
            mlp_dim=dim_dict[sz]//4 * 2,
            dropout=0.0
        ).to(device)
        discriminator = Discriminator(img_resolution=sz, img_channels=6, divide=divide).to(device)

        if args.ckpt:
            try:
                sd = args.ckpt[f"state_dict_{i}"]
                msd = model.state_dict()
                keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
                skip = [k for k in sd.keys() if k not in keep]
                new  = [k for k in msd.keys() if k not in sd]
                print("[load] keep:", len(keep), "skip:", len(skip), "new:", len(new))
                model.load_state_dict(keep, strict=False)
            except Exception as e:
                print(f'can not load for {sz} resolution: {e}')
        
        model.train()
        discriminator.train()
        Model.append(model)
        Dis.append(discriminator)
    
    criterion = nn.L1Loss()
    lambda_perc = 0.02

    # For evaluation
    lpips_fn  = LPIPS(net='vgg', eval_mode=True).requires_grad_(False).to(device)
    for p in lpips_fn.parameters():
        if p.requires_grad:
            p.requires_grad = False

    params_g, params_d = [], []
    for model, dis in zip(Model, Dis):
        params_g.extend(model.parameters())
        params_d.extend(dis.parameters())
        
        
    optimizer_g = optim.Adam(params_g, lr=1e-4, betas=(0., 0.99))   # Optimizer for generator
    optimizer_d = optim.Adam(params_d, lr=2e-5, betas=(0., 0.99))   # Optimizer for discriminator

    start_epoch = 0
    max_epoch = args.max_epoch
    best_score =  float('inf') 
    len_step   = len(Model)

    with open(os.path.join(save_path, 'log.txt'), 'a') as f:
        f.write("ADAM = 0, 0.99, Loss = L1 + Adv + lambda_perc*LPIPS\n")
        f.write("Epoch, Loss, PSNR, LPIPS\n")
        
        
    # Start Training
    print("Start Training ...")
    for epoch in tqdm(range(start_epoch, max_epoch), desc="Epochs"):
        epoch_loss = 0

        epoch_g_total = 0
        epoch_l1    = 0
        epoch_g_adv   = 0
        epoch_g_perc  = 0
        epoch_d_total = 0

        for model, dis in zip(Model, Dis):
            model.train()
            dis.train()

        batch_save_folder = os.path.join(save_path, f'epoch_{epoch + 1}')
        os.makedirs(batch_save_folder, exist_ok=True)
        for batch_idx, (q_img, v_img, q_img_origin, subnet_targets, q_mask, v_mask, q_mask_origin, q_img_path, v_img_path) in enumerate(train_loader):
            q_img, q_mask = q_img.to(device), q_mask.to(device)
            v_img, v_mask = v_img.to(device), v_mask.to(device)
            q_img_origin, q_mask_origin = q_img_origin.to(device), q_mask_origin.to(device)
            subnet_targets = [tar.to(device) for tar in subnet_targets] 
            subnet_targets.append(q_img_origin)     
            lq_img = q_img.clone()

            outputs, targets, v_imgs, q_masks, v_masks, t_Dis = [], [], [], [], [], []
            for i, model in enumerate(Model): 
                sz = input_size
                targets.append(q_img_origin)
                q_masks.append(q_mask_origin)
                v_masks.append(v_mask)
                v_imgs.append(v_img)
                q_img = model(q_img, v_img, v_img, q_mask, v_mask).clamp(0, 1)
                outputs.append(q_img)
                t_Dis.append(Dis[i])
    
            
            # Discriminator
            optimizer_d.zero_grad()
            loss_d = 0.0
            for out_i, sub_tar_i, ref_i, q_m_i, v_m_i, D_i in zip(outputs, subnet_targets, v_imgs, q_masks, v_masks, t_Dis):
                cond_real = torch.cat([sub_tar_i*q_m_i, ref_i*v_m_i], dim=1)    
                cond_fake = torch.cat([out_i.detach()*q_m_i, ref_i*v_m_i], dim=1)  
                loss_d += F.softplus(-D_i(cond_real)).mean() + F.softplus(D_i(cond_fake)).mean()
            (loss_d * 0.1).backward()
            optimizer_d.step()

            # Generator
            optimizer_g.zero_grad()
            g_adv = 0.0
            l1loss = 0.0
            perc_loss = 0.0
            for out_i, sub_tar_i, ref_i, q_m_i, v_m_i, D_i in zip(outputs, subnet_targets, v_imgs, q_masks, v_masks, t_Dis):
                g_adv   += F.softplus(-D_i(torch.cat([out_i*q_m_i, ref_i*v_m_i], dim=1))).mean()
                # g_adv   += F.softplus(-D_i(out_i*q_m_i)).mean()
                l1loss  += criterion(out_i, sub_tar_i)     
                perc_loss += lpips_fn(out_i, sub_tar_i, normalize=True).mean()
            
            total_loss = l1loss + g_adv + lambda_perc * perc_loss
            total_loss.backward()
            optimizer_g.step()  

            epoch_loss += total_loss.item()

            if global_step % 100 == 0:
                writer.add_scalar("Train/G_total", total_loss.item(),            global_step)           # 전체(G)
                writer.add_scalar("Train/G_L1",    l1loss.item(),               global_step)           # L1
                writer.add_scalar("Train/G_adv",   g_adv.item() / len_step,    global_step)           # G_adv 평균
                writer.add_scalar("Train/G_perc", perc_loss.item() / len_step, global_step)
                writer.add_scalar("Train/D",       loss_d.item() / len_step,    global_step)           # D (scaled)
                
            epoch_g_total += total_loss.item()
            epoch_l1    += l1loss.item()
            epoch_g_adv   += (g_adv.item() / len_step)
            epoch_g_perc += (perc_loss.item() / len_step)
            epoch_d_total += (loss_d.item() / len_step)

            global_step += 1

            if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
                print(
                    f"Epoch [{epoch + 1}/{max_epoch}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"L1_Loss: {l1loss.item():.6f}, G_Loss: {g_adv.item() / len_step:.6f}, D_Loss: {loss_d.item() / len_step:.6f}"
                )

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{max_epoch}], Average Training Loss: {avg_epoch_loss:.4f}")

        for model, dis in zip(Model, Dis):
            model.eval()
            dis.eval()
    
        # Tensorboard
        writer.add_scalar("Epoch/Train_G_total", epoch_g_total / len(train_loader), epoch + 1)
        writer.add_scalar("Epoch/Train_G_L1",    epoch_l1  / len(train_loader), epoch + 1)
        writer.add_scalar("Epoch/Train_G_adv",   epoch_g_adv / len(train_loader), epoch + 1)
        writer.add_scalar("Epoch/Train_G_perc", epoch_g_perc/len(train_loader), epoch + 1)
        writer.add_scalar("Epoch/Train_D",       epoch_d_total/ len(train_loader), epoch + 1)

        # Validataion
        with torch.no_grad():
            val_psnr = 0.0
            val_lpips = 0.0
            
            for batch_idx, (q_img, v_img, q_img_origin, q_mask, v_mask, q_mask_origin, q_img_path, v_img_path) in enumerate(val_loader):
                q_img, q_mask = q_img.to(device), q_mask.to(device)
                v_img, v_mask = v_img.to(device), v_mask.to(device)
                q_img_origin, q_mask_origin = q_img_origin.to(device), q_mask_origin.to(device)
                
                outputs = []
                for i, model in enumerate(Model):
                    q_img = model(q_img, v_img, v_img, q_mask, v_mask).clamp(0, 1)
                    outputs.append(q_img)  

                out = outputs[-1]
                
                batch_psnr      = sum(calculate_psnr(out[i], q_img_origin[i]) for i in range(q_img_origin.size(0)))
                avg_batch_psnr  = batch_psnr / q_img_origin.size(0)
                avg_batch_lpips = lpips_fn(out, q_img_origin, normalize=True).mean()  
                val_psnr  += avg_batch_psnr
                val_lpips += avg_batch_lpips
                
                if batch_idx == 0:
                    save_output_images(torch.cat(outputs, -1), torch.cat([v_img, q_img_origin], -1), batch_save_folder)
                    
            avg_val_psnr  = val_psnr  / len(val_loader)
            avg_val_lpips = val_lpips / len(val_loader)
            print(f"Epoch [{epoch + 1}/{max_epoch}], Average Validation PSNR: {avg_val_psnr:.2f} dB, Average Validation LPIPS: {avg_val_lpips:.2f}")
        
            # Tensorboard
            writer.add_scalar("Epoch/Loss",               avg_epoch_loss, epoch + 1)
            writer.add_scalar("Epoch/Validation_PSNR",    avg_val_psnr,  epoch + 1)
            writer.add_scalar("Epoch/Validation_LPIPS",   avg_val_lpips, epoch + 1)


        with open(os.path.join(save_path, 'log.txt'), 'a') as f:
            f.write(f"{epoch + 1},{avg_epoch_loss:.4f},{avg_val_psnr:.2f},{avg_val_lpips:.2f}\n")

        save_dict = {}
        for i, (model, dis) in enumerate(zip(Model, Dis)):
            save_dict[f"state_dict_{i}"]  = model.state_dict()
            save_dict[f"dis_state_dict_{i}"] = dis.state_dict()

        torch.save(save_dict, os.path.join(save_path, "last.pt"))
        if avg_val_lpips < best_score:
            best_score = avg_val_lpips
            torch.save(save_dict, os.path.join(save_path, "best.pt"))

    writer.flush()
    writer.close()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="", help="Checkpoint path for continual learning")
    parser.add_argument('--lq_dir', type=str, default="/media/HDD/CVLAB/FaceSR/FFHQ/Original/Train", help="LQ images directory path")
    parser.add_argument('--mask_dir', type=str, default="/media/HDD/CVLAB/FaceSR/FFHQ-Mask/Original/Train", help="LQ mask directory path")
    parser.add_argument('--save_dir', type=str, default="", help="Results path")
    parser.add_argument('--input_size', type=int, default=256, help="Input resolution")
    parser.add_argument('--target_size', type=int, default=256, help="Target resolution. It's same with input resolution.")
    parser.add_argument('--output_size', type=int, default=512, help="Output resolution." \
                        "Before pass the restored output to refinement Network (CodeFormer), we upscle the output (256->512).")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_epoch', type=int, default=40)
    args = parser.parse_args() 
    main(args)