import os 
import cv2
import glob
import math
import numpy as np
import blobfile as bf
from torch.utils.data import Dataset
from .degradation import (random_mixed_kernels,
                         bivariate_Gaussian,
                         random_add_gaussian_noise,
                         random_add_jpg_compression,
                         add_jpg_compression)



def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def imread(img_path, sz, mode='RGB'):
    if mode == 'RGB':
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = (image / 255.0).astype(np.float32)
        image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_CUBIC)
    
    elif mode == 'GRAY':
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = (image / 255.0).astype(np.float32)
        image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_CUBIC)
        image = (image > 0.5).astype(np.float32)
        image = image[:, :, np.newaxis]
        
    return np.transpose(image.clip(0,1), (2, 0, 1))



def scale_downsample(scale_base, f):
    return 1.0 + (scale_base - 1.0) * f

def scale_jpeg_quality(q_base, f):
    q_f = q_base * (1 + f)               
    return min(round(q_f), 100)

def degradation_image(data,
                      size,
                      subnet = None,
                      blur_kernel_size = 41,
                      kernel_list = ['iso'], 
                      kernel_prob = [1], 
                      blur_sigma = [0.2, 10],
                      downsample_range = [1, 16], 
                      noise_range = [0, 15], 
                      jpeg_range =  [30, 100]):
    
        img_gt = cv2.imread(data)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)     
        if img_gt.shape[0] != size:
            img_gt = cv2.resize(img_gt, (size, size), interpolation=cv2.INTER_LINEAR)
        img_gt = (img_gt / 255.0).astype(np.float32)
        h, w, _ = img_gt.shape
        
        # generate lq image 
        kernel, b_sigma = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            blur_kernel_size,
            blur_sigma,
            blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        kernel = kernel.astype(np.float32)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(downsample_range[0], downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if noise_range is not None:
            img_lq, _ = random_add_gaussian_noise(img_lq, noise_range)
        if jpeg_range is not None:
            img_lq, quality = random_add_jpg_compression(img_lq, jpeg_range)
        
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        lq = (img_lq.clip(0, 1))
        lq = np.transpose(lq, (2, 0, 1))

        if subnet:
            subnet_targets = []
            for f in subnet.values():
                new_kernel = bivariate_Gaussian(blur_kernel_size, b_sigma*f, b_sigma*f, 0).astype(np.float32)
                subnet_lq = cv2.filter2D(img_gt, -1, new_kernel)
                new_scale = scale_downsample(scale, f)
                subnet_lq = cv2.resize(subnet_lq, (int(w // new_scale), int(h // new_scale)), interpolation=cv2.INTER_LINEAR)
                subnet_lq = add_jpg_compression(subnet_lq, scale_jpeg_quality(quality, f))
                subnet_lq = cv2.resize(subnet_lq, (w, h), interpolation=cv2.INTER_LINEAR)
                subnet_targets.append(np.transpose(subnet_lq.clip(0, 1), (2, 0, 1)))
            return lq, subnet_targets

        return lq
    
    
class ImageDataset_FFHQ(Dataset):
    def __init__(self,
        img_dir,
        mask_dir,
        size,
        subnet = None,
    ):
        super().__init__()
        self.data = _list_image_files_recursively(img_dir)
        self.images = []
        self.masks = []
        for fn in self.data:
            filename = os.path.basename(fn)
            mask_fn = f'{mask_dir}/{filename}'
            if os.path.exists(mask_fn.replace('Original', 'Ref')) and os.path.exists(mask_fn):
                self.images.append(fn)
                self.masks.append(mask_fn)

        self.subnet = subnet   
        self.size = size
        print(len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        q_mask_fn = self.masks[idx]
        v_mask_fn = q_mask_fn.replace('Original', 'Ref')

        q_img_path = self.images[idx]
        v_img_path = q_img_path.replace('Original', 'Ref')

        q_mask = imread(q_mask_fn, self.size, mode='GRAY')
        q_mask_origin = q_mask.copy()

        v_img = imread(v_img_path, self.size)
        v_mask = imread(v_mask_fn, self.size, mode='GRAY')
        
        q_img_origin = imread(q_img_path, self.size)    
        
        if self.subnet:
            q_img, subnet_targets = degradation_image(q_img_path, self.size, subnet=self.subnet)  
            return q_img, v_img, q_img_origin, subnet_targets, q_mask, v_mask, q_mask_origin, q_img_path, v_img_path
        else:
            q_img = degradation_image(q_img_path, self.size) 
        return q_img, v_img, q_img_origin, q_mask, v_mask, q_mask_origin, q_img_path, v_img_path
    

class ImageDataset_CelebA(Dataset):
    def __init__(self,
        lq_dir,
        gt_dir,
        mask_dir,
        size
    ):
        super().__init__()
        self.images = _list_image_files_recursively(lq_dir)
        self.gt = sorted(glob.glob(f'{gt_dir}/*.png'))
        self.gt_images = []
        self.masks = []

        for fn in self.gt:
            filename = os.path.basename(fn)
            mask_fn = f'{mask_dir}/{filename}'
            if os.path.exists(mask_fn):
                self.gt_images.append(fn)
                self.masks.append(mask_fn)
        
        self.size = size
        print("LQ: ", len(self.images))
        print("GT: ", len(self.gt_images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        q_mask_fn = self.masks[idx]
        v_mask_fn = q_mask_fn.replace('Q_Mask', 'Ref_Mask')    

        q_img_path = self.images[idx]
        gt_img_path = self.gt_images[idx]
        v_img_path = gt_img_path.replace('GT', 'Ref')   

        q_mask = imread(q_mask_fn, self.size, mode='GRAY')
        q_mask_origin = q_mask.copy()

        q_img = imread(q_img_path, self.size)
        q_img_origin = imread(gt_img_path, 512)   

        v_img = imread(v_img_path, self.size)
        v_mask = imread(v_mask_fn, self.size, mode='GRAY')
  
        return q_img, v_img, q_img_origin, q_mask, v_mask, q_mask_origin, q_img_path, v_img_path
