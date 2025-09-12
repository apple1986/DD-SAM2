import numpy as np
import torch
import random
import os
import SimpleITK as sitk
from skimage import measure

## normalize
def norm_img(img):
    img = (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)

    return img

def znorm_img(img):
    img = (img - np.mean(img)) / np.std(img)
    return img

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def cal_dice(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    dice_score = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    
    return dice_score

def save_nii(name, data, data_path):
    save_path = os.path.join(data_path, name)
    if len(data.shape) > 3:
        data = data[0, ...]
    if  "torch" in str(data.dtype):
        data = torch.permute(data, dims=(2, 0, 1)) # notice: PLA as RSA
        img_itk = sitk.GetImageFromArray(data.numpy().astype(np.uint16))
        img_itk.SetSpacing(spacing=(0.89, 0.89, 3.0))
        sitk.WriteImage(img_itk, save_path)
    else:
        data = np.transpose(data, axes=(2, 0, 1))
        img_itk = sitk.GetImageFromArray(data.astype(np.uint16))
        img_itk.SetSpacing(spacing=(0.89, 0.89, 3.0))
        sitk.WriteImage(img_itk, save_path)
    return 0

## find the index that have the largest summaized value
def find_nozero_largest_slice(mask_vol):
    if torch.is_tensor(mask_vol): # 11HWD
        slice_id = torch.argmax(torch.sum(mask_vol, dim=(0,1,2,3)))
    else:
        slice_id = np.argmax(np.sum(mask_vol, dim=(0,1,2,3)))
    
    return slice_id




def set_random_seed(seed=42, deterministic=True):
    """Fix random seed for reproducibility in NumPy and PyTorch."""
    random.seed(seed)            # Python built-in random
    np.random.seed(seed)         # NumPy random seed
    torch.manual_seed(seed)      # PyTorch random seed (CPU)
    torch.cuda.manual_seed(seed) # PyTorch random seed (GPU)
    torch.cuda.manual_seed_all(seed) # If using multiple GPUs

    if deterministic:
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False     # Disables benchmarking for reproducibility

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC.astype(np.uint8)