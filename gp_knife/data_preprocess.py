import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F


#########################
def resample_vol_torch(img_vol, original_spacing, new_spacing, mode="trilinear",  align_corners=True):
    """
    img_vol: HWD
    original_spacing: original resolution
    new_spacing: resample resolution

    """
    img_vol = torch.from_numpy(img_vol)
    if mode == "nearest":
        img_vol = img_vol.float()
        align_corners = None
    # original_size = img_vol.shape
    img_vol = img_vol[None, None, ...] # BCHWD
    scale_factor_np = original_spacing / new_spacing
    img_vol_resample = F.interpolate(img_vol, scale_factor=(scale_factor_np[0], scale_factor_np[1], scale_factor_np[2]), mode=mode, align_corners=align_corners)
    # img_vol_resample = img_vol_resample.numpy()[0, 0, ...]
    
    return img_vol_resample

def pad_or_crop_3d(image, target_shape):
    """
    Pads or crops a 3D medical image to the desired shape.
    
    Args:
        image (torch.Tensor): Input image of shape (C, D, H, W) or (D, H, W)
        target_shape (tuple): Desired (D, H, W) size

    Returns:
        torch.Tensor: Resized image with shape (C, D, H, W) or (D, H, W)
    """
    assert len(image.shape) in [3, 4, 5], "Image must be (D, H, W) or (C, D, H, W) or or (B, C, D, H, W)"
    
    if len(image.shape) == 5:
        image = image.squeeze(0)  # remove batch channel if exists (Batch)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add channel if missing (C=1)

    C, D, H, W = image.shape
    target_D, target_H, target_W = target_shape

    # Compute padding/cropping
    pad_d = max(target_D - D, 0)
    pad_h = max(target_H - H, 0)
    pad_w = max(target_W - W, 0)

    crop_d = max(D - target_D, 0)
    crop_h = max(H - target_H, 0)
    crop_w = max(W - target_W, 0)

    # Padding (only if the image is smaller than target)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pad_dims = (pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2,
                    pad_d // 2, pad_d - pad_d // 2)  # Symmetric padding
        image = F.pad(image, pad_dims, mode='constant', value=0)

    # Cropping (only if the image is larger than target)
    if crop_d > 0 or crop_h > 0 or crop_w > 0:
        crop_start_d = crop_d // 2
        crop_start_h = crop_h // 2
        crop_start_w = crop_w // 2
        image = image[:, crop_start_d:crop_start_d + target_D,
                         crop_start_h:crop_start_h + target_H,
                         crop_start_w:crop_start_w + target_W]
    ## convert to  numpy
    if len(image.shape) == 4 and C == 1:
        image = image.squeeze(0).numpy()
    else:
        image = image.numpy()

    return image

def save_nii(data, save_path, spacing=(1,1,1)):
    ## Notice: need change HWD to DHW for the data, then it will save as HWD
    img_itk = sitk.GetImageFromArray(data.astype(np.uint16))
    img_itk.SetSpacing(spacing)
    sitk.WriteImage(img_itk, save_path)