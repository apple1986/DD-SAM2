import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

def boundary_loss(pred, target):
    """
    Boundary loss using distance transform.
    
    Args:
        pred (torch.Tensor): Model predictions (B, C, H, W, D) with values in [0,1].
        target (torch.Tensor): Ground truth (B, C, H, W, D) with values in {0,1}.
    
    Returns:
        torch.Tensor: Boundary loss value.
    """
    target_np = target.cpu().numpy()
    dist_map = torch.tensor(distance_transform_edt(1 - target_np), device=pred.device)

    pred_softmax = torch.sigmoid(pred)  # Ensure prediction is in [0,1]
    loss = (pred_softmax * dist_map).mean()  # Higher penalty for errors near boundaries
    return loss

def hausdorff_loss(pred, target):
    """
    Hausdorff Distance-based loss using distance transform.
    
    Args:
        pred (torch.Tensor): Model predictions (B, C, H, W, D) in [0,1].
        target (torch.Tensor): Ground truth (B, C, H, W, D) in {0,1}.
    
    Returns:
        torch.Tensor: Hausdorff distance loss.
    """
    target_np = target.cpu().numpy()
    pred_np = (torch.sigmoid(pred) > 0.5).cpu().numpy()

    dist_target = torch.tensor(distance_transform_edt(1 - target_np), device=pred.device)
    dist_pred = torch.tensor(distance_transform_edt(1 - pred_np), device=pred.device)

    loss = (dist_target * pred + dist_pred * target).mean()
    return loss

def perimeter_loss(pred):
    """
    Perimeter loss to encourage smooth segmentation masks.
    
    Args:
        pred (torch.Tensor): Model predictions (B, C, H, W, D) in [0,1].
    
    Returns:
        torch.Tensor: Perimeter loss.
    """
    grad_x = torch.abs(pred[:, :, :-1, :, :] - pred[:, :, 1:, :, :])  # Gradient along X
    grad_y = torch.abs(pred[:, :, :, :-1, :] - pred[:, :, :, 1:, :])  # Gradient along Y
    grad_z = torch.abs(pred[:, :, :, :, :-1] - pred[:, :, :, :, 1:])  # Gradient along Z

    loss = (grad_x + grad_y + grad_z).mean()
    return loss

def combined_shape_loss(pred, target, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Combines Dice, Boundary, and Perimeter losses.
    
    Args:
        pred (torch.Tensor): Model predictions (B, C, H, W, D) in logits.
        target (torch.Tensor): Ground truth (B, C, H, W, D) in {0,1}.
        alpha, beta, gamma (float): Weight factors for each loss.
    
    Returns:
        torch.Tensor: Combined loss.
    """
    dice_loss = 1 - (2 * (pred * target).sum() + 1) / (pred.sum() + target.sum() + 1)
    bound_loss = boundary_loss(pred, target)
    perim_loss = perimeter_loss(pred)

    return alpha * dice_loss + beta * bound_loss + gamma * perim_loss
