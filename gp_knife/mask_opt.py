import torch
import torch.nn.functional as F

def morph_operation(masks, operation="dilate", kernel_size=3, iterations=1):
    """
    Perform dilation or erosion on a batch of 2D masks.

    Args:
        masks (torch.Tensor): Binary masks of shape (B, C, H, W).
        operation (str): "dilate" or "erode".
        kernel_size (int): Size of the structuring element (should be odd).
        iterations (int): Number of times to apply the operation.

    Returns:
        torch.Tensor: Processed masks of shape (B, C, H, W).
    """
    assert operation in ["dilate", "erode"], "Operation must be 'dilate' or 'erode'"
    
    # Create a structuring element (ones kernel)
    padding = kernel_size // 2
    # kernel = torch.ones((1, 1, kernel_size, kernel_size), device=masks.device)

    for _ in range(iterations):
        if operation == "dilate":
            masks = F.max_pool2d(masks, kernel_size, stride=1, padding=padding)
        elif operation == "erode":
            masks = -F.max_pool2d(-masks, kernel_size, stride=1, padding=padding)
    
    return masks


def morph_operation_gpu(masks, operation="dilate", kernel_size=3, iterations=1):
    """
    Perform GPU-accelerated dilation or erosion on a batch of 2D masks.

    Args:
        masks (torch.Tensor): Binary masks of shape (B, C, H, W) on GPU.
        operation (str): "dilate" or "erode".
        kernel_size (int): Size of the structuring element (should be odd).
        iterations (int): Number of times to apply the operation.

    Returns:
        torch.Tensor: Processed masks of shape (B, C, H, W).
    """
    assert operation in ["dilate", "erode"], "Operation must be 'dilate' or 'erode'"
    
    # Ensure masks are on GPU
    masks = masks.to("cuda", dtype=torch.float32)

    # Compute padding size
    padding = kernel_size // 2

    for _ in range(iterations):
        if operation == "dilate":
            masks = F.max_pool2d(masks, kernel_size, stride=1, padding=padding)
        elif operation == "erode":
            masks = -F.max_pool2d(-masks, kernel_size, stride=1, padding=padding)

    return masks

def morph3d(volume, kernel_size=3, operation='dilate'):
    """
    Perform 3D morphological operations (erosion or dilation) on a 5D tensor (BCHWD).

    Args:
        volume (torch.Tensor): Input 3D volume of shape (B, C, H, W, D)
        kernel_size (int): Size of the structuring element (must be odd)
        operation (str): Either 'dilate' or 'erode'

    Returns:
        torch.Tensor: Processed volume after morphological operation
    """
    assert operation in ['dilate', 'erode'], "Operation must be 'dilate' or 'erode'"

    # Define structuring element (3D kernel filled with ones)
    # kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=volume.device)

    # Apply padding to maintain size
    pad = kernel_size // 2
    volume_padded = F.pad(volume, (pad, pad, pad, pad, pad, pad), mode='replicate')

    # Use 3D convolution for morphological operation
    if operation == 'dilate':
        output = F.max_pool3d(volume_padded, kernel_size, stride=1)
    else:  # Erosion
        output = -F.max_pool3d(-volume_padded, kernel_size, stride=1)

    return output

def binary_morph3d(volume, kernel_size=3, operation='dilate'):
    """
    Perform 3D morphological operations (erosion or dilation) on a binary mask (BCHWD).
    
    Args:
        volume (torch.Tensor): Binary input volume of shape (B, C, H, W, D) with values in {0, 1}
        kernel_size (int): Size of the structuring element (must be >=1)
        operation (str): 'dilate' or 'erode'
    
    Returns:
        torch.Tensor: Processed binary volume with the same shape as input (B, C, H, W, D)
    """
    assert operation in ['dilate', 'erode'], "Operation must be 'dilate' or 'erode'"
    assert kernel_size >= 1, "Kernel size must be at least 1"

    # Define the structuring element (3D kernel)
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=volume.device)

    # Compute padding: If kernel size is even, we need asymmetric padding
    pad_front = kernel_size // 2
    pad_back = kernel_size - pad_front - 1  # Ensures correct shape for even sizes

    # Manually apply padding before convolution for even kernel sizes
    volume_padded = F.pad(volume, (pad_front, pad_back, pad_front, pad_back, pad_front, pad_back), mode='constant', value=0)

    # Apply 3D convolution
    conv_output = F.conv3d(volume_padded.float(), kernel, stride=1)

    # Apply morphological operation
    if operation == 'dilate':
        output = (conv_output > 0).float()  # Any 1 in the window -> 1
    else:  # Erosion
        output = (conv_output == kernel.sum()).float()  # All 1s in the window -> 1, else 0

    return output



if __name__ == "__main__":
    # Example Usage on GPU
    B, C, H, W = 2, 1, 10, 10  # Batch size 2, 1 channel, 10x10 mask
    masks = torch.zeros((B, C, H, W), device="cuda")
    masks[:, :, 3:7, 3:7] = 1  # Create a square region of ones

    # Perform dilation and erosion
    dilated_masks = morph_operation_gpu(masks, operation="dilate", kernel_size=3, iterations=1)
    eroded_masks = morph_operation_gpu(masks, operation="erode", kernel_size=3, iterations=1)

    print(masks[0, 0].cpu().numpy())
    print(dilated_masks[0, 0].cpu().numpy())  # Move to CPU for visualization
    print(eroded_masks[0, 0].cpu().numpy())  # Move to CPU for visualization