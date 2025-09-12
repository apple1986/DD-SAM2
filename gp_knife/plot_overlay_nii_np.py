import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswlits3d_v2")
import numpy as np
import os
import matplotlib.pyplot as plt 
import cv2
from glob import glob
import shutil
import nibabel as nib
from gp_knife.plot_mask_bbox_np import show_box, get_bbox
import torch



## overly mask on mr
def plot_overlay_bbox(image_path, mask_path, output_path):
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()
    
    if image is None or mask is None:
        print(f"Error loading image or mask: {image_path}, {mask_path}")
        return
    
    # Assuming the image and mask are 3D and we want to plot the middle slice
    slice_index = np.argmax(np.sum(mask, axis=(0, 1)))
    image_slice = image[:, :, slice_index]
    image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
    image_slice = image_slice.astype(np.uint8)
    mask_slice = mask[:, :, slice_index].astype(np.uint8)
    
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_slice, cmap='gray')
    # plt.imshow(mask_slice, cmap='jet', alpha=0.5)  # Overlay mask with transparency
    # plt.axis('off')

    ## Find bbox from the binary mask
    bboxes = get_bbox(mask_slice, bbox_shift=10)

    # Display the image with boundary overlay
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(image_slice, cmap="gray")  # Convert BGR to RGB for display
    ## plot the bbox on the mr slice
    show_box(bboxes, ax)
    plt.axis("off")
    plt.title("MRI Image with BBox Overlay-shift=10")
    # plt.show()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_overlay_mask_boundary(image_path, mask_path, output_path):
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    
    image = image_nii.get_fdata()
    mask = mask_nii.get_fdata()
    
    if image is None or mask is None:
        print(f"Error loading image or mask: {image_path}, {mask_path}")
        return
    
    # Assuming the image and mask are 3D and we want to plot the middle slice
    slice_index = np.argmax(np.sum(mask, axis=(0, 1)))
    image_slice = image[:, :, slice_index]
    image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
    image_slice = image_slice.astype(np.uint8)
    mask_slice = mask[:, :, slice_index].astype(np.uint8)
    
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_slice, cmap='gray')
    # plt.imshow(mask_slice, cmap='jet', alpha=0.5)  # Overlay mask with transparency
    # plt.axis('off')

    # Find contours from the binary mask
    contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert grayscale MRI image to BGR for color overlay
    mri_colored = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
    # Draw the contours (boundary) on the image in red (or any color)
    cv2.drawContours(mri_colored, contours, -1, (0, 0, 255), thickness=2)  # Red boundary

    # Display the image with boundary overlay
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(mri_colored, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.axis("off")
    plt.title("MRI Image with Boundary Overlay")
    # plt.show()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


############################################
def plot_overlay_mask_boundary_simply(curMR, curGT):
    """
    curMR: HWD
    curGT: HWD
    """

    if torch.is_tensor(curMR):
        image = curMR.detach().cpu().numpy() 
        mask = curGT.detach().cpu().numpy()

    # Assuming the image and mask are 3D and we want to plot the middle slice
    slice_index = np.argmax(np.sum(mask, axis=(0, 1)))
    image_slice = image[:, :, slice_index]
    image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
    image_slice = image_slice.astype(np.uint8)
    mask_slice = mask[:, :, slice_index].astype(np.uint8)
    
    ## Find bbox from the binary mask
    bboxes = get_bbox(mask_slice, bbox_shift=5)
    print(f"slice ID: {slice_index}, bbox: {bboxes}")

    # Display the image with boundary overlay
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(image_slice, cmap="gray")  # Convert BGR to RGB for display
    ## plot the bbox on the mr slice
    show_box(bboxes, ax)

    # Find contours from the binary mask
    contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert grayscale MRI image to BGR for color overlay
    mri_colored = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
    # Draw the contours (boundary) on the image in red (or any color)
    cv2.drawContours(mri_colored, contours, -1, (0, 0, 255), thickness=2)  # Red boundary
    ax.imshow(cv2.cvtColor(mri_colored, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display


    plt.axis("off")
    plt.title("MRI Image with BBox Overlay-shift=5")
    plt.show()


def overlay_pd_gt_onMR(mr_slice, prediction, ground_truth, alpha=0.5, save_name=None):
    """
    Overlay prediction results on an MR slice with ground truth boundary.
    
    Args:
        mr_slice (numpy.ndarray): 2D grayscale MRI slice.
        prediction (numpy.ndarray): 2D binary prediction mask (same shape as MR slice).
        ground_truth (numpy.ndarray): 2D binary ground truth mask (same shape as MR slice).
        alpha (float): Transparency factor for the overlay (0: fully transparent, 1: fully opaque).
    
    Returns:
        overlayed_image (numpy.ndarray): The final image with overlays.
    """

    # Ensure inputs are binary masks (0 and 1)
    prediction = (prediction > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)

    # Convert grayscale MR slice to 3-channel (RGB)
    mr_rgb = cv2.cvtColor(mr_slice, cv2.COLOR_GRAY2RGB)

    # Define colors
    prediction_color = [0, 255, 0]  # Green overlay for prediction
    gt_boundary_color = (0, 0, 255)  # Red contour for ground truth

    # Create an overlay with transparency where the mask is nonzero
    overlay = np.zeros_like(mr_rgb, dtype=np.uint8)
    overlay[prediction > 0] = prediction_color  # Apply color to predicted regions

    # Blend the overlay with the original image (alpha blending)
    blended = mr_rgb.copy()
    mask_indices = prediction > 0
    blended[mask_indices] = cv2.addWeighted(mr_rgb, 1 - alpha, overlay, alpha, 0)[mask_indices]

    # Find contours of the ground truth mask and draw on the blended image
    contours, _ = cv2.findContours(ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, gt_boundary_color, thickness=1)

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    plt.axis("off")
    # plt.title("MRI Image with Segmentation Overlay")
    # plt.show()

    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        # plt.savefig("figure.png", bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.title("MRI Image with Segmentation Overlay")
        plt.show()

    return blended