import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from gp_knife.plot_mask_bbox_np import show_box, get_bbox
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_fix


## overly mask on mr
def plot_overlay_bbox_all(image_path, mask_path, output_path, slice_index=None):
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
    print(f"Select slice: {slice_index}")
    
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_slice, cmap='gray')
    # plt.imshow(mask_slice, cmap='jet', alpha=0.5)  # Overlay mask with transparency
    # plt.axis('off')

    ## Find bbox from the binary mask
    # bboxes = get_bbox(mask_slice, bbox_shift=0)
    bbox_shift = 5
    bboxes_all = get_all_bounding_boxes_with_ccl_box_fix(mask_slice, bbox_shift=bbox_shift)
    # # Choose the largest bounding box based on the area
    # if len(bboxes_all) > 1:
    #     bboxes = max(bboxes_all, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))  # (x2 - x1) * (y2 - y1)
    # else:
    #     bboxes = bboxes_all[0]

    # Display the image with boundary overlay
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_slice, cmap="gray")  # Convert BGR to RGB for display
    ## plot the bbox on the mr slice
    if bboxes_all.ndim > 1:
        for n in range(0, bboxes_all.shape[0]):
            show_box(bboxes_all[n], ax)
    else:
        show_box(bboxes_all, ax)
    plt.axis("off")
    # title_name = f"MRI Image with BBox Overlay-shift {bbox_shift}, slice {slice_index}"
    # plt.title(title_name)
    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0) # , pad_inches=0
    plt.close()

def plot_overlay_mask_boundary_all(image_path, mask_path, output_path, slice_index=None):
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
    # If multiple contours are found, select the largest one
    # if len(contours) > 1:
    #     contours = [max(contours, key=cv2.contourArea)]
    # Convert grayscale MRI image to BGR for color overlay
    mri_colored = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
    # Draw the contours (boundary) on the image in red (or any color)
    cv2.drawContours(mri_colored, contours, -1, (0, 0, 255), thickness=1)  # Red boundary

    # Display the image with boundary overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(mri_colored, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.axis("off")
    # plt.title("MRI Image with Boundary Overlay")
    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def crop_black_background(input_path, output_path, threshold=10):
    """
    Crop the black background from an image.
    
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.
        threshold (int): Intensity threshold to distinguish background.
    """
    # Load the image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Error: Unable to load the image at {input_path}")
        return

    # Convert to grayscale if the image is in color (BGR) or has alpha (BGRA)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply a threshold to remove faint background
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find non-zero coordinates after thresholding
    coords = np.column_stack(np.nonzero(binary))
    if coords.size == 0:
        print("Error: The image contains only background after thresholding.")
        return
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the original image using the calculated bounding box
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]

    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to {output_path}")

def crop_black_background_basedMR(input_path, mask_path, output_path, threshold=10):
    """
    Crop the black background from an image and the corresponding mask.
    
    Args:
        input_path (str): Path to the input MR image.
        mask_path (str): Path to the corresponding mask image.
        output_path (str): Path to save the cropped image and mask.
        threshold (int): Intensity threshold to distinguish background.
    """

    # Load the MR image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if image is None or mask is None:
        print(f"Error: Unable to load the image or mask at {input_path} or {mask_path}")
        return

    # Convert to grayscale if the image is in color (BGR) or has alpha (BGRA)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply a threshold to remove faint background
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find non-zero coordinates after thresholding
    coords = np.column_stack(np.nonzero(binary))
    if coords.size == 0:
        print("Error: The image contains only background after thresholding.")
        return
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the original image and mask using the calculated bounding box
    # cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

    # Concatenate the output paths for saving both images
    # output_image_path = output_path.replace(".png", "_cropped.png")
    # output_mask_path = output_path.replace(".png", "_cropped_mask.png")

    # Save the cropped images
    # cv2.imwrite(output_image_path, cropped_image)
    cv2.imwrite(output_path, cropped_mask)

    # print(f"Cropped image saved to {output_image_path}")
    print(f"Cropped mask saved to {output_path}")