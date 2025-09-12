import cv2
import numpy as np
import torch
import random
from scipy.ndimage import label, find_objects

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


###########################: find bounding box
def find_box(mask):
    # find box: the left-upper and right-bottom points
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    input_box = np.array([x, y, x+w, y+h])
    return input_box

def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image
    Shape: 255 x 256
    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def get_bbox256_torch(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)
    Notice: it only used for one object in a mask

    Parameters
    ----------
    mask_256 : BHW
        the mask of the resized image
    Shape: 255 x 256
    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
        bounding box coordinates in the resized image
    """
    B, H, W = mask_256.shape
    bboxes256 = torch.ones((B, 1, 4)).to(mask_256.device) * (-100)
    for n in range(B):
        pd_one = mask_256[n,:,:]
        idx_fg = torch.argwhere(pd_one > 0.5)
        if (idx_fg.sum() > 0):
            # print(idx_fg)
            # idx_bg = torch.argwhere(pd_one < 0.5)
            x_min, x_max = torch.min(idx_fg[:,1]), torch.max(idx_fg[:, 1])
            y_min, y_max = torch.min(idx_fg[:,0]), torch.max(idx_fg[:, 0])
            x_min = max(0, x_min - bbox_shift)
            x_max = min(W, x_max + bbox_shift)
            y_min = max(0, y_min - bbox_shift)
            y_max = min(H, y_max + bbox_shift)
            bboxes256[n, 0, :] = torch.tensor([x_min, y_min, x_max, y_max]) #Nx1x4

    return bboxes256

def get_bbox256_cv(mask_256, bbox_shift=3):
    B, H, W = mask_256.shape
    binary_mask = mask_256.detach().cpu().numpy()
    bboxes256 = np.ones((B, 1, 4))* (-100)#.to(mask_256.device) * (-100)
    for n in range(B):
        pd_one = binary_mask[n, :, :].astype(np.uint8)
        if (pd_one.sum()> 0):
            # Find contours in the binary mask
            contours, _ = cv2.findContours(pd_one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize variables to keep track of the largest bounding box
            max_area = 0
            
            for contour in contours:
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Update the largest bounding box if this one is bigger
                if area > max_area:
                    max_area = area
                    x_min = max(0, x - bbox_shift)
                    x_max = min(W, x + w + bbox_shift)
                    y_min = max(0, y - bbox_shift)
                    y_max = min(H, y + h + bbox_shift)
                    bboxes256[n, 0, :] = np.array([x_min, y_min, x_max, y_max])
    bboxes256 = torch.tensor(bboxes256).to(mask_256.device)

    return bboxes256  
#################################################
def get_multi_bounding_boxes_numpy(mask):
    """
    Get all bounding box coordinates from a binary mask using NumPy.
    mask: (H, W) array (binary: 0 or 1)
    Returns: List of bounding boxes [(x1, y1, x2, y2), ...]
    """
    objects = np.unique(mask)[1:]  # Exclude background (0)
    bounding_boxes = []

    for obj_id in objects:
        y_idxs, x_idxs = np.where(mask == obj_id)
        if len(y_idxs) == 0:
            continue

        x_min, x_max = x_idxs.min(), x_idxs.max()
        y_min, y_max = y_idxs.min(), y_idxs.max()

        bounding_boxes.append((x_min, y_min, x_max, y_max))
    
    ## convert to Nx4
    bounding_boxes = np.array(bounding_boxes)
    # print(f"bounding box shape: {bounding_boxes.shape}")

    return bounding_boxes

def get_multi_bounding_boxes_torch(mask):
    """
    Get all bounding box coordinates from a binary mask using PyTorch.
    mask: (H, W) tensor (binary: 0 or 1)
    Returns: List of bounding boxes [(x1, y1, x2, y2), ...]
    """
    # Get connected components (assumes mask is binary)
    num_objects = mask.max().item()  # Number of objects (1 if single mask)
    bounding_boxes = []

    # Iterate through each object in the mask
    for obj_id in range(1, num_objects + 1):
        obj_mask = (mask == obj_id).nonzero()  # Get (y, x) coordinates
        if obj_mask.numel() == 0:
            continue  # Skip if no object found
        
        y_min, x_min = obj_mask[:, 0].min().item(), obj_mask[:, 1].min().item()
        y_max, x_max = obj_mask[:, 0].max().item(), obj_mask[:, 1].max().item()

        bounding_boxes.append((x_min, y_min, x_max, y_max))
    ## convert to Nx4
    bounding_boxes = torch.tensor(bounding_boxes)
    # print(f"bounding box shape: {bounding_boxes.shape}")


    return bounding_boxes



##################point
def find_centroid(mask):
    # find the centeroid of the object
    # calculate moments of binary image
    # return xy direction
    M = cv2.moments(mask)
    # calculate x,y coordinate of center
    cCol = int(M["m10"] / M["m00"])
    cRow = int(M["m01"] / M["m00"])
    # print(cRow, " ", cCol)
    return np.array([[cCol, cRow]])
################################### find the high confident point
def find_position(unlabel_pd, conf_thresh=0.95):
    B, H, W = unlabel_pd.shape 
    temp_pd = unlabel_pd.view(B, -1)           
    M = temp_pd.argmax(1) # B
    ## if the gt is blank, it will result in error prompt
    is_null = temp_pd.sum(dim=1) < conf_thresh
    if is_null.sum() > 0:
        M[is_null] = 0
        # M[is_null] = torch.tensor(32896, device=M.device)
    #     print("Here")
    # else:
    #     print("is not null")

    idx = torch.cat(((M / H).view(-1, 1), (M % W).view(-1, 1)), dim=1).long()
    idx[:, [0,1]] = idx[:, [1,0]] # (Y,X) --> (X,Y)
    input_points = idx.unsqueeze(1)
    # input_labels = torch.ones((B, 1), device=args.cuda_num)
    input_labels = (~is_null).float().reshape(B, -1) #torch.ones((B, 1), device=args.cuda_num)
    point_label = (input_points, input_labels)
    return point_label

def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], point_label

def generate_click_prompt(img, msk, pt_label = 1):
    # return: img, prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0) # b 2
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)  # c b 2
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1) # b 2 d
    msk = torch.stack(msk_list, dim=-1) # b h w d
    msk = msk.unsqueeze(1) # b c h w d
    return img, pt, msk #[b, 2, d], [b, c, h, w, d]

def generate_unique_random_numbers(n, start, end):
    # generate n number in [start, end]
    return random.sample(range(start, end + 1), n)

def find_point_label(gt_sam):
    # get 10 points for gt
    B, _, _ = gt_sam.shape
    points_coord = torch.zeros((B, 10, 2), device=gt_sam.device)
    points_label = torch.zeros((B, 10), device=gt_sam.device)
    SEL_PT_NUM = 1 # how many positive poinsts are selected?
    for n in range(B):
        gt_one = gt_sam[n,:,:] # HW
        idx_fg = torch.argwhere(gt_one == 1) # 1 is the class label
        idx_bg = torch.argwhere(gt_one != 1)
        if len(idx_fg) == 0:
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            ## sample 10 points in a random way
            random_numbers = generate_unique_random_numbers(10, 0, len(idx_bg)-1)
            points_coord[n,:,:] = idx_bg[random_numbers] # 10x2
            # points_label[0,:] = torch.zeros((len(random_numbers)), device=gt_sam.device) # 1x10
        else:
            idx_fg[:, [0,1]] = idx_fg[:, [1,0]] # make [row, col] to [x, y]
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            # five points from object, five points for background
            random_numbers = generate_unique_random_numbers(SEL_PT_NUM, 0, len(idx_fg)-1)
            # foreground: points and labels
            points_coord[n,:SEL_PT_NUM,:] = idx_fg[random_numbers]
            points_label[n,:SEL_PT_NUM] = 1
            # backgrouond: points and labels
            random_numbers = generate_unique_random_numbers(10-SEL_PT_NUM, 0, len(idx_bg)-1)
            points_coord[n,SEL_PT_NUM:,] = idx_bg[random_numbers]
            points_label[n,SEL_PT_NUM:] = 0
    return points_coord, points_label

def find_point_label_pseudo(pd_sam):
    # input: pd_sam is a probability map
    # get 10 points for pseudo-label: 1 for highest probability, 9 for bg
    B, _, _ = pd_sam.shape
    points_coord = torch.zeros((B, 10, 2), device=pd_sam.device)
    points_label = torch.zeros((B, 10), device=pd_sam.device)
    for n in range(B):
        pd_one = pd_sam[n,:,:] # HW
        idx_fg = torch.argwhere(pd_one > 0.5).to(device=pd_sam.device) # 1 is the class label, 0.5 is threshold
        idx_bg = torch.argwhere(pd_one < 0.5).to(device=pd_sam.device) 
        if len(idx_fg) == 0:
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            ## sample 10 points in a random way
            random_numbers = generate_unique_random_numbers(10, 0, len(idx_bg)-1)
            points_coord[n,:,:] = idx_bg[random_numbers] # 10x2
            # points_label[0,:] = torch.zeros((len(random_numbers)), device=gt_sam.device) # 1x10
        else:
            idx_fg[:, [0,1]] = idx_fg[:, [1,0]] # make [row, col] to [x, y]
            idx_bg[:, [0,1]] = idx_bg[:, [1,0]] # make [row, col] to [x, y]
            # five points from object, five points for background
            random_numbers = generate_unique_random_numbers(1, 0, len(idx_fg)-1)
            # foreground: points and labels
            points_coord[n, :1, :] = idx_fg[random_numbers]
            points_label[n, :1] = 1
            # backgrouond: points and labels
            random_numbers = generate_unique_random_numbers(9, 0, len(idx_bg)-1)
            points_coord[n, 1:, :] = idx_bg[random_numbers]
            points_label[n, 1:] = 0
    return points_coord, points_label

###########################
def get_one_bounding_boxes_with_ccl(mask):
    """
    Extract bounding boxes for all connected components of each labeled object in a segmentation mask.

    Args:
        mask (torch.Tensor): A (H, W) tensor where each unique value represents a cell label.

    Returns:
        dict: {label: [(x_min, y_min, x_max, y_max), ...]}
    """
    if torch.torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()  # Convert to NumPy for processing
    else:
        mask_np = mask
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)


    sel_bbox = np.ones((4))* (-100)#.to(mask_256.device) * (-100), save the mask
    if len(unique_labels) < 1: # empty, there is no box in the masl
        sel_bbox = np.ones((4))* (-100)
        return sel_bbox

    bboxes = {}
    for label_value in unique_labels:
        binary_mask = (mask_np == label_value).astype(np.int32)  # Create a binary mask for the current label
        
        # Apply connected component labeling
        labeled_mask, num_components = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        boxes = []
        
        for s in slices:
            if s is not None:
                y_min, x_min = s[0].start, s[1].start
                y_max, x_max = s[0].stop - 1, s[1].stop - 1
                boxes.append((x_min, y_min, x_max, y_max))

        bboxes[label_value] = boxes
    ## randomly select one bboxes and save as tensor
    if len(bboxes[unique_labels[0]]) > 1:
        ## get and rand int
        NUM = torch.randint(0, len(bboxes[unique_labels[0]]), (1,)).numpy()[0]
        sel_bbox = np.array(bboxes[unique_labels[0]][NUM])
    else:
        sel_bbox = np.array(bboxes[unique_labels[0]][0])
    

    return sel_bbox


def get_all_bounding_boxes_with_ccl(mask):
    """
    Extract bounding boxes for all connected components of each labeled object in a segmentation mask.

    Args:
        mask (torch.Tensor): A (H, W) tensor where each unique value represents a cell label.

    Returns:
        dict: {label: [(x_min, y_min, x_max, y_max), ...]}
    """
    if torch.torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()  # Convert to NumPy for processing
    else:
        mask_np = mask
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)


    sel_bbox = np.ones((1, 4))* (-100)#.to(mask_256.device) * (-100), save the mask, 1x4
    if len(unique_labels) < 1: # empty, there is no box in the masl
        sel_bbox = np.ones((1, 4))* (-100)
        return sel_bbox

    bboxes = {}
    for label_value in unique_labels:
        binary_mask = (mask_np == label_value).astype(np.int32)  # Create a binary mask for the current label
        
        # Apply connected component labeling
        labeled_mask, num_components = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        boxes = []
        
        for s in slices:
            if s is not None:
                y_min, x_min = s[0].start, s[1].start
                y_max, x_max = s[0].stop - 1, s[1].stop - 1
                boxes.append((x_min, y_min, x_max, y_max))

        bboxes[label_value] = boxes
    ## select all bboxes and save as tensor
    if len(bboxes[unique_labels[0]]) > 1:
        sel_bbox = []
        for n in range(len(bboxes[unique_labels[0]])):
            sel_bbox.append(np.array(bboxes[unique_labels[0]][n]))
        sel_bbox = np.array(sel_bbox)
    else:
        sel_bbox = np.array(bboxes[unique_labels[0]][0])
    
    
    return sel_bbox

def get_all_bounding_boxes_with_ccl_box_fix(mask, bbox_shift=0):
    """
    Extract bounding boxes for all connected components of each labeled object in a segmentation mask.

    Args:
        mask (torch.Tensor): A (H, W) tensor where each unique value represents a cell label.

    Returns:
        dict: {label: [(x_min, y_min, x_max, y_max), ...]}
    """
    if torch.torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()  # Convert to NumPy for processing
    else:
        mask_np = mask
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)


    sel_bbox = np.ones((1, 4))* (-100)#.to(mask_256.device) * (-100), save the mask, 1x4
    if len(unique_labels) < 1: # empty, there is no box in the masl
        sel_bbox = np.ones((1, 4))* (-100)
        return sel_bbox

    bboxes = {}
    for label_value in unique_labels:
        binary_mask = (mask_np == label_value).astype(np.int32)  # Create a binary mask for the current label
        
        # Apply connected component labeling
        labeled_mask, num_components = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        boxes = []
        
        for s in slices:
            if s is not None:
                y_min, x_min = max(0, s[0].start - bbox_shift), max(0, s[1].start - bbox_shift)
                y_max, x_max = max(0, s[0].stop - 1 + bbox_shift), max(0, s[1].stop - 1 + bbox_shift)
                boxes.append((x_min, y_min, x_max, y_max))

        bboxes[label_value] = boxes
    ## select all bboxes and save as tensor
    if len(bboxes[unique_labels[0]]) > 1:
        sel_bbox = []
        for n in range(len(bboxes[unique_labels[0]])):
            sel_bbox.append(np.array(bboxes[unique_labels[0]][n]))
        sel_bbox = np.array(sel_bbox)
    else:
        sel_bbox = np.array(bboxes[unique_labels[0]][0])
    
    
    return sel_bbox

def get_all_bounding_boxes_with_ccl_box_rand(mask, bbox_shift=5):
    """
    Extract bounding boxes for all connected components of each labeled object in a segmentation mask.

    Args:
        mask (torch.Tensor): A (H, W) tensor where each unique value represents a cell label.

    Returns:
        dict: {label: [(x_min, y_min, x_max, y_max), ...]}
    """
    set_random_seed(2024)
    if torch.torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()  # Convert to NumPy for processing
    else:
        mask_np = mask
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background (label 0)


    sel_bbox = np.ones((1, 4))* (-100)#.to(mask_256.device) * (-100), save the mask, 1x4
    if len(unique_labels) < 1: # empty, there is no box in the masl
        sel_bbox = np.ones((1, 4))* (-100)
        return sel_bbox

    bboxes = {}
    for label_value in unique_labels:
        binary_mask = (mask_np == label_value).astype(np.int32)  # Create a binary mask for the current label
        
        # Apply connected component labeling
        labeled_mask, num_components = label(binary_mask)

        # Extract bounding boxes for each connected component
        slices = find_objects(labeled_mask)
        boxes = []
        
        for s in slices:
            if s is not None:
                # y_min, x_min = max(0, s[0].start - bbox_shift), max(0, s[1].start - bbox_shift)
                # y_max, x_max = max(0, s[0].stop - 1 + bbox_shift), max(0, s[1].stop - 1 + bbox_shift)
                y_min = max(0,  s[0].start - random.randint(0, bbox_shift))
                x_min = max(0,  s[1].start - random.randint(0, bbox_shift))
                y_max = max(0,  s[0].stop - 1 + random.randint(0, bbox_shift))
                x_max = max(0,  s[1].stop - 1 + random.randint(0, bbox_shift))
                boxes.append((x_min, y_min, x_max, y_max))

        bboxes[label_value] = boxes
    ## select all bboxes and save as tensor
    if len(bboxes[unique_labels[0]]) > 1:
        sel_bbox = []
        for n in range(len(bboxes[unique_labels[0]])):
            sel_bbox.append(np.array(bboxes[unique_labels[0]][n]))
        sel_bbox = np.array(sel_bbox)
    else:
        sel_bbox = np.array(bboxes[unique_labels[0]][0])
    
    
    return sel_bbox


if __name__=="__main__":
    mask = torch.zeros((100, 100), dtype=torch.uint8)
    mask[30:60, 40:80] = 1  # Example object 1
    mask[10:20, 10:30] = 1  # Example object 2

    bboxes = get_all_bounding_boxes_with_ccl_box_fix(mask, bbox_shift=1)
    print(bboxes)  # [(40, 30, 79, 59), (10, 10, 29, 19)]