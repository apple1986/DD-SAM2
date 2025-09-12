
from skimage import measure, morphology
import numpy as np
from sympy import false
import torch
import random

##########################################
def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

## extract the bounding box from the mask
def get_bounding_box(box_func, GT_np, label_id, bbox_shift=0):
    marker_data_id = (GT_np == label_id).astype(np.uint8)
    marker_zids, _, _ = np.where(marker_data_id > 0) # get the z index of the  segmented lesion
    marker_zids = np.sort(np.unique(marker_zids))
    bbox_dict = {} # key: z_index, value: bbox
    for z in marker_zids:
        z_box_check = box_func(marker_data_id[z, :, :], bbox_shift=bbox_shift)
        bbox_dict[z] = z_box_check
    
    return bbox_dict, marker_zids


def extract_boxes(bbox_dict):
    all_boxes = []
    for z, box in bbox_dict.items():
        box = np.array(box)
        if box.ndim == 1:
            all_boxes.append((z, 0, box))
        else:
            for i, b in enumerate(box):
                all_boxes.append((z, i, b))
    return all_boxes


## find the centroid of the bounding box
def compute_centroid(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def group_by_centroid_proximity(boxes, max_dist=35):
    groups = []
    for z, obj_id, box in sorted(boxes):
        cx, cy = compute_centroid(box)
        matched = False
        for group in groups:
            last_z, _, last_box = group[-1]
            if abs(z - last_z) <= 1:
                last_cx, last_cy = compute_centroid(last_box)
                if np.hypot(cx - last_cx, cy - last_cy) < max_dist:
                    group.append((z, obj_id, box))
                    matched = True
                    break
        if not matched:
            groups.append([(z, obj_id, box)])
    return groups

def summarize(groups):
    summaries = []
    for group in groups:
        slices = sorted([z for z, _, _ in group])
        summaries.append((slices[0], slices[-1], slices[len(slices)//2]))
    return summaries

## generate the bounding box for the 3D mask
def generate_input_bbox(batch_size, image_depth, mr_vol, gt_vol, bbox_dict, device="cuda:0", image_size=512, NUM_SLICE=8):
    ## sample for the volume: random index a number to get the 8 slices, 4 groups, 
    ## random genertate 4 numbers in the range of 0 to D-8
    # NUM_SLICE = 8
    first_sel_idxs = random.sample(range(0, image_depth-NUM_SLICE), batch_size) # batchsize=4
    first_sel_bboxes = torch.zeros(batch_size, 4).to(device) # Bx4
    input_chuck = torch.zeros((batch_size, NUM_SLICE, 3, image_size, image_size)).to(device) # BT3HW
    gt_chuck = torch.zeros((batch_size, NUM_SLICE, 1, image_size, image_size)).to(device) # BT1HW
    for n, idx in enumerate(first_sel_idxs):
        input_chuck[n , ...] = mr_vol[idx:idx+NUM_SLICE, ...].unsqueeze(0) # 8x3x512x512
        gt_chuck[n , ...]  = gt_vol[0, :, :, :, idx:idx+NUM_SLICE].permute(3, 0, 1, 2).float() # 8x1x512x512
        if bbox_dict[idx].ndim > 1:
            first_sel_bboxes[n, ...] = torch.from_numpy(bbox_dict[idx][0]).to(device)
        else:
            first_sel_bboxes[n, ...] = torch.from_numpy(bbox_dict[idx]).to(device) # 4x4, the bbox from the first slice

    return input_chuck, gt_chuck, first_sel_bboxes


def generate_input_bbox_endvis(batch_size, image_depth, mr_vol, gt_vol, bbox_dict, device="cuda:0", image_size=512, NUM_SLICE=8):
    ## sample for the volume: random index a number to get the 8 slices, 4 groups, 
    ## random genertate 4 numbers in the range of 0 to D-8
    # NUM_SLICE = 8

    valid_keys = [k for k in bbox_dict.keys() if k <= image_depth - NUM_SLICE]
    assert len(valid_keys) >= batch_size, "Not enough valid slices with bounding boxes."

    first_sel_idxs = random.sample(valid_keys, batch_size)
    # first_sel_idxs = random.sample(range(0, image_depth-NUM_SLICE), batch_size) # batchsize=4


    
    first_sel_bboxes = torch.zeros(batch_size, 4).to(device) # Bx4
    input_chuck = torch.zeros((batch_size, NUM_SLICE, 3, image_size, image_size)).to(device) # BT3HW
    gt_chuck = torch.zeros((batch_size, NUM_SLICE, 1, image_size, image_size)).to(device) # BT1HW
    for n, idx in enumerate(first_sel_idxs):
        input_chuck[n , ...] = mr_vol[idx:idx+NUM_SLICE, ...].unsqueeze(0) # 8x3x512x512
        gt_chuck[n , ...]  = gt_vol[0, :, :, :, idx:idx+NUM_SLICE].permute(3, 0, 1, 2).float() # 8x1x512x512
        if bbox_dict[idx].ndim > 1:
            first_sel_bboxes[n, ...] = torch.from_numpy(bbox_dict[idx][0]).to(device)
        else:
            first_sel_bboxes[n, ...] = torch.from_numpy(bbox_dict[idx]).to(device) # 4x4, the bbox from the first slice

    return input_chuck, gt_chuck, first_sel_bboxes

import torch
import random

def generate_input_bbox_time_scale(batch_size, image_depth, mr_vol, gt_vol, bbox_dict,
                        device="cuda:0", image_size=512, NUM_SLICE=5,
                        time_scale_mode='multi'):
    """
    time_scale_mode: 'multi' will randomly choose from [1,2,3], else specify a fixed step
    """
    input_chuck = torch.zeros((batch_size, NUM_SLICE, 3, image_size, image_size), device=device)
    gt_chuck = torch.zeros((batch_size, NUM_SLICE, 1, image_size, image_size), device=device)
    first_sel_bboxes = torch.zeros(batch_size, 4, device=device)

    for n in range(batch_size):
        if time_scale_mode == 'multi':
            step = random.choice([1, 2, 3])
        else:
            step = int(time_scale_mode)  # use fixed step if provided

        # Generate index sequence with selected step
        idx_seq = [i * step for i in range(NUM_SLICE)]

        # Shifted start so that max(idx_seq) < image_depth
        max_start = image_depth - 1 - max(idx_seq)
        if max_start > 0:
            start_idx = random.randint(0, max_start)
            slice_indices = [start_idx + i for i in idx_seq]
        else:
            # fallback: take last NUM_SLICE slices with stride=1
            slice_indices = list(range(image_depth - NUM_SLICE, image_depth))

        for s_idx, slice_idx in enumerate(slice_indices):
            input_chuck[n, s_idx] = mr_vol[slice_idx]
            gt_chuck[n, s_idx] = gt_vol[0, :, :, :, slice_idx].unsqueeze(0).float()
            # Use bbox from the first frame in this sequence
            if s_idx == 0:
                if bbox_dict[slice_idx].ndim > 1:
                    first_sel_bboxes[n, ...] = torch.from_numpy(bbox_dict[slice_idx][0]).to(device)
                else:
                    first_sel_bboxes[n, ...] = torch.from_numpy(bbox_dict[slice_idx]).to(device)

    return input_chuck, gt_chuck, first_sel_bboxes


def generate_input_bbox_for_empty_bbox(batch_size, image_depth, mr_vol, gt_vol, bbox_dict, device="cuda:0", image_size=512, NUM_SLICE=8, MAX_AREA = false):
    ## sample for the volume: random index a number to get the 8 slices, 4 groups, 
    ## random genertate 4 numbers in the range of 0 to D-8
    # NUM_SLICE = 8
    first_sel_idxs = random.sample(range(0, image_depth-NUM_SLICE), batch_size) # batchsize=4
    # you need to confirm that there is bbox in this slice
    while all(key in bbox_dict for key in first_sel_idxs) is not True:
        first_sel_idxs = random.sample(range(0, image_depth-NUM_SLICE), batch_size) 



    first_sel_bboxes = torch.zeros(batch_size, 4).to(device) # Bx4
    input_chuck = torch.zeros((batch_size, NUM_SLICE, 3, image_size, image_size)).to(device) # BT3HW
    gt_chuck = torch.zeros((batch_size, NUM_SLICE, 1, image_size, image_size)).to(device) # BT1HW
    for n, idx in enumerate(first_sel_idxs):
        input_chuck[n , ...] = mr_vol[idx:idx+NUM_SLICE, ...].unsqueeze(0) # 8x3x512x512
        gt_chuck[n , ...]  = gt_vol[0, :, :, :, idx:idx+NUM_SLICE].permute(3, 0, 1, 2).float() # 8x1x512x512
        if bbox_dict[idx].ndim > 1:
            if MAX_AREA is True:
                # Compute areas
                areas = [(xmax - xmin) * (ymax - ymin) for xmin, ymin, xmax, ymax in bbox_dict[idx]]
                # Get the index of the maximum area
                max_index = np.argmax(areas)
                ## get the first bbox
                first_sel_bboxes[n, ...] = torch.from_numpy(bbox_dict[idx][max_index]).to(device)
            else:
                first_sel_bboxes[n, ...] = torch.from_numpy(bbox_dict[idx][0]).to(device)
        else:
            first_sel_bboxes[n, ...] = torch.from_numpy(bbox_dict[idx]).to(device) # 4x4, the bbox from the first slice

    return input_chuck, gt_chuck, first_sel_bboxes