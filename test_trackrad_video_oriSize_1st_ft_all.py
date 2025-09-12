import sys
sys.path.append("/home/gxu/proj1/lesionSeg/utswtumor_track")
sys.path.append("/home/gxu/proj1/lesionSeg/utswtumor_track/MedSAM2_new")
sys.path.append("/home/gxu/proj1/lesionSeg/utswtumor_track/MedSAM2_new/sam2")

import numpy as np
from os.path import join
from os import makedirs, listdir
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import SimpleITK as sitk
import os
import argparse
import shutil

from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor_npz
from sam2.utils.transforms import SAM2Transforms

from torch import multiprocessing as mp
from torchvision import transforms
from dataset.trackrad2025 import ToTensor, CenterCrop
from builders.trackrad_builder import load_data
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_fix, get_all_bounding_boxes_with_ccl_box_rand, get_all_bounding_boxes_with_ccl
from gp_knife.freq_func import set_random_seed
from gp_knife.eval_model import get_dice_hd_nsd_asd
from medpy.metric import dc
from skimage import measure, morphology

## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)
###########################
## set parameters
parser = argparse.ArgumentParser(
    description="Run inference on validation set with MedSAM2"
)
parser.add_argument(
    "-data_root",
    type=str,
    default=None,
    help="Path to the data folder",
)
parser.add_argument(
    "-pred_save_dir",
    type=str,
    default="/home/gxu/proj1/lesionSeg/utswtumor_track/MedSAM2_new/checkpoints",
    help="Path to save the segmentation results",
)
parser.add_argument(
    "-sam2_checkpoint",
    type=str,
    default='/home/gxu/proj1/lesionSeg/utswtumor_track/MedSAM2_new/checkpoints/medsam2/MedSAM2_latest.pt',
    help="SAM2 pretrained model checkpoint",
)
parser.add_argument(
    "-model_cfg",
    type=str,
    default="configs/sam2.1_hiera_t512.yaml",
    help="Model config file"
)
parser.add_argument("-device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-bbox_shift",
    type=int,
    default=5,
    help="Bounding box perturbation",
)
parser.add_argument(
    "-num_workers",
    type=int,
    default=16,
    help="Number of workers for multiprocessing",
)
# add option to propagate with either box or mask
parser.add_argument(
    '--propagate_with_box',
    default=True,
    action='store_true',
    help='whether to propagate with box'
)
parser.add_argument("--visualize", default=True, help="Save the .nii.gz slice segmentation results")
args = parser.parse_args()

## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)


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

def infer_video_mode(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box):
    segs_3D = np.zeros((img_resized.shape[0], H, W), dtype=np.uint8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, H, W)
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset,
                                                obj_id=1,
                                                box=bbox,
                                            )
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)
        ## do the reverse inference
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset,
                                                obj_id=1,
                                                box=bbox,
                                            )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)
    return segs_3D


def infer_video_mode_one_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box):
    ## XX: bbox should from the first slice and the last slice

    segs_3D = np.zeros((img_resized.shape[0], H, W), dtype=np.uint8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, H, W)
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset,
                                                obj_id=1,
                                                box=bbox,
                                            )
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)

    return segs_3D


def infer_video_mode_two_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box):
    segs_3D = np.zeros((img_resized.shape[0], H, W), dtype=np.uint8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, H, W)
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset,
                                                obj_id=1,
                                                box=bbox,
                                            )
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)
        ## do the reverse inference
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset+img_resized.shape[0]-1,
                                                obj_id=1,
                                                box=bbox, ##?? wrong: you need to use the last slice
                                            )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        predictor.reset_state(inference_state)
    return segs_3D


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

#######################################
## prepare hyperparameters
image_size = 512
args.visualize = False
data_root = args.data_root
args.bbox_shift = 0
device = "cuda:0"  # args.device

args.ch3 = True
flag_fix = False # False True
tv_types = ['test'] # 'train','val',

if flag_fix is True:
    box_func = get_all_bounding_boxes_with_ccl_box_fix 
else:
    box_func = get_all_bounding_boxes_with_ccl_box_rand


## set hyperparameters
root_path = "/home/gxu/proj1/lesionSeg/utswtumor_track"
model_cfg = args.model_cfg
checkpoint = args.sam2_checkpoint
checkpoint = "/home/gxu/proj1/lesionSeg/utswtumor_track/MedSAM2_new/checkpoints/medsam2_ft/ft_all/ft_all_var_lr/medsam2_best.pth"
num_workers = args.num_workers
propagate_with_box = args.propagate_with_box
use_ft = True # use the original model or the fine-tuned model
#############################
# initialized predictor

predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint, use_ft=use_ft)
save_eval_path_root = os.path.join(args.pred_save_dir, "medsam2")
os.makedirs(save_eval_path_root, exist_ok=True)
# load evaluation function
dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()
NUM = 0

## inference
for tv_type in tv_types:
    ## save segmentation results
    save_eval_path = os.path.dirname(checkpoint)
    makedirs(save_eval_path, exist_ok=True)
    ## copy the current testing script
    script_path = join(save_eval_path, os.path.basename(__file__))
    if not os.path.exists(script_path):
        shutil.copyfile(
            __file__, script_path
        )
    args.bbox_shift =  NUM # bbox shift
    sam2_transforms = SAM2Transforms(resolution=512, mask_threshold=0)

    ## load data
    all_test_transform = transforms.Compose([
            # CenterCrop(patch_size),
            ToTensor(),
    ])
    testLoader = load_data(root_path, all_test_transform, batch_size=1, split=tv_type) # test

    dice_all = [] # save all data
    hd95_all = []
    nsd_all = []
    asd_all = []
    cnt_case = 0
    for data_four in testLoader:
        cnt_case = cnt_case + 1
        print(f"cnt case {cnt_case}")
        curMR, curGT, firstGT = data_four["curMR"], data_four["curGT"], data_four["firstGT"]
        caseID = data_four["caseID"][0]
        curMR = curMR.to(device) # BCHWD
        curGT = curGT.to(device) # BCHWD
        firstGT = firstGT.to(device) # B1HWD

        ori_H, ori_W, ori_D = curMR.shape[2], curMR.shape[3], curMR.shape[4]

        if curMR.shape[2] != 512:
            curMR = F.interpolate(curMR, size=(512, 512, curMR.shape[4]), mode='trilinear', align_corners=False)
            curGT = F.interpolate(curGT.float(), size=(512, 512, curMR.shape[4]), mode='nearest', align_corners=None)
            curGT = curGT.squeeze(1) # # BHWD
            firstGT = F.interpolate(firstGT.float(), size=(512, 512, curMR.shape[4]), mode='nearest', align_corners=None)

        ## extract bbox from curGT
        curGT_np = torch.permute(curGT, dims=(0,3,1,2)).cpu().numpy()[0,...] # DHW
        D, H, W = curGT_np.shape
        segs_3D = np.zeros((D, H, W), dtype=np.uint8)

        bbox_dict, marker_zids = get_bounding_box(box_func, curGT_np, label_id=1, bbox_shift=args.bbox_shift)
        # boxes = extract_boxes(bbox_dict)
        # groups = group_by_centroid_proximity(boxes)
        # summary = summarize(groups)
        # print(f"summary: {summary}")
        slice_idx_start = 0
        slice_idx_end = D
        middle = 0
        ##  prepare the data for inference
        # print(f"Group {i}: Start: {slice_idx_start}, End: {slice_idx_end}, Middle: {middle}")
        key_slice_idx_offset = middle - slice_idx_start # get the offset for the current slice
        ## remove the slice without bbox
        img_resized = torch.permute(curMR, dims=(0,4,1,2,3))[0,...] ## d1hw
        # img_resized = img_resized[slice_idx_start:slice_idx_end+1, ...] # d'1hw
        img_resized = torch.cat((img_resized,img_resized,img_resized),dim=1) ## d3hw
        img_mean=(0.485, 0.456, 0.406)
        img_std=(0.229, 0.224, 0.225)
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].to(device)
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].to(device)
        img_resized -= img_mean
        img_resized /= img_std
        
        box_ori = bbox_dict[middle] 
        ## check if there are more than one bbox in a mask
        if box_ori.ndim > 1:
            ## inference one by one
            for num_box in range(box_ori.shape[0]):
                bbox = box_ori[num_box,:] #/ np.array([W, H, W, H]) * 1024 ## adjust to 1024x1024 scale
                # segs_3D_temp = infer_video_mode_two_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box)
                segs_3D_temp = infer_video_mode_one_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box)
                segs_3D[slice_idx_start:slice_idx_end+1, :, :] = segs_3D_temp
        else:
            bbox = bbox_dict[middle]  # get the bounding box for the current slice
            # segs_3D[slice_idx_start:slice_idx_end+1, :, :]  = infer_video_mode_two_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box)
            segs_3D[slice_idx_start:slice_idx_end+1, :, :]  = infer_video_mode_one_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box)


        # cal dice, hd95, nsd and asd            
        pd = torch.from_numpy(segs_3D)[None, None, ...] #BCDHW
        gt = torch.from_numpy(curGT_np)[None, None, ...] 
        # resize to original size
        pd = F.interpolate(pd, size=(ori_D, ori_H, ori_W), mode='nearest', align_corners=None)
        gt = F.interpolate(gt.float(), size=(ori_D, ori_H, ori_W), mode='nearest', align_corners=None)


        dice_case = dice_fun(pd, gt).numpy()[0,0]
        hd95_case = hd95_fun(pd, gt).numpy()[0,0]
        nsd_case = nsd_fun(pd, gt).numpy()[0,0]
        asd_case = asd_fun(pd, gt).numpy()[0,0]
        print(f"case {caseID} dice: {dice_case}")

        ## save each case results to txt
        with open(os.path.join(save_eval_path, "test_each_dice_hd95_nsd_asd_fix"+str(NUM)+".txt"), "a") as f:
            write_content = f"{caseID} dice:\t{dice_case:.4f}\thd95:\t{hd95_case:.4f}\tnsd:\t{nsd_case:.4f}\tasd:\t{asd_case:.4f}\t\n"
            f.write(write_content)

        dice_all.append(dice_case)
        hd95_all.append(hd95_case)
        nsd_all.append(nsd_case)
        asd_all.append(asd_case)
        
        # ## save results
        # curMR_np = curMR_np[..., 0] #DHW3-->DHW .transpose((1,2,0,3)
        # if args.visualize:
        #     for label_id in label_ids:
        #         seg_sitk = sitk.GetImageFromArray(segs_dict[label_id])
        #         seg_sitk.SetSpacing((0.89, 0.89, 3))
        #         sitk.WriteImage(seg_sitk, join(pred_save_dir, caseID+"_pd_sam2.nii.gz"))

        #     img_sitk = sitk.GetImageFromArray(curMR_np)
        #     img_sitk.SetSpacing((0.89, 0.89, 3))
        #     sitk.WriteImage(img_sitk, join(pred_save_dir, caseID+"_im_sam2.nii.gz"))
        #     gts_sitk = sitk.GetImageFromArray(curGT_np)
        #     gts_sitk.SetSpacing((0.89, 0.89, 3))
        #     sitk.WriteImage(gts_sitk, join(pred_save_dir, caseID+"_gt_sam2.nii.gz"))


    print(f"mean dice: {np.array(dice_all).mean()}")
    ## save results
    with open(os.path.join(save_eval_path, "test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "a") as f:
        write_content = f"{'#'*60}\nmean dice\tmean hd95\tmean nsd\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
        f.write(write_content)

    ## save format results
    with open(os.path.join(save_eval_path, "format_test_mean_dice_hd_nsd_asd"+str(NUM)+".txt"), "a") as f:
        write_content = f"{'#'*60}\nmean dice\tmean nsd\tmean hd95\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.4f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.2f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.2f}±{np.array(nsd_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.2f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(asd_all).mean():.2f}±{np.array(asd_all).std():.2f}\t\n"
        write_content = write_content + f"{np.array(dice_all).mean()*100:.2f}\t" \
                                        f"{np.array(nsd_all).mean():.2f}\t" \
                                        f"{np.array(hd95_all).mean()*100:.2f}\t" \
                                        f"{np.array(asd_all).mean():.2f}\n"
        f.write(write_content)

