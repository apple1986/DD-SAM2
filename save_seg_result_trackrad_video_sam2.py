## visualize the segmentation results of MedSAM2 on TrackRad2025 video dataset
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
from sam2.build_sam import build_sam2_video_predictor_npz_apt
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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    default='/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2.1_hiera_tiny.pt',
    help="SAM2 pretrained model checkpoint",
)
parser.add_argument(
    "-model_cfg",
    type=str,
    default="configs/sam2.1_hiera_t.yaml",
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
    ## XX: bbox should from the first slice

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
######################################
class MedSAM2(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam2_model = model
        # freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        

    def forward(self, image, box, masks=None):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=masks,
            )
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits
    
    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        # bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        bb_feat_sizes = [(128, 128), (64, 64), (32, 32)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features

def plot_feature_maps(tensor, num_slices=9, cmap='viridis'):
    """
    Plots a given number of feature map slices from a tensor of shape [1, C, H, W].

    Parameters:
    - tensor (torch.Tensor): Input tensor with shape [1, C, H, W]
    - num_slices (int): Number of slices (channels) to display
    - cmap (str): Colormap for display (default: 'viridis')
    """
    assert tensor.ndim == 4, "Tensor must be 4D (B, C, H, W)"
    assert tensor.shape[0] == 1, "Batch size must be 1"
    
    tensor = tensor[0]  # Remove batch dimension -> [C, H, W]
    num_channels = tensor.shape[0]
    num_slices = min(num_slices, num_channels)

    cols = int(num_slices ** 0.5)
    rows = (num_slices + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

    for i in range(rows * cols):
        ax = axs[i // cols, i % cols] if rows > 1 else axs[i % cols]
        if i < num_slices:
            ax.imshow(tensor[i].cpu().numpy(), cmap=cmap)
            ax.set_title(f'Channel {i}', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def rotate_to_original_orientation(curVol):
    """
    Rotate the image to match the original orientation.
    The input image is expected to be in sagittal view.
    """
    # convert the axial view to sagittal view
    curVol = np.transpose(curVol, (2, 1, 0)) # DHW -> HWD
    # roate the sagittal view to match the original image orientation
    curVol = np.rot90(curVol, k=1, axes=(0, 1)) # rotate 90 degrees counter-clockwise
    # flip up and bottom of the sagittal view
    curVol = np.flip(curVol, axis=0) # flip the first dimension
    return curVol
#######################################
## prepare hyperparameters
image_size = 1024
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
save_eval_path = os.path.join(root_path, "paper_seg_res")
model_name = "sam2"
model_cfg = args.model_cfg
checkpoint = args.sam2_checkpoint
num_workers = args.num_workers
propagate_with_box = args.propagate_with_box
use_ft = False # use the original model or the fine-tuned model
args.visualize = True # save the segmentation results as .nii.gz files
#############################
# initialized predictor
predictor = build_sam2_video_predictor_npz_apt(model_cfg, checkpoint, use_ft=use_ft, apt_flag="no", insert_pos="ori") #
# save_eval_path_root = os.path.join(args.pred_save_dir, "medsam2")
# os.makedirs(save_eval_path_root, exist_ok=True)
sam2_model = MedSAM2(model=predictor)
# sam2_model.load_state_dict(medsam2_checkpoint, strict=True)
sam2_model.eval()

# load evaluation function
dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()
NUM = 0
## inference
for tv_type in tv_types:
    ## save segmentation results
    save_seg_result_path = os.path.join(save_eval_path, f"{model_name}")
    makedirs(save_seg_result_path, exist_ok=True)
    ## copy the current testing script
    script_path = join(save_eval_path, os.path.basename(__file__))
    if not os.path.exists(script_path):
        shutil.copyfile(
            __file__, script_path
        )
    args.bbox_shift =  NUM # bbox shift
    sam2_transforms = SAM2Transforms(resolution=image_size, mask_threshold=0)

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
        print(f"caseID:{caseID}")
        # if caseID != "C_001": # C_006,  A_013 A_010 B_007 C_001
        #     continue
        curMR = curMR.to(device) # BCHWD
        curGT = curGT.to(device) # BCHWD
        firstGT = firstGT.to(device) # B1HWD

        ori_H, ori_W, ori_D = curMR.shape[2], curMR.shape[3], curMR.shape[4]

        if curMR.shape[2] != image_size:
            curMR = F.interpolate(curMR, size=(image_size, image_size, curMR.shape[4]), mode='trilinear', align_corners=False)
            curGT = F.interpolate(curGT.float(), size=(image_size, image_size, curMR.shape[4]), mode='nearest', align_corners=None)
            curGT = curGT.squeeze(1) # # BHWD
            firstGT = F.interpolate(firstGT.float(), size=(image_size, image_size, curMR.shape[4]), mode='nearest', align_corners=None)

        ## extract bbox from curGT
        curGT_np = torch.permute(curGT, dims=(0,3,1,2)).cpu().numpy()[0,...] # DHW
        D, H, W = curGT_np.shape
        segs_3D = np.zeros((D, H, W), dtype=np.uint8)

        bbox_dict, marker_zids = get_bounding_box(box_func, curGT_np, label_id=1, bbox_shift=args.bbox_shift)
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
        
      
        ## inference
        box_ori = bbox_dict[middle] 
        ## check if there are more than one bbox in a mask
        if box_ori.ndim > 1:
            ## inference one by one
            for num_box in range(box_ori.shape[0]):
                bbox = box_ori[num_box,:] #/ np.array([W, H, W, H]) * 1024 ## adjust to 1024x1024 scale
                segs_3D_temp = infer_video_mode_one_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box)
                segs_3D[slice_idx_start:slice_idx_end+1, :, :] = segs_3D_temp
        else:
            bbox = bbox_dict[middle]  # get the bounding box for the current slice
            segs_3D[slice_idx_start:slice_idx_end+1, :, :]  = infer_video_mode_one_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box)


        ## visualization

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
        dice_all.append(dice_case)

        ## save the segmentation results, ground truth and the original image
        if args.visualize:
            # save the original image
            curMR = F.interpolate(curMR, size=(ori_H, ori_W, curMR.shape[4]), mode='trilinear', align_corners=False)
            curMR_cuda = torch.permute(curMR, dims=(0,1,4,2,3)) # BCHWD -> BCDHW

            curMR = curMR_cuda.cpu().numpy()[0,0,...] # DHW
            # # convert the axial view to sagittal view
            # curMR = np.transpose(curMR, (2, 1, 0)) # DHW -> HWD
            # # roate the sagittal view to match the original image orientation
            # curMR = np.rot90(curMR, k=1, axes=(0, 1)) # rotate 90 degrees counter-clockwise
            # # flip up and bottom of the sagittal view
            # curMR = np.flip(curMR, axis=0) # flip the first dimension
            curMR = rotate_to_original_orientation(curMR) # rotate to original orientation
            curMR_np = sitk.GetImageFromArray(curMR)
            curMR_np.SetSpacing((1.0, 1.0, 1.0))
            curMR_np_path = join(save_seg_result_path, f"{caseID}_mr.nii.gz")
            sitk.WriteImage(curMR_np, curMR_np_path)
            print(f"Save original image to {curMR_np_path}")

            # save the segmentation results
            segs_3D = pd.numpy()[0,0]
            segs_3D = (segs_3D > 0.5).astype(np.uint8) # DHW
            ## write a function to roate the segmentation results to match the original image orientation
            # flip up and bottom
            segs_3D = rotate_to_original_orientation(segs_3D) # rotate to original orientation
            segs_3D = sitk.GetImageFromArray(segs_3D)
            segs_3D.SetSpacing((1.0, 1.0, 1.0))
            segs_3D_path = join(save_seg_result_path, f"{caseID}_pd.nii.gz")
            sitk.WriteImage(segs_3D, segs_3D_path)
            print(f"Save segmentation results to {segs_3D_path}")
            # save the ground truth
            curGT_np = gt.numpy()[0,0]
            curGT_np = rotate_to_original_orientation(curGT_np) # rotate to original orientation
            curGT_np = sitk.GetImageFromArray(curGT_np)
            curGT_np.SetSpacing((1.0, 1.0, 1.0))
            curGT_np_path = join(save_seg_result_path, f"{caseID}_gt.nii.gz")
            sitk.WriteImage(curGT_np, curGT_np_path)
            print(f"Save ground truth to {curGT_np_path}")

    ## print mean and std of dice
    dice_all = np.array(dice_all)
    print(f"Mean dice: {np.mean(dice_all)}, Std dice: {np.std(dice_all)}")


