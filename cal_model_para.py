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
import random

from sam2.sam2_video_trainer import SAM2VideoTrainer, SAM2VideoTrainer_1024
from sam2.build_sam import build_sam2_video_predictor_npz, build_sam2_video_predictor_npz_apt
from sam2.utils.transforms import SAM2Transforms
from sam2.modeling.sam.transformer import TwoWayAttentionBlock
from sam2.adapter_ap import add_adapters_to_imgenc

from torch import multiprocessing as mp
from torchvision import transforms
from torch.amp import autocast, GradScaler

from dataset.echocard import ToTensor, CenterCrop
from builders.echocard_builder import load_data
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_fix, get_all_bounding_boxes_with_ccl_box_rand, get_all_bounding_boxes_with_ccl
from gp_knife.freq_func import set_random_seed
from gp_knife.eval_model import get_dice_hd_nsd_asd
from gp_utils.process_mask import get_bounding_box, generate_input_bbox_for_empty_bbox
from gp_utils.medsam_video_mode import infer_video_mode_one_direction

import monai
from torch.optim.lr_scheduler import StepLR
from ptflops import get_model_complexity_info
from torchinfo import summary

## set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)


###########################
## set parameters
parser = argparse.ArgumentParser(
    description="Run training on training/validation set with MedSAM2"
)
## set paths
parser.add_argument(
    "-data_root",
    type=str,
    default="/home/gxu/proj1/lesionSeg/utswtumor_track",
    help="Path to the data folder",
)
parser.add_argument(
    "-pred_save_dir",
    type=str,
    default="/home/gxu/proj1/lesionSeg/utswtumor_track/MedSAM2_new/checkpoints_echocard",
    help="Path to save the segmentation results",
)
parser.add_argument(
    "-sam2_checkpoint",
    type=str,
    default='/home/gxu/proj1/lesionSeg/utswtumor_track/MedSAM2_new/checkpoints_echocard/medsam2/MedSAM2_latest.pt',
    help="SAM2 pretrained model checkpoint",
)
parser.add_argument(
    "-model_cfg",
    type=str,
    default="configs/sam2.1_hiera_t512.yaml",
    help="Model config file"
)
## set hyper parameters
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

#######################################
## build medsam bodel
def build_model(args):
    # build model
    model = build_sam2_video_predictor_npz(
        args.model_cfg, args.sam2_checkpoint, mode="train"
    )
    model.to(args.device)
    model.eval()
    return model

class WrappedSAM2(nn.Module):
    def __init__(self, model, img_size=1024):
        super().__init__()
        self.model = model
        self.img_size = img_size

    def forward(self, x):
        dummy_bboxes = torch.tensor([[0, 0, self.img_size, self.img_size]], dtype=torch.float).repeat(x.size(0), 1).to(x.device)  # example dummy box
        return self.model(x, dummy_bboxes)

def count_params_and_flops(model, input_shape=(3, 512, 512)):
    # Count trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Use ptflops for FLOPs
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(
            model,
            input_res=input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )

    return n_params, flops

#######################################
## prepare hyperparameters
args.visualize = False
data_root = args.data_root
args.bbox_shift = 0
device = "cuda:1"  # args.device
tv_types = ['train',] # 'val','test'
scaler = GradScaler(device=device)  # specify device

# set the bounding box function
box_func = get_all_bounding_boxes_with_ccl_box_rand

num_workers = args.num_workers
propagate_with_box = args.propagate_with_box
lr = 1e-5
max_epoch = 10
apt_flags = ["ori"]

root_path = "/home/gxu/proj1/lesionSeg/utswtumor_track"
num_adp = 6
apt_flag = "cnn_dw_di_v3_4"
#############################
# initialized medsam2
img_size = 1024
dd_flag = False

if img_size == 512:
    args.sam2_checkpoint = '/home/gxu/proj1/lesionSeg/utswtumor_track/MedSAM2_new/checkpoints_echocard/medsam2/MedSAM2_latest.pt'
    medsam2 = SAM2VideoTrainer(args.model_cfg, args.sam2_checkpoint, device = device)
    if dd_flag is True:
        add_adapters_to_imgenc(medsam2, adapter_dim=64, num_blocks=num_adp, apt_flag=apt_flag)
else:
    args.sam2_checkpoint = '/home/gxu/proj1/lesionSeg/utswlits3d_v2/checkpoint/SAM/sam2.1_hiera_tiny.pt'
    args.model_cfg = "configs/sam2.1_hiera_t.yaml"
    medsam2 = SAM2VideoTrainer_1024(args.model_cfg, args.sam2_checkpoint, device = device)
    if dd_flag is True:
        add_adapters_to_imgenc(medsam2, adapter_dim=64, num_blocks=num_adp, apt_flag=apt_flag)

medsam2 = medsam2.to(device)
wrapped_model = WrappedSAM2(medsam2, img_size=img_size)
# macs, params = get_model_complexity_info(
#     wrapped_model,
#     input_res=(1,  3, img_size, img_size),  # 512x512 input for MedSAM2
#     as_strings=True,
#     print_per_layer_stat=False,
#     verbose=False
# )
# print(f"\nTotal FLOPs: {macs}")
# print(f"Total Parameters: {params}")

# summary(wrapped_model, input_size=(1, 8, 3, img_size, img_size))

params, flops = count_params_and_flops(wrapped_model, input_shape=(1, 3, img_size, img_size))
print(f"Trainable Parameters: {params/1e6:.2f} M")
print(f"FLOPs (512x512 input): {flops/1e9:.2f} GFLOPs")

## use different package to count flops
from torchsummary import summary
summary(wrapped_model, input_size=(3, img_size, img_size))


from ptflops import get_model_complexity_info
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        wrapped_model,
        (1, 3, img_size, img_size),  # input shape (C,H,W)
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )
    print(f"\nTotal FLOPs: {macs}")
    print(f"Total Parameters: {params}")