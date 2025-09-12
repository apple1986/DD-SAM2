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
import math

from sam2.sam2_video_trainer import SAM2VideoTrainer
from sam2.build_sam import build_sam2_video_predictor_npz, build_sam2_video_predictor_npz_apt
from sam2.utils.transforms import SAM2Transforms
from sam2.modeling.sam.transformer import TwoWayAttentionBlock
from sam2.adapter_ap import add_adapters_to_imgenc

from torch import multiprocessing as mp
from torchvision import transforms
from torch.amp import autocast, GradScaler

from dataset.trackrad2025 import ToTensor, CenterCrop
from builders.trackrad_builder import load_data
from gp_knife.get_prompts import get_all_bounding_boxes_with_ccl_box_fix, get_all_bounding_boxes_with_ccl_box_rand, get_all_bounding_boxes_with_ccl
from gp_knife.freq_func import set_random_seed
from gp_knife.eval_model import get_dice_hd_nsd_asd
from gp_utils.process_mask import get_bounding_box, generate_input_bbox
from gp_utils.medsam_video_mode import infer_video_mode_one_direction

import monai
from torch.optim.lr_scheduler import StepLR


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

def eval_sam_model(valLoader, save_eval_path, checkpoint, epoch_num=1, num_blocks=6, data_type="val", apt_flag="mlp", insert_pos="imgenc"):
    box_func = get_all_bounding_boxes_with_ccl_box_rand # function to get the bounding box
    dice_all = [] # save all data
    hd95_all = []
    nsd_all = []
    asd_all = []
    cnt_case = 0
    predictor = build_sam2_video_predictor_npz_apt(model_cfg, checkpoint, use_ft=True, device=device, 
                                                   apt_flag=apt_flag, insert_pos=insert_pos, num_blocks=num_blocks)
    
    for data_four in valLoader:
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
                segs_3D_temp = infer_video_mode_one_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box)
                segs_3D[slice_idx_start:slice_idx_end+1, :, :] = segs_3D_temp
        else:
            bbox = bbox_dict[middle]  # get the bounding box for the current slice
            segs_3D[slice_idx_start:slice_idx_end+1, :, :]  = infer_video_mode_one_direction(img_resized, predictor, bbox, H, W, key_slice_idx_offset, propagate_with_box)


        # cal dice, hd95, nsd and asd            
        pd = torch.from_numpy(segs_3D)[None, None, ...] #BCDHW
        gt = torch.from_numpy(curGT_np)[None, None, ...] 
        # resize to original size
        pd = F.interpolate(pd, size=(ori_D, ori_H, ori_W), mode='nearest', align_corners=None)
        gt = F.interpolate(gt.float(), size=(ori_D, ori_H, ori_W), mode='nearest', align_corners=None)

        ## remove the first slice since it has give the gt
        pd = pd[:,:,1:, :,:]
        gt = gt[:,:,1:, :,:]

        dice_case = dice_fun(pd, gt).numpy()[0,0]
        hd95_case = hd95_fun(pd, gt).numpy()[0,0]
        nsd_case = nsd_fun(pd, gt).numpy()[0,0]
        asd_case = asd_fun(pd, gt).numpy()[0,0]
        print(f"case {caseID} dice: {dice_case}")

        ## save each case results to txt
        with open(os.path.join(save_eval_path, f"{data_type}_each_dice_hd95_nsd_asd0.txt"), "a") as f:
            write_content = f"{caseID} dice:\t{dice_case:.4f}\thd95:\t{hd95_case:.4f}\tnsd:\t{nsd_case:.4f}\tasd:\t{asd_case:.4f}\t\n"
            f.write(write_content)

        dice_all.append(dice_case)
        hd95_all.append(hd95_case)
        nsd_all.append(nsd_case)
        asd_all.append(asd_case)

    mean_dice = np.array(dice_all).mean()
    print(f"mean dice: {mean_dice}")
        
    ## save mean results
    with open(os.path.join(save_eval_path, f"{data_type}_each_dice_hd95_nsd_asd0.txt"), "a") as f:
        write_content = f"{'#'*60}\nEpoch: {epoch_num}\tmean dice\tmean hd95\tmean nsd\tmean asd\t\n"
        write_content = write_content + f"{np.array(dice_all).mean():.4f}±{np.array(dice_all).std():.2f}\t" \
                                        f"{np.array(hd95_all).mean():.4f}±{np.array(hd95_all).std():.2f}\t" \
                                        f"{np.array(nsd_all).mean():.4f}±{np.array(nsd_all).std():.4f}\t" \
                                        f"{np.array(asd_all).mean():.4f}±{np.array(asd_all).std():.4f}\t\n"
        f.write(write_content)

    ## save format results
    with open(os.path.join(save_eval_path, f"format_{data_type}_mean_dice_hd_nsd_asd0.txt"), "a") as f:
        write_content = f"{'#'*60}\nEpoch: {epoch_num}\tmean dice\tmean nsd\tmean hd95\tmean asd\t\n"
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

    return mean_dice

## build medsam bodel
def build_model(args):
    # build model
    model = build_sam2_video_predictor_npz(
        args.model_cfg, args.sam2_checkpoint, mode="train"
    )
    model.to(args.device)
    model.eval()
    return model


## adjust the learning rate of the adapters
def adjust_learning_rate_apt_img_msk(medsam2, lr_init=1e-6):
    # Step 1: Freeze all parameters
    for param in medsam2.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze Mask Decoder
    for param in medsam2.model.sam_mask_decoder.parameters():
        param.requires_grad = True

    # Step 3: Unfreeze LoRA parameters only
    for name, param in medsam2.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True

    # Collect parameters
    adapter_params = []
    decoder_params = []
    other_params = []

    for name, param in medsam2.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen ones
        elif 'sam_mask_decoder' in name:
            decoder_params.append(param)
        elif 'lora_' in name:
            adapter_params.append(param)
        else:
            other_params.append(param)  # Optional

    # Define optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {"params": decoder_params, "lr": lr_init},
        {"params": adapter_params, "lr": lr_init * 10},
        # {"params": other_params, "lr": lr_init},  # Optional
    ], weight_decay=1e-4)

    return optimizer


# ###lora
# class LoRALinear(nn.Module):
#     def __init__(self, original_linear, rank=4, alpha=1.0):
#         super().__init__()
#         self.original = original_linear
#         self.rank = rank
#         self.alpha = alpha

#         # Freeze the original weights
#         for param in self.original.parameters():
#             param.requires_grad = False

#         in_features = original_linear.in_features
#         out_features = original_linear.out_features

#         # LoRA adapters
#         self.lora_down = nn.Linear(in_features, rank, bias=False)
#         self.lora_up = nn.Linear(rank, out_features, bias=False)

#         # Scale
#         self.scale = alpha / rank

#         # Initialize LoRA layers
#         nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_up.weight)

#     def forward(self, x):
#         return self.original(x) + self.scale * self.lora_up(self.lora_down(x))

# def apply_lora_to_attention(model, rank=4, alpha=1.0):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear) and ('qkv' in name or 'proj' in name):
#             parent = get_parent_module(model, name)
#             attr_name = name.split('.')[-1]
#             orig_linear = getattr(parent, attr_name)
#             lora_linear = LoRALinear(orig_linear, rank=rank, alpha=alpha)
#             setattr(parent, attr_name, lora_linear)

# def get_parent_module(model, name):
#     """Helper to get parent module given module name."""
#     names = name.split('.')
#     for n in names[:-1]:
#         model = getattr(model, n)
#     return model

#######################################
## prepare hyperparameters
image_size = 512
args.visualize = False
data_root = args.data_root
args.bbox_shift = 0
device = "cuda:1"  # args.device
tv_types = ['train',] # 'val','test'
scaler = GradScaler(device=device)  # specify device

# set the bounding box function
box_func = get_all_bounding_boxes_with_ccl_box_rand
model_cfg = args.model_cfg
# checkpoint = args.sam2_checkpoint
num_workers = args.num_workers
propagate_with_box = args.propagate_with_box
lr = 1e-5
max_epoch = 30

apt_flags = ["lora",]
root_path = "/home/gxu/proj1/lesionSeg/utswtumor_track"
num_adp = 12
for apt_flag in apt_flags:
    print(f"lr:{lr}")      
    save_type = "ft_img_adp_"+apt_flag+"_1e5_"+str(num_adp)+"ft_lora"
    insert_pos = "imgenc"
    #############################
    # initialized medsam2
    medsam2 = SAM2VideoTrainer(model_cfg, args.sam2_checkpoint, device = device)
    ## inject adapters
    # add_adapters_to_imgenc(medsam2, adapter_dim=64, num_blocks=num_adp, apt_flag=apt_flag)
    # image_encoder = medsam2.model.image_encoder.trunk
    # apply_lora_to_attention(image_encoder, rank=4, alpha=16)
    model = add_adapters_to_imgenc(medsam2, adapter_dim=64, num_blocks=num_adp, apt_flag=apt_flag)
    print(medsam2.model)
    medsam2 = medsam2.to(device)

    optimizer = adjust_learning_rate_apt_img_msk(medsam2, lr_init=lr)
    ## adjust learning rate
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5) 
    save_eval_path_root = os.path.join(args.pred_save_dir, "medsam2_ft") # save all medsam2_ft path
    os.makedirs(save_eval_path_root, exist_ok=True)
    save_eval_path = os.path.join(save_eval_path_root, save_type) # save model path
    os.makedirs(save_eval_path, exist_ok=True)
    # load evaluation function
    dice_fun, hd95_fun, nsd_fun, asd_fun = get_dice_hd_nsd_asd()
    ## set loss functions
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    args.bbox_shift = 0 # bbox shift

    ## copy this file to the save path
    script_path = join(save_eval_path, os.path.basename(__file__))
    shutil.copyfile(__file__, script_path)

    cnt_case = 0
    ## load data
    all_train_transform = transforms.Compose([
            # CenterCrop(patch_size),
            ToTensor(),
    ])
    all_val_transform = transforms.Compose([
            ToTensor(),
    ])
    all_test_transform = transforms.Compose([
            ToTensor(),
    ])
    trainLoader = load_data(root_path, all_train_transform, batch_size=1, split="train")
    valLoader = load_data(root_path, all_val_transform, batch_size=1, split="val")
    testLoader = load_data(root_path, all_test_transform, batch_size=1, split="test")
    ## training
    print(f"cuda: {device}")
    pid = os.getpid()
    ## record the basic information
    with open(os.path.join(save_eval_path, f"train_info.txt"), "w") as f:
        write_content = f"lr: {lr}\tcuda_num: {device}\t max_epoch: {max_epoch}\n"
        f.write(f"Python PID: {pid}\n")
        f.write(write_content)
        write_content = f"save_path: {save_eval_path}\n"
        f.write(write_content)
        write_content = f"train data: {len(trainLoader)}\n"
        f.write(write_content)
        write_content = f"val data: {len(valLoader)}\n"
        f.write(write_content)
        write_content = f"test data: {len(testLoader)}\n"
        f.write(write_content)

    ## copy the current testing script
    script_path = join(save_eval_path, os.path.basename(__file__))
    # if not os.path.exists(script_path):
    shutil.copyfile(__file__, script_path)


    for epoch in range(0, max_epoch):
        print(f"Epoch {epoch+1}/{max_epoch}")
        epoch_loss = 0
        cnt_case = 0
        for data_four in trainLoader:
            cnt_case = cnt_case + 1
            print(f"cnt case {cnt_case}")
            curMR, curGT, firstGT = data_four["curMR"], data_four["curGT"], data_four["firstGT"]
            caseID = data_four["caseID"][0]
            curMR = curMR.to(device) # BCHWD
            curGT = curGT.to(device) # BCHWD
            firstGT = firstGT.to(device) # B1HWD
            ori_H, ori_W, ori_D = curMR.shape[2], curMR.shape[3], curMR.shape[4]
            if curMR.shape[2] != image_size:
                curMR = F.interpolate(curMR, size=(image_size, image_size, curMR.shape[4]), mode='trilinear', align_corners=False)
                curGT = F.interpolate(curGT.float(), size=(image_size, image_size, curMR.shape[4]), mode='nearest', align_corners=None)
                firstGT = F.interpolate(firstGT.float(), size=(image_size, image_size, curMR.shape[4]), mode='nearest', align_corners=None)

            ## extract bbox from curGT
            curGT_np = torch.permute(curGT.squeeze(1), dims=(0,3,1,2)).cpu().numpy()[0,...] # DHW
            D, H, W = curGT_np.shape
            segs_3D = np.zeros((D, H, W), dtype=np.uint8)
            bbox_dict, marker_zids = get_bounding_box(box_func, curGT_np, label_id=1, bbox_shift=args.bbox_shift)
            ## data normalization
            img_resized = torch.permute(curMR, dims=(0,4,1,2,3))[0,...] ## d1hw
            img_resized = torch.cat((img_resized,img_resized,img_resized),dim=1) ## d3hw
            img_mean=(0.485, 0.456, 0.406)
            img_std=(0.229, 0.224, 0.225)
            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].to(device)
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].to(device)
            img_resized -= img_mean
            img_resized /= img_std

            ## get the training batch and bbox
            ## for each vol, random sample 64 times each time generate 2 batch, each batch has 8 slices
            dice_case_all = 0
            for n in range(0, 64):
                input_chuck, gt_chuck, first_sel_bboxes = generate_input_bbox(batch_size=2, image_depth=D, mr_vol=img_resized, 
                                                                            gt_vol=curGT, bbox_dict=bbox_dict, device=device, image_size=image_size)
                ## training
                optimizer.zero_grad()
                with autocast(device_type=device):
                    all_masks, all_logits, all_ious = medsam2(input_chuck, first_sel_bboxes) # input_chuck: BTCHW: ([2, 8, 3, 512, 512])
                    all_prob_pred = torch.cat(all_masks, dim=1).unsqueeze(2) # BTCHW
                    ## calculate loss
                    loss = seg_loss(all_prob_pred, gt_chuck) + ce_loss(all_prob_pred, gt_chuck.float())
                ## update            
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                # cal dice, hd95, nsd and asd            
                pd = all_prob_pred.detach().cpu().view(all_prob_pred.shape[0], -1, H, W) # B*T*C*H*W--> B*TC*H*W
                gt = gt_chuck.detach().cpu().view(all_prob_pred.shape[0], -1, H, W)
                # to the original size
                pd = (pd > 0.5).float()
                pd = F.interpolate(pd, size=(ori_H, ori_W), mode='nearest', align_corners=None)
                gt = F.interpolate(gt, size=(ori_H, ori_W), mode='nearest', align_corners=None)
                dice_case = dice_fun(pd, gt).numpy()[0,0]
                print(f" slice dice: {dice_case}")
                dice_case_all = dice_case_all + dice_case

            # Write loss to txt file
            with open(os.path.join(save_eval_path, f"training_case_loss_dice_log.txt"), "a") as f:
                f.write(f"loss: {loss.item()/64.0:.6f}\tdice: {dice_case_all/64.0:.4f}\n")
            
        ## save each loss of a epoch
        epoch_loss = epoch_loss / (64*len(trainLoader))
        # Get current learning rate before scheduler changes it
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print(f"LR before scheduler: {current_lr}")
        # Write loss and learning rate to txt file
        with open(os.path.join(save_eval_path, f"training_epoch_loss_lr_log.txt"), "a") as f:
            f.write(f"Epoch {epoch}:\t{epoch_loss:.6f}\tLR (adapter):\t{optimizer.param_groups[1]['lr']:.4e}\n")
        ## adjust learning after 15 epochs   
        scheduler.step()

        ## save model
        save_eval_path = os.path.join(save_eval_path_root, save_type)
        checkpoint = os.path.join(save_eval_path, f"medsam2_last.pth")
        torch.save(medsam2.state_dict(), checkpoint)
        ## do evaluation
        dice_res = eval_sam_model(valLoader, save_eval_path, checkpoint, epoch_num=epoch+1, num_blocks=num_adp, 
                                  data_type="val", apt_flag=apt_flag, insert_pos=insert_pos)

        ## save the best model
        if epoch == 0:
            best_dice = dice_res
            best_epoch = epoch + 1
            best_checkpoint = checkpoint
            ## save the best model
            shutil.copyfile(best_checkpoint, os.path.join(save_eval_path, f"medsam2_best.pth"))
            print(f"save best model at epoch {best_epoch}")
        else:
            if dice_res >= best_dice:
                best_dice = dice_res
                best_epoch = epoch + 1
                best_checkpoint = checkpoint
                ## save the best model
                shutil.copyfile(best_checkpoint, os.path.join(save_eval_path, f"medsam2_best.pth"))
                print(f"save best model at epoch {best_epoch}")
        
    ## for the test set
    testLoader = load_data(root_path, all_val_transform, batch_size=1, split="test")
    checkpoint = os.path.join(save_eval_path, f"medsam2_best.pth") 
    dice_res = eval_sam_model(testLoader, save_eval_path, checkpoint, epoch_num=best_epoch, num_blocks=num_adp,
                              data_type="test", apt_flag=apt_flag, insert_pos=insert_pos)