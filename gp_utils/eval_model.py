



def eval_sam_model(valLoader, save_eval_path, checkpoint, epoch_num=1, data_type="val", apt_flag="mlp", insert_pos="imgenc"):
    box_func = get_all_bounding_boxes_with_ccl_box_rand # function to get the bounding box
    dice_all = [] # save all data
    hd95_all = []
    nsd_all = []
    asd_all = []
    cnt_case = 0
    predictor = build_sam2_video_predictor_npz_apt(model_cfg, checkpoint, use_ft=True, device=device, apt_flag=apt_flag, insert_pos=insert_pos)
    
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
