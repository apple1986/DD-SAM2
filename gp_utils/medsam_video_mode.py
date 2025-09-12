import numpy as np
import torch




## infer the video mode: the first slice is the key slice
def infer_video_mode(img_resized, medsam2, bbox, H, W, key_slice_idx_offset, propagate_with_box):
    segs_3D = np.zeros((img_resized.shape[0], H, W), dtype=np.uint8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = medsam2.init_state(img_resized, H, W)
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = medsam2.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset,
                                                obj_id=1,
                                                box=bbox,
                                            )
        for out_frame_idx, out_obj_ids, out_mask_logits in medsam2.propagate_in_video(inference_state):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        medsam2.reset_state(inference_state)
        ## do the reverse inference
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = medsam2.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset,
                                                obj_id=1,
                                                box=bbox,
                                            )

        for out_frame_idx, out_obj_ids, out_mask_logits in medsam2.propagate_in_video(inference_state, reverse=True):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        medsam2.reset_state(inference_state)
    return segs_3D


def infer_video_mode_one_direction(img_resized, medsam2, bbox, H, W, key_slice_idx_offset, propagate_with_box):
    ## XX: bbox should from the first slice and the last slice

    segs_3D = np.zeros((img_resized.shape[0], H, W), dtype=np.uint8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = medsam2.init_state(img_resized, H, W)
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = medsam2.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset,
                                                obj_id=1,
                                                box=bbox,
                                            )
        for out_frame_idx, out_obj_ids, out_mask_logits in medsam2.propagate_in_video(inference_state):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        medsam2.reset_state(inference_state)

    return segs_3D


def infer_video_mode_two_direction(img_resized, medsam2, bbox, H, W, key_slice_idx_offset, propagate_with_box):
    segs_3D = np.zeros((img_resized.shape[0], H, W), dtype=np.uint8)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = medsam2.init_state(img_resized, H, W)
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = medsam2.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset,
                                                obj_id=1,
                                                box=bbox,
                                            )
        for out_frame_idx, out_obj_ids, out_mask_logits in medsam2.propagate_in_video(inference_state):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        medsam2.reset_state(inference_state)
        ## do the reverse inference
        if propagate_with_box:
            _, out_obj_ids, out_mask_logits = medsam2.add_new_points_or_box(
                                                inference_state=inference_state,
                                                frame_idx=key_slice_idx_offset+img_resized.shape[0]-1,
                                                obj_id=1,
                                                box=bbox, ##?? wrong: you need to use the last slice
                                            )

        for out_frame_idx, out_obj_ids, out_mask_logits in medsam2.propagate_in_video(inference_state, reverse=True):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        medsam2.reset_state(inference_state)
    return segs_3D
