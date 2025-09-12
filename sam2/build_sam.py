# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def get_best_available_device():
    """
    Get the best available device in the order: CUDA, MPS, CPU
    Returns: device string for torch.device
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def build_sam2(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    use_ft=False,
    **kwargs,
):
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, use_ft=use_ft)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    use_ft=False,
    use_diff_mem_feat=False,
    num_mem_feat=None,
    **kwargs,
):
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)


    if use_diff_mem_feat:
        # Read config and init model
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        cfg["model"]["num_maskmem"] = num_mem_feat
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, ckpt_path, use_ft=use_ft, use_diff_mem_feat=use_diff_mem_feat)
    else:
        # Read config and init model
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, ckpt_path, use_ft=use_ft)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

def build_sam2_video_predictor_mem(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    use_ft=False,
    use_mem_ch_dim_reduction=False,
    **kwargs,
):
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor_mem.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model ap: you can change the hyperparameters here
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, use_ft=use_ft, use_mem_ch_dim_reduction=use_mem_ch_dim_reduction)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor_npz(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    use_ft=False,
    use_diff_mem_feat=False,
    **kwargs,
):
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, use_ft=use_ft, use_diff_mem_feat=use_diff_mem_feat)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model

## aping the build_sam2_video_predictor_npz function to add adapters
from sam2.adapter_ap import add_adapters_to_two_way_attention_blocks, add_adapters_to_imgenc, apply_lora_to_attention
from sam2.mem_ap import add_mem_ca_module
def build_sam2_video_predictor_npz_apt(
    config_file,
    ckpt_path=None,
    device=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    use_ft=False,
    use_mem_ch_dim_reduction=False,
    use_diff_mem_feat = False,
    num_mem_feat = 7,
    apt_flag="mlp",
    insert_pos="mskdec",
    num_blocks=6,
    dilation_rate=[1,3],
    **kwargs,
):
    # Use the provided device or get the best available one
    device = device or get_best_available_device()
    logging.info(f"Using device: {device}")

    if "memory_attention" in insert_pos:
            hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor_npz_mem.SAM2VideoPredictorNPZ",
    ]
    else:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ",
        ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    if use_diff_mem_feat:
        # Read config and init model
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        cfg["model"]["num_maskmem"] = num_mem_feat
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
    else:
        # Read config and init model
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
    if insert_pos == "mskdec":
        add_adapters_to_two_way_attention_blocks(model.sam_mask_decoder.transformer, reduction=16, apt_flag=apt_flag)
    elif (insert_pos == "imgenc") and (apt_flag != "lora"):
        add_adapters_to_imgenc(model, adapter_dim=64, num_blocks=num_blocks, flag_type="val", apt_flag=apt_flag, dilation_rate=dilation_rate)
    elif (insert_pos == "imgenc") and (apt_flag == "lora"):
        add_adapters_to_imgenc(model, adapter_dim=64, num_blocks=num_blocks, flag_type="val", apt_flag=apt_flag)
    elif (insert_pos == "imgenc") and (apt_flag == "mskdec"):
        add_adapters_to_two_way_attention_blocks(model.sam_mask_decoder.transformer, reduction=16, apt_flag=apt_flag)
        add_adapters_to_imgenc(model, adapter_dim=64, num_blocks=num_blocks, flag_type="val", apt_flag=apt_flag)
    elif (insert_pos == "memory_attention"):
        ## mem_ca
        add_mem_ca_module(model, mem_dim=64, flag_type="val", apt_flag=apt_flag)


    else:
        pass

    _load_checkpoint(model, ckpt_path, use_ft=use_ft, use_mem_ch_dim_reduction=use_mem_ch_dim_reduction, use_diff_mem_feat=use_diff_mem_feat)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model



def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path, use_ft=False, use_mem_ch_dim_reduction=False, use_diff_mem_feat=False):
    if ckpt_path is not None:
        # sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        if not use_ft: # use original pre-trained model
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        else:
            ## ap # Remove 'model.' prefix
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            sd = {k.replace("model.", ""): v for k, v in sd.items()}

        ## ap:
        if use_mem_ch_dim_reduction is True or use_diff_mem_feat is True:
            model_state = model.state_dict()
            # Filter only matching keys
            filtered_ckpt = {k: v for k, v in sd.items() if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered_ckpt)
            missing_keys, unexpected_keys = model.load_state_dict(model_state)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(sd)
            if missing_keys:
                logging.error(missing_keys)
                raise RuntimeError()
            if unexpected_keys:
                logging.error(unexpected_keys)
                raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
