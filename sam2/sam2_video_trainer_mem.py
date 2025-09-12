import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
# from sam2.modeling.sam2_base import SAM2Base
from sam2.build_sam import build_sam2_video_predictor_mem as build_sam2_video_predictor
from sam2.utils.transforms import SAM2Transforms
from sam2.modeling.sam2_utils import get_1d_sine_pe
from sam2.modeling.cal_similarity_ch import euclidean_similarity, cosine_similarity, dot_product_similarity, pearson_correlation, manhattan_similarity, jaccard_similarity, hamming_similarity, structural_similarity_index
from sam2.modeling.cal_similarity_sp import euclidean_similarity_sp, cosine_similarity_sp, dot_product_similarity_sp, pearson_correlation_sp, manhattan_similarity_sp, ssim_channelwise_sp
from hydra import compose
from sam2.modeling.mem_short_long import memory_compressor


## aping
def prune_memory_features(maskmem_features, maskmem_pos_enc, prune_frame_num=2, prune_sim="cosine"):
    # get the last frame's feature
    last_vision_feature = maskmem_features[-1] # LBC
    repeat_dims = [len(maskmem_features) - 2] + [1] * (len(last_vision_feature.shape))
    last_frame_expanded = last_vision_feature.unsqueeze(0).repeat(*repeat_dims)  # [num_frames-2, 1024, bs, 64]
    # get the candidate frames: exluude the first and last frames
    candidate_frames = torch.stack(maskmem_features[1:-1], dim=0)  # [num_frames-2, 1024, bs, 64] # NUMxLxBxC            
    # similarities = torch.cosine_similarity(candidate_frames, last_frame_expanded, dim=-1)  # [num_frames-2, 1024, bs]
    # similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]

    # channel dimension: compute the cosine similarity between the candidate frames and the last frame
    if prune_sim == "None":
        delete_indices = None
    else:
        delete_indices = []
    if prune_sim== "cosine":
        similarities = torch.cosine_similarity(candidate_frames, last_frame_expanded, dim=-1)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]
    elif prune_sim == "cosine_unsim":
        similarities = 1 - torch.cosine_similarity(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]   
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]
    elif prune_sim== "euclidean":
        similarities = euclidean_similarity(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]
    elif prune_sim== "dot":
        similarities = dot_product_similarity(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]
    elif prune_sim== "pearson":
        similarities = pearson_correlation(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]
    elif prune_sim== "manhattan":
        similarities = manhattan_similarity(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]                                                        
    elif prune_sim== "ssim":
        similarities = structural_similarity_index(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]  

    ## spatial dimension
    elif prune_sim== "cosine_sp":
        similarities = cosine_similarity_sp(candidate_frames, last_frame_expanded, dim=-1)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]
    elif prune_sim== "euclidean_sp":
        similarities = euclidean_similarity_sp(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]
    elif prune_sim== "dot_sp":
        similarities = dot_product_similarity_sp(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]
    elif prune_sim== "pearson_sp":
        similarities = pearson_correlation_sp(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]
    elif prune_sim== "manhattan_sp":
        similarities = manhattan_similarity_sp(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]                                                          
    elif prune_sim== "ssim_sp":
        similarities = ssim_channelwise_sp(candidate_frames, last_frame_expanded)  # [num_frames-2, 1024, bs]
        similarities = torch.sum(similarities, dim=1)  # [num_frames-2, bs]     
    # else:  
    #     delete_indices = None

    if delete_indices is not None:
        # [num_frames-2, bs] -> [num_frames-2], to support batch in training, we use mean instead of squeeze
        similarities = similarities.mean(dim=1)
        # the ranking is from large to small
        _, sorted_indices = torch.sort(similarities, descending=True)
        # we delete the frames with the largest cosine similarity, which are the most similar frames
        delete_indices = sorted_indices[:prune_frame_num].cpu().numpy() + 1  # +1 because we exclude the first frame, delete two frames
        # delete 'self.num_frame_to_prune' frames from the end to the beginning, or it might cause index error
        delete_indices = sorted(delete_indices, reverse=True)

        # delete the frames
        for i in delete_indices:
            maskmem_features.pop(i)
            maskmem_pos_enc.pop(i)
    
    return maskmem_features, maskmem_pos_enc


# ap: sort seperately for first and second half of memory features
def prune_memory_features_time_split(maskmem_features, maskmem_pos_enc, total_prune_frame_num=2, prune_sim="cosine"):
    if prune_sim == "None":
        return maskmem_features, maskmem_pos_enc
     
    if len(maskmem_features) <= 3 or total_prune_frame_num == 0:
        return maskmem_features, maskmem_pos_enc
    


    middle_idx = len(maskmem_features) // 2
    prune_frame_num = total_prune_frame_num // 2  # Split pruning evenly between two halves
    # References: first and last memory frames
    first_feature = maskmem_features[0]    # [L, B, C]
    last_feature = maskmem_features[-1]

    # Split memory into two halves (excluding first and last)
    first_half = maskmem_features[1:middle_idx]        # [N1]
    second_half = maskmem_features[middle_idx:-1]      # [N2]
    pos_enc_first_half = maskmem_pos_enc[1:middle_idx]
    pos_enc_second_half = maskmem_pos_enc[middle_idx:-1]

    def compute_similarity(candidates, reference, mode):
        """
        Compute similarity between candidate features and reference features.

        Args:
            candidates (List[Tensor]): List of [L, B, C] tensors.
            reference (Tensor): A [L, B, C] tensor.
            mode (str): Similarity metric to use.

        Returns:
            similarities (Tensor): Tensor of shape [N], similarity score for each candidate.
        """
        candidates_stacked = torch.stack(candidates, dim=0)  # [N, L, B, C]
        reference_expanded = reference.unsqueeze(0).expand(len(candidates), -1, -1, -1)  # [N, L, B, C]

        if mode == "cosine":
            sim = torch.cosine_similarity(candidates_stacked, reference_expanded, dim=-1)
        elif mode == "cosine_unsim":
            sim = 1 - torch.cosine_similarity(candidates_stacked, reference_expanded, dim=-1)
        elif mode == "euclidean":
            sim = euclidean_similarity(candidates_stacked, reference_expanded)  # [N, L, B]
        elif mode == "dot":
            sim = dot_product_similarity(candidates_stacked, reference_expanded)
        elif mode == "pearson":
            sim = pearson_correlation(candidates_stacked, reference_expanded)
        elif mode == "manhattan":
            sim = manhattan_similarity(candidates_stacked, reference_expanded)
        elif mode == "ssim":
            sim = structural_similarity_index(candidates_stacked, reference_expanded)
        elif mode == "cosine_sp":
            sim = cosine_similarity_sp(candidates_stacked, reference_expanded, dim=-1)
        elif mode == "euclidean_sp":
            sim = euclidean_similarity_sp(candidates_stacked, reference_expanded)
        elif mode == "dot_sp":
            sim = dot_product_similarity_sp(candidates_stacked, reference_expanded)
        elif mode == "pearson_sp":
            sim = pearson_correlation_sp(candidates_stacked, reference_expanded)
        elif mode == "manhattan_sp":
            sim = manhattan_similarity_sp(candidates_stacked, reference_expanded)
        elif mode == "ssim_sp":
            sim = ssim_channelwise_sp(candidates_stacked, reference_expanded)
        else:
            raise NotImplementedError(f"Similarity mode '{mode}' is not supported.")

        # Sum over sequence length (L), then mean over batch (B)
        return torch.sum(sim, dim=1).mean(dim=1)  # → [N]

    def prune_segment(segment, pos_segment, reference, base_offset):
        if len(segment) <= prune_frame_num:
            return [], []

        sim_scores = compute_similarity(segment, reference, prune_sim)
        _, sorted_indices = torch.sort(sim_scores, descending=True)
        delete_indices = sorted_indices[:prune_frame_num]

        # Convert to original index in maskmem_features
        global_indices = [base_offset + 1 + i.item() for i in delete_indices]  # +1 to skip first frame
        return global_indices

    delete_indices_first = prune_segment(first_half, pos_enc_first_half, first_feature, base_offset=0)
    delete_indices_second = prune_segment(second_half, pos_enc_second_half, last_feature, base_offset=middle_idx)

    # Merge and sort delete indices in reverse to avoid shifting issues
    delete_raw_indices = sorted(delete_indices_first + delete_indices_second, reverse=True)

    for idx in delete_raw_indices:
        maskmem_features.pop(idx)
        maskmem_pos_enc.pop(idx)

    return maskmem_features, maskmem_pos_enc


class SAM2VideoTrainer(nn.Module):
    """
    SAM2VideoTrainer is a PyTorch module for training a video segmentation model using SAM2.
    Attributes:
        device (torch.device): The device to run the model on.
        model (nn.Module): The SAM2 video predictor model.
        num_feature_levels (int): Number of feature levels in the model.
        memory_size (int): Size of the memory for storing features.
        _transforms (SAM2Transforms): Transformations applied to the input data.
        _bb_feat_sizes (list): Spatial dimensions for backbone feature maps.
        num_maskmem (int): Number of mask memory features.
        sam_point_coords (torch.Tensor): Placeholder for SAM point coordinates.
        sam_point_labels (torch.Tensor): Placeholder for SAM point labels.
        _orig_hw (list): Original height and width of the input frames.
        maskmem_features (list): List of mask memory features.
        maskmem_pos_enc (list): List of mask memory positional encodings.
        batch_size (int): Batch size of the input data.
        obj_ptrs (list): List of object pointers.
    """

    def __init__(self, model_cfg, sam2_checkpoint, device, memory_size=7, mask_threshold=0.5, 
                 use_mask_threshold=False, use_mem_ch_dim_reduction=False):
        """
        Initializes the SAM2VideoTrainer class.

        Args:
            model_cfg (dict): Configuration dictionary for the model.
            sam2_checkpoint (str): Path to the SAM2 checkpoint file.
            device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
            memory_size (int, optional): Size of the memory. Defaults to 7.
            mask_threshold (float, optional): Threshold for mask prediction. Defaults to 0.5.
            use_mask_threshold (bool, optional): Flag to use mask thresholding. Defaults to False.

        Attributes:
            device (torch.device): The device to run the model on.
            model (SAM2VideoPredictor): The SAM2 video predictor model.
            num_feature_levels (int): Number of feature levels in the model.
            memory_size (int): Size of the memory.
            _transforms (SAM2Transforms): Transformations applied to the input data.
            _bb_feat_sizes (list): Spatial dimensions for backbone feature maps.
            num_maskmem (int): Number of mask memories.
            sam_point_coords (torch.Tensor): Tensor for SAM point coordinates.
            sam_point_labels (torch.Tensor): Tensor for SAM point labels.
            mask_threshold (float): Threshold for mask prediction.
            use_mask_threshold (bool): Flag to use mask thresholding.
        """
        super().__init__()
        self.device = device
        self.model = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=self.device, mode="train", use_mem_ch_dim_reduction=use_mem_ch_dim_reduction
        )
        self.model.train()
        self.num_feature_levels = self.model.num_feature_levels

        self.num_maskmem = 7 # number of mask memory features:aping: save 7 frames as memory
        self.memory_size = (
            memory_size if memory_size <= self.num_maskmem else self.num_maskmem
        )

        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=0.5,
            max_hole_area=0,
            max_sprinkle_area=0,
        )

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            # (256, 256), # aping
            (128, 128),
            (64, 64),
            (32, 32),
        ]

        self.sam_point_coords = torch.zeros(1, 1, 2, device=device)
        self.sam_point_labels = -torch.ones(1, 1, dtype=torch.int32, device=device)
        self.mask_threshold = mask_threshold
        self.use_mask_threshold = use_mask_threshold

        self.prune_sim = compose(config_name=model_cfg)['model']['sim_method']
        # self.mem_split_ls = compose(config_name=model_cfg)['model']['mem_split_ls']

        self.init_state()

    def init_state(self):
        """
        Initializes the state variables for the video trainer.

        This method sets the initial state of various attributes used in the video
        training process. It resets the original height and width, mask memory
        features, mask memory positional encoding, batch size, and object pointers
        to their default values.

        Attributes:
            _orig_hw (tuple or None): Original height and width of the video frames.
            maskmem_features (Any or None): Features related to mask memory.
            maskmem_pos_enc (Any or None): Positional encoding for mask memory.
            batch_size (int or None): Size of the batch for training.
            obj_ptrs (list): List of object pointers used in the training process.
        """
        self._orig_hw = None
        self.maskmem_features = None
        self.maskmem_pos_enc = None
        self.batch_size = None
        self.current_frame_idx = 0
        self.obj_ptrs = []
        self.num_frames = 0

    def reset_state(self):
        """
        Resets the state of the video trainer.

        This method clears the internal state variables, setting them to their initial values:
        - `_orig_hw`: Set to None. Represents the original height and width.
        - `maskmem_features`: Set to None. Represents the mask memory features.
        - `maskmem_pos_enc`: Set to None. Represents the mask memory positional encoding.
        - `batch_size`: Set to None. Represents the batch size.
        - `obj_ptrs`: Set to an empty list. Represents the object pointers.
        """
        self._orig_hw = None
        self.maskmem_features = None
        self.maskmem_pos_enc = None
        self.batch_size = None
        self.current_frame_idx = 0
        self.obj_ptrs = []
        self.num_frames = 0

    def forward(self, videos, bboxes, labels=None):
        """
        Forward pass for processing video frames and predicting masks, logits, and IoUs.

        Args:
            videos (torch.Tensor): A tensor of shape (batch_size, num_frames, C, H, W) representing the input video frames.
            bboxes (torch.Tensor): A tensor of shape (batch_size, 4) representing the bounding boxes for the first frame.
            labels (torch.Tensor, optional): A tensor of shape (batch_size, num_frames, H, W) representing the ground truth masks for each frame. Defaults to None.

        Returns:
            tuple: A tuple containing:
            - all_masks (list of torch.Tensor): A list of tensors representing the predicted masks for each frame.
            - all_logits (list of torch.Tensor): A list of tensors representing the predicted logits for each frame.
            - all_ious (list of torch.Tensor): A list of tensors representing the predicted IoUs for each frame.
        """
        self.init_state()
        batch_size, num_frames, C, H, W = videos.shape
        self.num_frames = num_frames
        self._orig_hw = [H, W]
        self.batch_size = batch_size

        # Extract features for all frames in the video
        videos = videos.view(batch_size * num_frames, C, H, W)
        features = self.model.forward_image(videos)  # Extract features for all frames
        features = {
            k: (
                v.view(batch_size, num_frames, *v.shape[1:])
                if not isinstance(v, list)
                else ([_v.view(batch_size, num_frames, *_v.shape[1:]) for _v in v])
            )
            for k, v in features.items()
        }
        frame_features = self.preprocess_frame_features(
            features, batch_size, num_frames
        ) # ap: preprocess frame features, e.g. 16 frames

        # Process the first frame with bounding boxes as prompts
        first_frame_features = frame_features[0]
        first_frame_bbox = bboxes.view(batch_size, 4)

        # Predict the first frame masks and IoUs
        first_frame_masks, first_frame_logits, first_frame_ious, object_score_logits = (
            self._predict_first_frame(first_frame_features, first_frame_bbox)
        )

        # Initialize memory with first frame predictions
        prev_pred_mask = first_frame_masks if labels is None else labels[:, 0]
        memory = self._initialize_memory(first_frame_features, prev_pred_mask, object_score_logits) # ap: include feat, and feat_poistion : self.maskmem_features, self.maskmem_pos_enc

        # Process remaining frames
        all_masks, all_logits, all_ious = (
            [first_frame_masks],
            [first_frame_logits],
            [first_frame_ious],
        )
        for t in range(1, num_frames): # for the remaining frames
            self.current_frame_idx = t
            frame_feature = frame_features[t] # obtain current frame feature
            masks, logits, ious, object_score_logits = self._predict_frame( # predict the current mask with previous memory
                frame_feature, memory, prev_pred_mask
            )
            all_masks.append(masks)
            all_logits.append(logits)
            all_ious.append(ious)
            if t < num_frames - 1: # FIFO
                prev_pred_mask = masks if labels is None else labels[:, t]
                memory = self._update_memory(frame_feature, prev_pred_mask, memory, object_score_logits) # store previous frame into memory bank

        self.reset_state()
        return all_masks, all_logits, all_ious

    def normalize_bbox(self, bbox):
        """
        Normalize the given bounding box coordinates.

        This method transforms the bounding box coordinates to a normalized form
        based on the original height and width of the image.

        Args:
            bbox (list or ndarray): The bounding box coordinates to be normalized.

        Returns:
            list or ndarray: The normalized bounding box coordinates.
        """
        unnorm_bbox = self._transforms.transform_boxes(
            bbox, normalize=True, orig_hw=self._orig_hw
        )
        return unnorm_bbox

    def _get_points_placeholder(self, batch_size=None):
        """
        Generates a placeholder for point coordinates and labels.

        Args:
            batch_size (int, optional): The size of the batch. If not provided,
                        defaults to the instance's batch_size attribute.

        Returns:
            tuple: A tuple containing:
            - torch.Tensor: Expanded point coordinates tensor of shape (batch_size, -1, -1).
            - torch.Tensor: Expanded point labels tensor of shape (batch_size, -1).
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        points_placeholder = (
            self.sam_point_coords.expand(batch_size, -1, -1),
            self.sam_point_labels.expand(batch_size, -1),
        )
        return points_placeholder

    def unbind_frame_features(self, frame_features, num_frames):
        """
        Unbind image features from the model.
        """
        keys = frame_features.keys()
        unbinded_frame_features = []
        for frame_idx in range(num_frames):
            frame_feature = {}
            for k in keys:
                frame_feature[k] = (
                    frame_features[k][:, frame_idx]
                    if not isinstance(frame_features[k], list)
                    else [v[:, frame_idx] for v in frame_features[k]]
                )
            unbinded_frame_features.append(frame_feature)
        return unbinded_frame_features

    def preprocess_frame_features(self, frame_features, batch_size, num_frames):
        """
        Preprocess frame features.
        """
        frame_features = self.unbind_frame_features(frame_features, num_frames)
        preprocessed_frame_features = []
        for frame_idx, frame_feature in enumerate(frame_features):
            feature_maps = frame_feature["backbone_fpn"][-self.num_feature_levels :]
            # flatten NxCxHxW to HWxNxC
            vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
            if (
                frame_idx == 0 and self.model.directly_add_no_mem_embed
            ):  # Add no memory embedding
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            # HWxNxC to NxCxHxW
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(
                    vision_feats[::-1], self._bb_feat_sizes[::-1]
                )
            ][::-1]
            _features = {
                "image_embed": feats[-1],
                "high_res_feats": feats[:-1],
                "backbone_fpn": frame_feature["backbone_fpn"][
                    -self.num_feature_levels :
                ],
                "vision_pos_enc": frame_feature["vision_pos_enc"][
                    -self.num_feature_levels :
                ],
            }
            preprocessed_frame_features.append(_features)
        return preprocessed_frame_features

    def _embed_bbox(self, bbox):
        """
        Embed bounding boxes.
        """
        bbox = self.normalize_bbox(bbox)
        box_coords = bbox.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=bbox.device)
        box_labels = box_labels.repeat(bbox.size(0), 1)
        concat_points = (box_coords, box_labels)
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points, boxes=None, masks=None
        )
        return sparse_embeddings, dense_embeddings

    def _predict_first_frame(self, features, bbox):
        """
        Predict masks and IoUs for the first frame.
        """
        sparse_embeddings, dense_embeddings = self._embed_bbox(bbox)

        low_res_masks, ious, sam_output_tokens, object_score_logits = (
            self.model.sam_mask_decoder(
                image_embeddings=features["image_embed"],
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=features["high_res_feats"],
            )
        )

        sam_output_token = sam_output_tokens[:, -1]
        obj_ptr = self.model.obj_ptr_proj(sam_output_token)
        self.obj_ptrs.append(obj_ptr)
        pred_mask, pred_logit = self._postprocess_masks(low_res_masks)
        return pred_mask, pred_logit, ious[:, -1], object_score_logits

    def _postprocess_masks(self, logits, size=None):
        """
        Perform post-processing on output masks.
        """
        size = self._orig_hw if size is None else size
        logits = F.interpolate(logits, size, mode="bilinear", align_corners=False)
        logits = logits[:, -1].unsqueeze(1)
        masks = torch.sigmoid(logits)
        if self.use_mask_threshold:
            masks = (masks > self.mask_threshold).float()
        return masks, logits

    def _extract_memory_features(self, features, masks, object_score_logits):
        """
        Extracts memory features from the given features and masks.

        Args:
            features (dict): A dictionary containing feature maps from the backbone FPN.
            masks (Tensor): A tensor representing the masks to be used by the memory encoder.

        Returns:
            dict: A dictionary containing:
            - "vision_features" (Tensor): The vision features extracted and processed by the memory encoder.
            - "vision_pos_enc" (Tensor): The positional encoding of the vision features.
        """
        pix_feat = features["backbone_fpn"][-1]
        maskmem_out = self.model.memory_encoder(
            pix_feat, masks, skip_mask_sigmoid=True  # sigmoid already applied ap: outout dim: 256->64
        )
        maskmem_features = maskmem_out["vision_features"] # feat+mask as memory
        
        if self.model.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.model.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )
        maskmem_features = maskmem_features.flatten(2).permute(2, 0, 1) # LBC
        maskmem_pos_enc = maskmem_out["vision_pos_enc"][-1].flatten(2).permute(2, 0, 1)
        return {"vision_features": maskmem_features, "vision_pos_enc": maskmem_pos_enc}

    def _initialize_memory(self, features, masks, object_score_logits):
        """
        Initialize memory for the first frame.
        """
        maskmem_out = self._extract_memory_features(features, masks, object_score_logits)
        self.maskmem_features = [maskmem_out["vision_features"]]
        self.maskmem_pos_enc = [maskmem_out["vision_pos_enc"]]
        return self.maskmem_features, self.maskmem_pos_enc

    def _update_memory(self, features, masks, memory=None, object_score_logits=None):
        """
        Update memory with new frame data.
        """
        if memory is None:
            maskmem_features, maskmem_pos_enc = (
                self.maskmem_features, # the first frame memory features
                self.maskmem_pos_enc,
            )
        else:
            maskmem_features, maskmem_pos_enc = memory

        maskmem_out = self._extract_memory_features(features, masks, object_score_logits)
        maskmem_features.append(maskmem_out["vision_features"]) # ap: you can update memory here, e.g: prune memory features
        maskmem_pos_enc.append(maskmem_out["vision_pos_enc"])
        if len(maskmem_features) >= self.memory_size: # ori len(maskmem_features) > self.memory_size:
            # ap: decide prunning or fusing memory features
            # if self.mem_split_ls is True:
            #     # ap: split memory features into two parts, and fuse them
            #     # maskmem_features: len: 7, each [L, B, C]
            #     maskmem_features, maskmem_pos_enc = memory_compressor(maskmem_features, maskmem_pos_enc)
            # else: # prunning based on similarity
            # ap: you can update memory here, e.g: prune memory features
            ## pruning
            # maskmem_features, maskmem_pos_enc = prune_memory_features(maskmem_features, maskmem_pos_enc, prune_frame_num=2, prune_sim=self.prune_sim)
            maskmem_features, maskmem_pos_enc = prune_memory_features_time_split(maskmem_features, maskmem_pos_enc, total_prune_frame_num=2, prune_sim=self.prune_sim)
            if len(maskmem_features) >= self.memory_size:
                self.maskmem_features = maskmem_features[-self.memory_size :] # FIFO: save last 7 frames
                self.maskmem_pos_enc = maskmem_pos_enc[-self.memory_size :]
            else:
                self.maskmem_features = maskmem_features
                self.maskmem_pos_enc = maskmem_pos_enc
        return maskmem_features, maskmem_pos_enc # ori
        # return self.maskmem_features, self.maskmem_pos_enc # ap: it is wrong, because the memory is not updated, it should not be self.maskmem_features and self.maskmem_pos_enc

    def _prepare_memory(self, memory):
        """
        Prepare memory for the current frame.
        """
        if memory is None:
            maskmem_features, maskmem_pos_enc = (
                self.maskmem_features,
                self.maskmem_pos_enc,
            )
        else:
            maskmem_features, maskmem_pos_enc = memory
        for idx in range(len(maskmem_pos_enc)):
            rel_pos = len(maskmem_pos_enc) - idx
            maskmem_pos_enc[idx] = (
                maskmem_pos_enc[idx] + self.model.maskmem_tpos_enc[rel_pos - 1]
            )
        obj_ptrs = torch.stack(self.obj_ptrs, dim=0) # ap: object pointers, shape: [num_obj_ptrs, batch_size, mem_dim], used to track and refer to objects across frames in a video
        
        if self.model.add_tpos_enc_to_obj_ptrs: # ap: add temporal position encoding to object pointers
            max_obj_ptrs_in_encoder = self.num_frames
            pos_list = [self.current_frame_idx]
            t_diff_max = max_obj_ptrs_in_encoder - 1
            tpos_dim = self.model.hidden_dim if self.model.proj_tpos_enc_in_obj_ptrs else self.model.mem_dim
            obj_pos = torch.tensor(pos_list, device=obj_ptrs.device)
            obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
            obj_pos = self.model.obj_ptr_tpos_proj(obj_pos)
            obj_pos = obj_pos.unsqueeze(1).expand(-1, self.batch_size, self.model.mem_dim)
        else:
            obj_pos = obj_ptrs.new_zeros(
                len(self.obj_ptrs), self.batch_size, self.model.mem_dim
            )
        C = self.model.hidden_dim
        if self.model.mem_dim < C:
            # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
            obj_ptrs = obj_ptrs.reshape(
                -1, self.batch_size, C // self.model.mem_dim, self.model.mem_dim
            )
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
            obj_pos = obj_pos.repeat_interleave(C // self.model.mem_dim, dim=0)
        num_obj_ptr_tokens = obj_ptrs.shape[0]
        memory = torch.cat(maskmem_features + [obj_ptrs], dim=0)
        memory_pos_embed = torch.cat(maskmem_pos_enc + [obj_pos], dim=0)
        return memory, memory_pos_embed, num_obj_ptr_tokens

    def _predict_frame(self, features, memory, prev_mask=None):
        """
        Predict masks and IoUs for subsequent frames using memory.
        """
        memory, memory_pos_embed, num_obj_ptr_tokens = self._prepare_memory(memory)

        current_vision_feats = [
            x.flatten(2).permute(2, 0, 1) for x in features["backbone_fpn"]
        ]
        current_vision_pos_embeds = [
            x.flatten(2).permute(2, 0, 1) for x in features["vision_pos_enc"]
        ]
        pix_feat_with_mem = self.model.memory_attention( # update memory with current frame features
            curr=current_vision_feats[-1:],
            curr_pos=current_vision_pos_embeds[-1:],
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(
            *features["backbone_fpn"][-1].shape
        )
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=self._get_points_placeholder(),
            boxes=None,
            masks=None,
        )
        low_res_masks, ious, _, object_score_logits = self.model.sam_mask_decoder(
            image_embeddings=pix_feat_with_mem,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=features["high_res_feats"],
        )

        pred_mask, pred_logit = self._postprocess_masks(low_res_masks)
        return pred_mask, pred_logit, ious[:, -1], object_score_logits



class SAM2VideoTrainer_1024(nn.Module):
    """
    SAM2VideoTrainer is a PyTorch module for training a video segmentation model using SAM2.
    Attributes:
        device (torch.device): The device to run the model on.
        model (nn.Module): The SAM2 video predictor model.
        num_feature_levels (int): Number of feature levels in the model.
        memory_size (int): Size of the memory for storing features.
        _transforms (SAM2Transforms): Transformations applied to the input data.
        _bb_feat_sizes (list): Spatial dimensions for backbone feature maps.
        num_maskmem (int): Number of mask memory features.
        sam_point_coords (torch.Tensor): Placeholder for SAM point coordinates.
        sam_point_labels (torch.Tensor): Placeholder for SAM point labels.
        _orig_hw (list): Original height and width of the input frames.
        maskmem_features (list): List of mask memory features.
        maskmem_pos_enc (list): List of mask memory positional encodings.
        batch_size (int): Batch size of the input data.
        obj_ptrs (list): List of object pointers.
    """

    def __init__(self, model_cfg, sam2_checkpoint, device, memory_size=7, mask_threshold=0.5, use_mask_threshold=False):
        """
        Initializes the SAM2VideoTrainer class.

        Args:
            model_cfg (dict): Configuration dictionary for the model.
            sam2_checkpoint (str): Path to the SAM2 checkpoint file.
            device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
            memory_size (int, optional): Size of the memory. Defaults to 7.
            mask_threshold (float, optional): Threshold for mask prediction. Defaults to 0.5.
            use_mask_threshold (bool, optional): Flag to use mask thresholding. Defaults to False.

        Attributes:
            device (torch.device): The device to run the model on.
            model (SAM2VideoPredictor): The SAM2 video predictor model.
            num_feature_levels (int): Number of feature levels in the model.
            memory_size (int): Size of the memory.
            _transforms (SAM2Transforms): Transformations applied to the input data.
            _bb_feat_sizes (list): Spatial dimensions for backbone feature maps.
            num_maskmem (int): Number of mask memories.
            sam_point_coords (torch.Tensor): Tensor for SAM point coordinates.
            sam_point_labels (torch.Tensor): Tensor for SAM point labels.
            mask_threshold (float): Threshold for mask prediction.
            use_mask_threshold (bool): Flag to use mask thresholding.
        """
        super().__init__()
        self.device = device
        self.model = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=self.device, mode="train"
        )
        self.model.train()
        self.num_feature_levels = self.model.num_feature_levels

        self.num_maskmem = 7
        self.memory_size = (
            memory_size if memory_size <= self.num_maskmem else self.num_maskmem
        )

        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=0.5,
            max_hole_area=0,
            max_sprinkle_area=0,
        )

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256), # aping
            (128, 128),
            (64, 64),
            # (32, 32),
        ]

        self.sam_point_coords = torch.zeros(1, 1, 2, device=device)
        self.sam_point_labels = -torch.ones(1, 1, dtype=torch.int32, device=device)
        self.mask_threshold = mask_threshold
        self.use_mask_threshold = use_mask_threshold

        self.init_state()

    def init_state(self):
        """
        Initializes the state variables for the video trainer.

        This method sets the initial state of various attributes used in the video
        training process. It resets the original height and width, mask memory
        features, mask memory positional encoding, batch size, and object pointers
        to their default values.

        Attributes:
            _orig_hw (tuple or None): Original height and width of the video frames.
            maskmem_features (Any or None): Features related to mask memory.
            maskmem_pos_enc (Any or None): Positional encoding for mask memory.
            batch_size (int or None): Size of the batch for training.
            obj_ptrs (list): List of object pointers used in the training process.
        """
        self._orig_hw = None
        self.maskmem_features = None
        self.maskmem_pos_enc = None
        self.batch_size = None
        self.current_frame_idx = 0
        self.obj_ptrs = []
        self.num_frames = 0

    def reset_state(self):
        """
        Resets the state of the video trainer.

        This method clears the internal state variables, setting them to their initial values:
        - `_orig_hw`: Set to None. Represents the original height and width.
        - `maskmem_features`: Set to None. Represents the mask memory features.
        - `maskmem_pos_enc`: Set to None. Represents the mask memory positional encoding.
        - `batch_size`: Set to None. Represents the batch size.
        - `obj_ptrs`: Set to an empty list. Represents the object pointers.
        """
        self._orig_hw = None
        self.maskmem_features = None
        self.maskmem_pos_enc = None
        self.batch_size = None
        self.current_frame_idx = 0
        self.obj_ptrs = []
        self.num_frames = 0

    def forward(self, videos, bboxes, labels=None):
        """
        Forward pass for processing video frames and predicting masks, logits, and IoUs.

        Args:
            videos (torch.Tensor): A tensor of shape (batch_size, num_frames, C, H, W) representing the input video frames.
            bboxes (torch.Tensor): A tensor of shape (batch_size, 4) representing the bounding boxes for the first frame.
            labels (torch.Tensor, optional): A tensor of shape (batch_size, num_frames, H, W) representing the ground truth masks for each frame. Defaults to None.

        Returns:
            tuple: A tuple containing:
            - all_masks (list of torch.Tensor): A list of tensors representing the predicted masks for each frame.
            - all_logits (list of torch.Tensor): A list of tensors representing the predicted logits for each frame.
            - all_ious (list of torch.Tensor): A list of tensors representing the predicted IoUs for each frame.
        """
        self.init_state()
        batch_size, num_frames, C, H, W = videos.shape
        self.num_frames = num_frames
        self._orig_hw = [H, W]
        self.batch_size = batch_size

        # Extract features for all frames in the video
        videos = videos.view(batch_size * num_frames, C, H, W)
        features = self.model.forward_image(videos)  # Extract features for all frames
        features = {
            k: (
                v.view(batch_size, num_frames, *v.shape[1:])
                if not isinstance(v, list)
                else ([_v.view(batch_size, num_frames, *_v.shape[1:]) for _v in v])
            )
            for k, v in features.items()
        }
        frame_features = self.preprocess_frame_features(
            features, batch_size, num_frames
        )

        # Process the first frame with bounding boxes as prompts
        first_frame_features = frame_features[0]
        first_frame_bbox = bboxes.view(batch_size, 4)

        # Predict the first frame masks and IoUs
        first_frame_masks, first_frame_logits, first_frame_ious, object_score_logits = (
            self._predict_first_frame(first_frame_features, first_frame_bbox)
        )

        # Initialize memory with first frame predictions
        prev_pred_mask = first_frame_masks if labels is None else labels[:, 0]
        memory = self._initialize_memory(first_frame_features, prev_pred_mask, object_score_logits)

        # Process remaining frames
        all_masks, all_logits, all_ious = (
            [first_frame_masks],
            [first_frame_logits],
            [first_frame_ious],
        )
        for t in range(1, num_frames):
            self.current_frame_idx = t
            frame_feature = frame_features[t]
            masks, logits, ious, object_score_logits = self._predict_frame(
                frame_feature, memory, prev_pred_mask
            )
            all_masks.append(masks)
            all_logits.append(logits)
            all_ious.append(ious)
            if t < num_frames - 1:
                prev_pred_mask = masks if labels is None else labels[:, t]
                memory = self._update_memory(frame_feature, prev_pred_mask, memory, object_score_logits)

        self.reset_state()
        return all_masks, all_logits, all_ious

    def normalize_bbox(self, bbox):
        """
        Normalize the given bounding box coordinates.

        This method transforms the bounding box coordinates to a normalized form
        based on the original height and width of the image.

        Args:
            bbox (list or ndarray): The bounding box coordinates to be normalized.

        Returns:
            list or ndarray: The normalized bounding box coordinates.
        """
        unnorm_bbox = self._transforms.transform_boxes(
            bbox, normalize=True, orig_hw=self._orig_hw
        )
        return unnorm_bbox

    def _get_points_placeholder(self, batch_size=None):
        """
        Generates a placeholder for point coordinates and labels.

        Args:
            batch_size (int, optional): The size of the batch. If not provided,
                        defaults to the instance's batch_size attribute.

        Returns:
            tuple: A tuple containing:
            - torch.Tensor: Expanded point coordinates tensor of shape (batch_size, -1, -1).
            - torch.Tensor: Expanded point labels tensor of shape (batch_size, -1).
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        points_placeholder = (
            self.sam_point_coords.expand(batch_size, -1, -1),
            self.sam_point_labels.expand(batch_size, -1),
        )
        return points_placeholder

    def unbind_frame_features(self, frame_features, num_frames):
        """
        Unbind image features from the model.
        """
        keys = frame_features.keys()
        unbinded_frame_features = []
        for frame_idx in range(num_frames):
            frame_feature = {}
            for k in keys:
                frame_feature[k] = (
                    frame_features[k][:, frame_idx]
                    if not isinstance(frame_features[k], list)
                    else [v[:, frame_idx] for v in frame_features[k]]
                )
            unbinded_frame_features.append(frame_feature)
        return unbinded_frame_features

    def preprocess_frame_features(self, frame_features, batch_size, num_frames):
        """
        Preprocess frame features.
        """
        frame_features = self.unbind_frame_features(frame_features, num_frames)
        preprocessed_frame_features = []
        for frame_idx, frame_feature in enumerate(frame_features):
            feature_maps = frame_feature["backbone_fpn"][-self.num_feature_levels :]
            # flatten NxCxHxW to HWxNxC
            vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
            if (
                frame_idx == 0 and self.model.directly_add_no_mem_embed
            ):  # Add no memory embedding
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            # HWxNxC to NxCxHxW
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(
                    vision_feats[::-1], self._bb_feat_sizes[::-1]
                )
            ][::-1]
            _features = {
                "image_embed": feats[-1],
                "high_res_feats": feats[:-1],
                "backbone_fpn": frame_feature["backbone_fpn"][
                    -self.num_feature_levels :
                ],
                "vision_pos_enc": frame_feature["vision_pos_enc"][
                    -self.num_feature_levels :
                ],
            }
            preprocessed_frame_features.append(_features)
        return preprocessed_frame_features

    def _embed_bbox(self, bbox):
        """
        Embed bounding boxes.
        """
        bbox = self.normalize_bbox(bbox)
        box_coords = bbox.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=bbox.device)
        box_labels = box_labels.repeat(bbox.size(0), 1)
        concat_points = (box_coords, box_labels)
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points, boxes=None, masks=None
        )
        return sparse_embeddings, dense_embeddings

    def _predict_first_frame(self, features, bbox):
        """
        Predict masks and IoUs for the first frame.
        """
        sparse_embeddings, dense_embeddings = self._embed_bbox(bbox)

        low_res_masks, ious, sam_output_tokens, object_score_logits = (
            self.model.sam_mask_decoder(
                image_embeddings=features["image_embed"],
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=features["high_res_feats"],
            )
        )

        sam_output_token = sam_output_tokens[:, -1]
        obj_ptr = self.model.obj_ptr_proj(sam_output_token)
        self.obj_ptrs.append(obj_ptr)
        pred_mask, pred_logit = self._postprocess_masks(low_res_masks)
        return pred_mask, pred_logit, ious[:, -1], object_score_logits

    def _postprocess_masks(self, logits, size=None):
        """
        Perform post-processing on output masks.
        """
        size = self._orig_hw if size is None else size
        logits = F.interpolate(logits, size, mode="bilinear", align_corners=False)
        logits = logits[:, -1].unsqueeze(1)
        masks = torch.sigmoid(logits)
        if self.use_mask_threshold:
            masks = (masks > self.mask_threshold).float()
        return masks, logits

    def _extract_memory_features(self, features, masks, object_score_logits):
        """
        Extracts memory features from the given features and masks.

        Args:
            features (dict): A dictionary containing feature maps from the backbone FPN.
            masks (Tensor): A tensor representing the masks to be used by the memory encoder.

        Returns:
            dict: A dictionary containing:
            - "vision_features" (Tensor): The vision features extracted and processed by the memory encoder.
            - "vision_pos_enc" (Tensor): The positional encoding of the vision features.
        """
        pix_feat = features["backbone_fpn"][-1]
        maskmem_out = self.model.memory_encoder(
            pix_feat, masks, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        
        if self.model.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.model.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )
        maskmem_features = maskmem_features.flatten(2).permute(2, 0, 1)
        maskmem_pos_enc = maskmem_out["vision_pos_enc"][-1].flatten(2).permute(2, 0, 1)
        return {"vision_features": maskmem_features, "vision_pos_enc": maskmem_pos_enc}

    def _initialize_memory(self, features, masks, object_score_logits):
        """
        Initialize memory for the first frame.
        """
        maskmem_out = self._extract_memory_features(features, masks, object_score_logits)
        self.maskmem_features = [maskmem_out["vision_features"]]
        self.maskmem_pos_enc = [maskmem_out["vision_pos_enc"]]
        return self.maskmem_features, self.maskmem_pos_enc

    def _update_memory(self, features, masks, memory=None, object_score_logits=None):
        """
        Update memory with new frame data.
        """
        if memory is None:
            maskmem_features, maskmem_pos_enc = (
                self.maskmem_features,
                self.maskmem_pos_enc,
            )
        else:
            maskmem_features, maskmem_pos_enc = memory

        maskmem_out = self._extract_memory_features(features, masks, object_score_logits)
        maskmem_features.append(maskmem_out["vision_features"])
        maskmem_pos_enc.append(maskmem_out["vision_pos_enc"])
        if len(maskmem_features) > self.memory_size:
            self.maskmem_features = maskmem_features[-self.memory_size :]
            self.maskmem_pos_enc = maskmem_pos_enc[-self.memory_size :]
        return maskmem_features, maskmem_pos_enc

    def _prepare_memory(self, memory):
        """
        Prepare memory for the current frame.
        """
        if memory is None:
            maskmem_features, maskmem_pos_enc = (
                self.maskmem_features,
                self.maskmem_pos_enc,
            )
        else:
            maskmem_features, maskmem_pos_enc = memory
        for idx in range(len(maskmem_pos_enc)):
            rel_pos = len(maskmem_pos_enc) - idx
            maskmem_pos_enc[idx] = (
                maskmem_pos_enc[idx] + self.model.maskmem_tpos_enc[rel_pos - 1]
            )
        obj_ptrs = torch.stack(self.obj_ptrs, dim=0)
        
        if self.model.add_tpos_enc_to_obj_ptrs:
            max_obj_ptrs_in_encoder = self.num_frames
            pos_list = [self.current_frame_idx]
            t_diff_max = max_obj_ptrs_in_encoder - 1
            tpos_dim = self.model.hidden_dim if self.model.proj_tpos_enc_in_obj_ptrs else self.model.mem_dim
            obj_pos = torch.tensor(pos_list, device=obj_ptrs.device)
            obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
            obj_pos = self.model.obj_ptr_tpos_proj(obj_pos)
            obj_pos = obj_pos.unsqueeze(1).expand(-1, self.batch_size, self.model.mem_dim)
        else:
            obj_pos = obj_ptrs.new_zeros(
                len(self.obj_ptrs), self.batch_size, self.model.mem_dim
            )
        C = self.model.hidden_dim
        if self.model.mem_dim < C:
            # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
            obj_ptrs = obj_ptrs.reshape(
                -1, self.batch_size, C // self.model.mem_dim, self.model.mem_dim
            )
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
            obj_pos = obj_pos.repeat_interleave(C // self.model.mem_dim, dim=0)
        num_obj_ptr_tokens = obj_ptrs.shape[0]
        memory = torch.cat(maskmem_features + [obj_ptrs], dim=0)
        memory_pos_embed = torch.cat(maskmem_pos_enc + [obj_pos], dim=0)
        return memory, memory_pos_embed, num_obj_ptr_tokens

    def _predict_frame(self, features, memory, prev_mask=None):
        """
        Predict masks and IoUs for subsequent frames using memory.
        """
        memory, memory_pos_embed, num_obj_ptr_tokens = self._prepare_memory(memory)

        current_vision_feats = [
            x.flatten(2).permute(2, 0, 1) for x in features["backbone_fpn"]
        ]
        current_vision_pos_embeds = [
            x.flatten(2).permute(2, 0, 1) for x in features["vision_pos_enc"]
        ]
        pix_feat_with_mem = self.model.memory_attention(
            curr=current_vision_feats[-1:],
            curr_pos=current_vision_pos_embeds[-1:],
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(
            *features["backbone_fpn"][-1].shape
        )
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=self._get_points_placeholder(),
            boxes=None,
            masks=None,
        )
        low_res_masks, ious, _, object_score_logits = self.model.sam_mask_decoder(
            image_embeddings=pix_feat_with_mem,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=features["high_res_feats"],
        )

        pred_mask, pred_logit = self._postprocess_masks(low_res_masks)
        return pred_mask, pred_logit, ious[:, -1], object_score_logits




class TestSAM2VideoTrainer(unittest.TestCase):
    def setUp(self):
        # Initialize parameters
        sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create an instance of SAM2VideoTrainer
        self.trainer = SAM2VideoTrainer(model_cfg, sam2_checkpoint, device=self.device)

        # Define input video and bounding boxes
        self.batch_size = 2
        self.num_frames = 2
        self.C = 3
        self.H = 1024
        self.W = 1024

        # Create random video data and bounding boxes
        self.videos = torch.randn(
            self.batch_size, self.num_frames, self.C, self.H, self.W
        ).to(self.device)
        self.masks = torch.zeros(
            self.batch_size, self.num_frames, 1, self.H, self.W
        ).to(self.device)
        self.bboxes = torch.tensor(
            [[100, 100, 200, 200], [150, 150, 250, 250]], dtype=torch.float32
        ).to(self.device)

    def test_forward(self):
        # Execute the forward method
        # masks, ious = self.trainer(self.videos, self.bboxes, self.masks)
        masks, ious = self.trainer(self.videos, self.bboxes, None)

        print("Masks shape:", masks[0].shape)
        print("IoUs shape:", ious[0].shape)

        print("Masks:", masks)
        print("IoUs:", ious)


if __name__ == "__main__":
    unittest.main()
