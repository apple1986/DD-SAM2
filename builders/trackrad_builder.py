import os
from torch.utils.data import DataLoader
from dataset.trackrad2025 import TrackRAD2025

## load train, validation or test dataset
def load_data(root_path, transform_policy, batch_size, split="train", use_num_cases=None):
    """
    root_path: the project root path
    transform_policy: transform method
    batch_size: batch size for training
    split: train, val or test
    """
    ## set data path, data transform method

    sample_dataset = TrackRAD2025(base_dir=root_path, split=split, 
                            transform=transform_policy, use_num_cases=use_num_cases)
    if split == "train":
        print(f"total samples: {len(sample_dataset)}")
        sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,) # training dataset
        print(f"{len(sample_loader)} iterations per epoch")
    else:
        sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,) # validation or testing dataset
        print(f"total {split} samples: {len(sample_dataset)}")
    return sample_loader
