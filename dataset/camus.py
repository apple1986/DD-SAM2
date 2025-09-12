import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import nibabel as nib

class CAMUS(Dataset):
    """ CAMUS Dataset for multi-object segmentation and tracking """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        # Load list of relative paths
        split_file = os.path.join(self._base_dir, f'../dataset/CAMUS/{split}_camus_rel_path.txt')
        with open(split_file, 'r') as f:
            self.curMR_list = [line.strip() for line in f.readlines()]

        if num is not None:
            self.curMR_list = self.curMR_list[:num]


        self.curMR_list = [str(item.replace(' \n', '')) for item in self.curMR_list]
        print(f"Total {len(self.curMR_list)} samples loaded for split: {split}")
        self.abs_path = os.path.join(base_dir, "../dataset/CAMUS")

    def __len__(self):
        return len(self.curMR_list)

    def __getitem__(self, idx):
        sample_path = self.curMR_list[idx]  # e.g. "sequence01.npz"
        curMR_paths = sample_path.split(" ")
        curMR_path = os.path.join(self.abs_path, curMR_paths[0])
        curGT_path = os.path.join(self.abs_path, curMR_paths[1])
        
        ## open files
        curMR = nib.load(curMR_path).get_fdata()
        curGT = nib.load(curGT_path).get_fdata()


        ## doing precessing: shape HWD-># CHWD
        video = np.expand_dims(curMR, axis=0) # shape [1, H, W, D]
        mask = np.expand_dims(curGT, axis=0) # shape [1, H, W, D]

        video = (video - video.min()) / (video.max() - video.min() + 1e-10)
        mask = mask.astype(np.uint8)
        # mask = torch.from_numpy(mask)

        # First-frame GT (used as prompt)
        first_gt = mask[0]  # shape [1, H, W]

        # Extract case ID from filename (e.g., "sequence01")
        case_id = os.path.basename(curMR_path)

        sample = {
            'curMR': video,     # [N, 3, H, W]
            'curGT': mask,      # [N, 7, H, W]
            'firstGT': first_gt,  # [1, H, W]
            'caseID': case_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample



def convert_mask_exclude_background(mask, num_classes=8):
    """
    Convert [N, 1, H, W] label mask to one-hot [N, C, H, W] excluding background (class 0).

    Args:
        mask: torch.Tensor, shape [N, 1, H, W], values in 0 to num_classes-1
        num_classes: int, total number of classes including background (e.g., 8)

    Returns:
        one_hot_mask: [N, C-1, H, W] (excluding background)
    """
    N, _, H, W = mask.shape
    mask = mask.squeeze(1)  # [N, H, W]

    one_hot = F.one_hot(mask.long(), num_classes=num_classes)  # [N, H, W, C]
    one_hot = one_hot.permute(0, 3, 1, 2).float()  # [N, C, H, W]
    return one_hot[:, 1:].numpy()  # exclude background (class 0)


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        curMR, curGT, firstGT, caseID = sample['curMR'], sample['curGT'], sample['firstGT'], sample['caseID']

        # pad the sample if necessary
        if curGT.shape[0] <= self.output_size[0] or curGT.shape[1] <= self.output_size[1] or curGT.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - curGT.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - curGT.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - curGT.shape[2]) // 2 + 3, 0)
            curMR = np.pad(curMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            # firstGT = np.pad(firstGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = curMR.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        curGT = curGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        curMR = curMR[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        # firstGT = firstGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'curMR': curMR, 'curGT': curGT, 'firstGT': firstGT, "caseID": caseID}


class RandomCrop(object):
    """
    Crop randomly the curMR in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        curMR, curGT, firstGT, caseID = sample['curMR'], sample['curGT'], sample['firstGT'], sample['caseID']

        # pad the sample if necessary
        if curGT.shape[0] <= self.output_size[0] or curGT.shape[1] <= self.output_size[1] or curGT.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - curGT.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - curGT.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - curGT.shape[2]) // 2 + 3, 0)
            curMR = np.pad(curMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            firstGT = np.pad(firstGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = curMR.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        curGT = curGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        curMR = curMR[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        firstGT = firstGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        curGT = curGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'curMR': curMR, 'curGT': curGT, 'firstGT': firstGT, "caseID": caseID}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        curMR, curGT, firstGT, caseID = sample['curMR'], sample['curGT'], sample['firstGT'], sample['caseID']
        k = np.random.randint(0, 4)
        curMR = np.rot90(curMR, k)
        curGT = np.rot90(curGT, k)
        axis = np.random.randint(0, 2)
        curMR = np.flip(curMR, axis=axis).copy()
        curGT = np.flip(curGT, axis=axis).copy()

        curGT = np.rot90(curGT, k)
        firstGT = np.rot90(firstGT, k)
        axis = np.random.randint(0, 2)
        curGT = np.flip(curGT, axis=axis).copy()
        firstGT = np.flip(firstGT, axis=axis).copy()


        return {'curMR': curMR, 'curGT': curGT, 'firstGT': firstGT, "caseID": caseID}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        curMR, curGT, firstGT, caseID = sample['curMR'], sample['curGT'], sample['firstGT'], sample['caseID']
        noise = np.clip(self.sigma * np.random.randn(curMR.shape[0], curMR.shape[1], curMR.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        curMR = curMR + noise
        curGT = curGT + noise
        return {'curMR': curMR, 'curGT': curGT, 'firstGT': firstGT, "caseID": caseID}


class CreateOnehotcurGT(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        curMR, curGT, firstGT, caseID = sample['curMR'], sample['curGT'], sample['firstGT'], sample['caseID']
        onehot_curGT = np.zeros((self.num_classes, curGT.shape[0], curGT.shape[1], curGT.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_curGT[i, :, :, :] = (curGT == i).astype(np.float32)
        return {'curMR': curMR, 'curGT': curGT,'onehot_curGT':onehot_curGT}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        curMR = sample['curMR']
        curMR = curMR.astype(np.float32)
        curGT = sample['curGT']
        curGT = curGT.astype(np.float32)

        if 'onehot_curGT' in sample:
            return {'curMR': torch.from_numpy(curMR), 'curGT': torch.from_numpy(sample['curGT']).long(),
                    'onehot_curGT': torch.from_numpy(sample['onehot_curGT']).long(),
                    'caseID': sample['caseID']}
        else:
            return {'curMR': torch.from_numpy(curMR), 'curGT': torch.from_numpy(curGT), 
                     'firstGT': torch.from_numpy(curGT).long(), 
                     'caseID': sample['caseID']}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)