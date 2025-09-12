import os
from sympy import use
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import nibabel as nib
import SimpleITK

class TrackRAD2025(Dataset):
    """ TrackRAD2025 Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None, use_num_cases=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/../dataset/trackrad2025_labeled_training_data/train_trackrad_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
            if use_num_cases is not None:
                self.curMR_list = self.curMR_list[:use_num_cases]
        elif split == 'val':
            with open(self._base_dir+'/../dataset/trackrad2025_labeled_training_data/val_trackrad_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../dataset/trackrad2025_labeled_training_data/test_trackrad_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
 
        self.curMR_list = [str(item.replace(' \n', '')) for item in self.curMR_list]
        if num is not None:
            self.curMR_list = self.curMR_list[:num]
        print("total {} samples".format(len(self.curMR_list)))
 
        self.abs_path = os.path.join(base_dir, "../dataset/trackrad2025_labeled_training_data")

    def __len__(self):
        return len(self.curMR_list)

    def __getitem__(self, idx):
        curMR_name = self.curMR_list[idx] # path of the files: curMR, curGT, the first slice of curGT
        ## open files
        curMR_paths = curMR_name.split(" ")

        curMR_obj = SimpleITK.ReadImage(os.path.join(self.abs_path, curMR_paths[0]))
        curMR = SimpleITK.GetArrayFromImage(curMR_obj) # HWD
        # curMR = np.transpose(curMR, (2, 0, 1))  # make HWD to DHW

        curGT_obj = SimpleITK.ReadImage(os.path.join(self.abs_path, curMR_paths[1]))
        curGT = SimpleITK.GetArrayFromImage(curGT_obj) # HWD
        # curGT = np.transpose(curGT, (2, 0, 1))  # make HWD to DHW

        firstGT_obj = SimpleITK.ReadImage(os.path.join(self.abs_path, curMR_paths[2]))
        firstGT = SimpleITK.GetArrayFromImage(firstGT_obj) # HWD
        # firstGT = np.transpose(firstGT, (2, 0, 1))  # make HWD to DHW

  
        caseID = curMR_paths[0].split('/')[2].split('_frames.mha')[0] # like: 70125695/70125695_20240828_mr.nii.gz

        ## normalization
        curMR = (curMR - curMR.min()) / (curMR.max() - curMR.min() + 1e-10)

        sample = {'curMR': curMR, 'curGT': curGT, 'firstGT': firstGT, "caseID": caseID}
        if self.transform:
            sample = self.transform(sample)

        return sample



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        curMR = sample['curMR']
        curMR = curMR.reshape(1, curMR.shape[0], curMR.shape[1], curMR.shape[2]).astype(np.float32)
        curGT = sample['curGT']
        curGT = curGT.reshape(1, curGT.shape[0], curGT.shape[1], curGT.shape[2]).astype(np.float32)
        # firstGT = sample['firstGT']
        # firstGT = firstGT.reshape(1, firstGT.shape[0], firstGT.shape[1], firstGT.shape[2]).astype(np.float32)

        if 'onehot_curGT' in sample:
            return {'curMR': torch.from_numpy(curMR), 'curGT': torch.from_numpy(sample['curGT']).long(),
                    'onehot_curGT': torch.from_numpy(sample['onehot_curGT']).long(),
                    'caseID': sample['caseID']}
        else:
            return {'curMR': torch.from_numpy(curMR), 'curGT': torch.from_numpy(curGT), 
                     'firstGT': torch.from_numpy(curGT).long(), 
                     'caseID': sample['caseID']}


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