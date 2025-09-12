import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import nibabel as nib
import SimpleITK
import cv2

def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame

    # v = v.transpose((3, 0, 1, 2))
    v = v.transpose((3, 1, 2, 0)) # CHWD

    return v

class EchoCard(Dataset):
    """ EchoCard Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/../dataset/EchoNet_Dynamic/train_echocard_rel_path.txt', 'r') as f:
                self.curVid_list = f.readlines()
        elif split == 'val':
            with open(self._base_dir+'/../dataset/EchoNet_Dynamic/val_echocard_rel_path.txt', 'r') as f:
                self.curVid_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../dataset/EchoNet_Dynamic/test_echocard_rel_path.txt', 'r') as f:
                self.curVid_list = f.readlines()
 
        self.curVid_list = [str(item.replace(' \n', '')) for item in self.curVid_list]
        if num is not None:
            self.curVid_list = self.curVid_list[:num]
        print("total {} samples".format(len(self.curVid_list)))
 
        self.abs_path = os.path.join(base_dir, "../dataset/EchoNet_Dynamic")

    def __len__(self):
        return len(self.curVid_list)

    def __getitem__(self, idx):
        curVid_name = self.curVid_list[idx] # path of the files: curVid, curGT, the first slice of curGT
        ## open files
        curVid_paths = curVid_name.split(" ")

        curVid = loadvideo(os.path.join(self.abs_path, curVid_paths[0])) #3HWD
        curGT = loadvideo(os.path.join(self.abs_path, curVid_paths[1])) # 3HWD
        curGT = (curGT > 125).astype(np.uint8)[0:1, ...] # threshold: 1HWD

  
        caseID = curVid_paths[0].split('/')[-1].split('.avi')[0] # 

        ## normalization
        curVid = (curVid - curVid.min()) / (curVid.max() - curVid.min() + 1e-10)

        sample = {'curVid': curVid, 'curGT': curGT,  "caseID": caseID}
        if self.transform:
            sample = self.transform(sample)

        return sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        curVid, curGT,  caseID = sample['curVid'], sample['curGT'],  sample['caseID']

        # pad the sample if necessary
        if curGT.shape[0] <= self.output_size[0] or curGT.shape[1] <= self.output_size[1] or curGT.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - curGT.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - curGT.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - curGT.shape[2]) // 2 + 3, 0)
            curVid = np.pad(curVid, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            # firstGT = np.pad( [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = curVid.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        curGT = curGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        curVid = curVid[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        # firstGT = firstGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'curVid': curVid, 'curGT': curGT,   "caseID": caseID}


class RandomCrop(object):
    """
    Crop randomly the curVid in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        curVid, curGT,  caseID = sample['curVid'], sample['curGT'],  sample['caseID']

        # pad the sample if necessary
        if curGT.shape[0] <= self.output_size[0] or curGT.shape[1] <= self.output_size[1] or curGT.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - curGT.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - curGT.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - curGT.shape[2]) // 2 + 3, 0)
            curVid = np.pad(curVid, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            firstGT = np.pad( [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = curVid.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        curGT = curGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        curVid = curVid[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        firstGT = firstGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        curGT = curGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'curVid': curVid, 'curGT': curGT,   "caseID": caseID}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        curVid, curGT,  caseID = sample['curVid'], sample['curGT'],  sample['caseID']
        k = np.random.randint(0, 4)
        curVid = np.rot90(curVid, k)
        curGT = np.rot90(curGT, k)
        axis = np.random.randint(0, 2)
        curVid = np.flip(curVid, axis=axis).copy()
        curGT = np.flip(curGT, axis=axis).copy()

        curGT = np.rot90(curGT, k)
        firstGT = np.rot90( k)
        axis = np.random.randint(0, 2)
        curGT = np.flip(curGT, axis=axis).copy()
        firstGT = np.flip( axis=axis).copy()


        return {'curVid': curVid, 'curGT': curGT,   "caseID": caseID}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        curVid, curGT,  caseID = sample['curVid'], sample['curGT'],  sample['caseID']
        noise = np.clip(self.sigma * np.random.randn(curVid.shape[0], curVid.shape[1], curVid.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        curVid = curVid + noise
        curGT = curGT + noise
        return {'curVid': curVid, 'curGT': curGT,   "caseID": caseID}


class CreateOnehotcurGT(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        curVid, curGT,  caseID = sample['curVid'], sample['curGT'],  sample['caseID']
        onehot_curGT = np.zeros((self.num_classes, curGT.shape[0], curGT.shape[1], curGT.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_curGT[i, :, :, :] = (curGT == i).astype(np.float32)
        return {'curVid': curVid, 'curGT': curGT,'onehot_curGT':onehot_curGT}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        curVid = sample['curVid']
        curVid = curVid.reshape(3, curVid.shape[1], curVid.shape[2], curVid.shape[3]).astype(np.float32)
        curGT = sample['curGT']
        curGT = curGT.reshape(1, curGT.shape[1], curGT.shape[2], curGT.shape[3]).astype(np.float32)

        if 'onehot_curGT' in sample:
            return {'curVid': torch.from_numpy(curVid), 'curGT': torch.from_numpy(sample['curGT']).long(),
                    'onehot_curGT': torch.from_numpy(sample['onehot_curGT']).long(),
                    'caseID': sample['caseID']}
        else:
            return {'curVid': torch.from_numpy(curVid), 'curGT': torch.from_numpy(curGT), 
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