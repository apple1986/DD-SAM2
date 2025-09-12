import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import nibabel as nib
import SimpleITK as sitk
import torchio as tio
import ants

# def register_and_transform(preMR_path, curMR_path, preGT_path):
#     # Load input images
#     preMR = sitk.ReadImage(preMR_path, sitk.sitkFloat32)
#     curMR = sitk.ReadImage(curMR_path, sitk.sitkFloat32)
#     preGT = sitk.ReadImage(preGT_path, sitk.sitkUInt8)

#     # Set up Elastix for rigid registration
#     elastixImageFilter = sitk.ElastixImageFilter()
#     elastixImageFilter.SetFixedImage(curMR)
#     elastixImageFilter.SetMovingImage(preMR)
    
#     parameterMap = sitk.GetDefaultParameterMap("rigid")  # Options: "rigid", "affine", "bspline"
#     elastixImageFilter.SetParameterMap(parameterMap)
#     elastixImageFilter.Execute()

#     # Get registered preMR image
#     preMR_registered = elastixImageFilter.GetResultImage()

#     # Transform the ground truth label
#     transformix = sitk.TransformixImageFilter()
#     transformix.SetMovingImage(preGT)
#     transformix.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
#     transformix.Execute()

#     curGT_estimated = transformix.GetResultImage()

#     # Convert to numpy arrays and return
#     curGT_np = sitk.GetArrayFromImage(curGT_estimated)
#     # preMR_registered_np = sitk.GetArrayFromImage(preMR_registered)

#     return curGT_np #, preMR_registered_np


# def register_with_torchio(preMR_path, curMR_path, preGT_path):
#     # Load images
#     preMR = tio.ScalarImage(preMR_path)
#     curMR = tio.ScalarImage(curMR_path)
#     preGT = tio.LabelMap(preGT_path)

#     # Rigid registration
#     registration = tio.Registration(
#         method='Rigid',
#         moving=preMR,
#         fixed=curMR,
#     )
#     transform = registration.fit()
    
#     preMR_registered = transform(preMR)
#     curGT_estimated = transform(preGT)

#     # Convert to numpy
#     curGT_np = curGT_estimated.data.squeeze().numpy()
#     # preMR_reg_np = preMR_registered.data.squeeze().numpy()
#     # preMR_np = preMR.data.squeeze().numpy()
#     # preGT_np = preGT.data.squeeze().numpy()

#     return curGT_np #, preMR_reg_np, preMR_np, preGT_np

def register_and_transform_ants(preMR_path, curMR_path, preGT_path, transform_type="SyN"):
    # Load images
    fixed = ants.image_read(curMR_path)      # current MR (target)
    moving = ants.image_read(preMR_path)     # pre-treatment MR (source)
    label = ants.image_read(preGT_path)      # pre-treatment GT

    # Perform rigid registration
    transform = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform_type) # Similarity Rigid SyN

    # Apply the transform to moving label (nearest interpolation for masks)
    warped_label = ants.apply_transforms(
        fixed=fixed,
        moving=label,
        transformlist=transform['fwdtransforms'],
        interpolator='nearestNeighbor'
    )

    # Return NumPy arrays
    curGT_np = warped_label.numpy()
    # preMR_np = moving.numpy()
    # curMR_np = fixed.numpy()
    # preGT_np = label.numpy()

    return curGT_np#, preMR_np, curMR_np, preGT_np


def get_cur_tumor_from_registration(pre_liver_path, cur_liver_path, pre_tumor_path, transform_type):
    """
    Registers preLiverGT to curLiverGT and transforms preTumorGT accordingly.

    Parameters:
        pre_liver_path (str): path to previous liver mask (preLiverGT)
        cur_liver_path (str): path to current liver mask (curLiverGT)
        pre_tumor_path (str): path to previous tumor mask (preTumorGT)

    Returns:
        curTumorGT (np.ndarray): estimated tumor mask in current space (NumPy array)
    """
    # Load all masks
    pre_liver = ants.image_read(pre_liver_path)
    cur_liver = ants.image_read(cur_liver_path)
    pre_tumor = ants.image_read(pre_tumor_path)

    # Register preLiver to curLiver (use mask-to-mask registration)
    tx = ants.registration(
        fixed=cur_liver,
        moving=pre_liver,
        type_of_transform=transform_type,  # or 'Affine'/'SyN' for more complex Rigid
        verbose=False
    )

    # Apply the transform to preTumorGT
    cur_tumor_estimated = ants.apply_transforms(
        fixed=cur_liver,
        moving=pre_tumor,
        transformlist=tx['fwdtransforms'],
        interpolator='nearestNeighbor'
    )

    return cur_tumor_estimated.numpy()


def register_and_estimate_curTumor(preMR_path, curMR_path, preLiverGT_path, curLiverGT_path, preTumorGT_path, transform_type='SyN'):
    preMR = ants.image_read(preMR_path)
    curMR = ants.image_read(curMR_path)
    preLiverGT = ants.image_read(preLiverGT_path)
    curLiverGT = ants.image_read(curLiverGT_path)
    preTumorGT = ants.image_read(preTumorGT_path)

    registration = ants.registration(
        fixed=curMR,
        moving=preMR,
        type_of_transform=transform_type,
        fixed_mask=curLiverGT,
        moving_mask=preLiverGT,
        verbose=False
    )

    curTumor_estimated = ants.apply_transforms(
        fixed=curMR,
        moving=preTumorGT,
        transformlist=registration['fwdtransforms'],
        interpolator='nearestNeighbor'
    )

    return curTumor_estimated.numpy()


###############################################
class UTSWLiver_Reg(Dataset):
    """ Liver Dataset """
    def __init__(self, base_dir=None, split='train',tv_type='itv', num=None, transform=None, register_method=None):
        self._base_dir = base_dir
        self.transform = transform
        self.register_method = register_method
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/../dataset/utswdataset_gp/train_'+tv_type+'_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'val':
            with open(self._base_dir+'/../dataset/utswdataset_gp/val_'+tv_type+'_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../dataset/utswdataset_gp/test_'+tv_type+'_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'all':
            with open(self._base_dir+'/../dataset/HengRui_LiTS/all_mr_'+tv_type+'_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'brain_all':
            with open(self._base_dir+'/../dataset/dataset_brain_ori/all_mr_tv_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'brain_ctv':
            with open(self._base_dir+'/../dataset/dataset_brain_ori/all_mr_ctv_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'brain_ptv':
            with open(self._base_dir+'/../dataset/dataset_brain_ori/all_mr_ptv_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'brain_gtv':
            with open(self._base_dir+'/../dataset/dataset_brain_ori/all_mr_gtv_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()



        self.curMR_list = [str(item.replace(' \n', '')) for item in self.curMR_list]
        if num is not None:
            self.curMR_list = self.curMR_list[:num]
        print("total {} samples".format(len(self.curMR_list)))
        if split == 'all':
            self.abs_path = os.path.join(base_dir, "../dataset/HengRui_LiTS")
        elif "brain" in split:
            self.abs_path = os.path.join(base_dir, "../dataset/dataset_brain_ori")
        else:
            self.abs_path = os.path.join(base_dir, "../dataset/utswdataset_gp")

    def __len__(self):
        return len(self.curMR_list)

    def __getitem__(self, idx):
        curMR_name = self.curMR_list[idx] # path of the files: curMR, preMR, preGT, curGT
        ## open files
        curMR_paths = curMR_name.split(" ")
        curMR_path = os.path.join(self.abs_path, curMR_paths[0])
        preMR_path = os.path.join(self.abs_path, curMR_paths[1])
        preGT_path = os.path.join(self.abs_path, curMR_paths[2])
        curGT_path = os.path.join(self.abs_path, curMR_paths[3])
        pre_liver_path = preMR_path.replace("_mr.nii.gz", "_gt_liver.nii.gz")
        cur_liver_path = curMR_path.replace("_mr.nii.gz", "_gt_liver.nii.gz")

        curMR = nib.load(curMR_path).get_fdata()
        preMR = nib.load(preMR_path).get_fdata()
        preGT = nib.load(preGT_path).get_fdata()
        curGT = nib.load(curGT_path).get_fdata()

        curGT_np_est = register_and_transform_ants(preMR_path, curMR_path, preGT_path, transform_type=self.register_method)
        # curGT_np_est = get_cur_tumor_from_registration(pre_liver_path, cur_liver_path, preGT_path, transform_type=self.register_method)

        ## Translation, Rigid Similarity QuickRigid DenseRigid BOLDRigid Affine AffineFast TRSAA Elastic ElasticSyN
        # curGT_np_est = register_and_estimate_curTumor(preMR_path, curMR_path, pre_liver_path, cur_liver_path, preGT_path, transform_type=self.register_method)

        
        caseID = curMR_paths[0].split('/')[1].split('_mr.nii.gz')[0] # like: 70125695/70125695_20240828_mr.nii.gz

        ## normalization
        curMR = (curMR - curMR.min()) / (curMR.max() - curMR.min() + 1e-10)
        preMR = (preMR - preMR.min()) / (preMR.max() - preMR.min() + 1e-10)
        # curMR = (curMR - np.mean(curMR)) / np.std(curMR)

        sample = {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID, "curGT_est": curGT_np_est}
        if self.transform:
            sample = self.transform(sample) #?? why lack of curGT_est

        return sample, curGT_np_est



class UTSWLiver_Reg_bk(Dataset):
    """ Liver Dataset """
    def __init__(self, base_dir=None, split='train',tv_type='itv', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/../dataset/utswdataset_gp/train_'+tv_type+'_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'val':
            with open(self._base_dir+'/../dataset/utswdataset_gp/val_'+tv_type+'_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../dataset/utswdataset_gp/test_'+tv_type+'_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'all':
            with open(self._base_dir+'/../dataset/HengRui_LiTS/all_mr_'+tv_type+'_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'brain_all':
            with open(self._base_dir+'/../dataset/dataset_brain_ori/all_mr_tv_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'brain_ctv':
            with open(self._base_dir+'/../dataset/dataset_brain_ori/all_mr_ctv_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'brain_ptv':
            with open(self._base_dir+'/../dataset/dataset_brain_ori/all_mr_ptv_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()
        elif split == 'brain_gtv':
            with open(self._base_dir+'/../dataset/dataset_brain_ori/all_mr_gtv_data_rel_path.txt', 'r') as f:
                self.curMR_list = f.readlines()



        self.curMR_list = [str(item.replace(' \n', '')) for item in self.curMR_list]
        if num is not None:
            self.curMR_list = self.curMR_list[:num]
        print("total {} samples".format(len(self.curMR_list)))
        if split == 'all':
            self.abs_path = os.path.join(base_dir, "../dataset/HengRui_LiTS")
        elif "brain" in split:
            self.abs_path = os.path.join(base_dir, "../dataset/dataset_brain_ori")
        else:
            self.abs_path = os.path.join(base_dir, "../dataset/utswdataset_gp")

    def __len__(self):
        return len(self.curMR_list)

    def __getitem__(self, idx):
        curMR_name = self.curMR_list[idx] # path of the files: curMR, preMR, preGT, curGT
        ## open files
        curMR_paths = curMR_name.split(" ")
        curMR_path = os.path.join(self.abs_path, curMR_paths[0])
        preMR_path = os.path.join(self.abs_path, curMR_paths[1])
        preGT_path = os.path.join(self.abs_path, curMR_paths[2])
        curGT_path = os.path.join(self.abs_path, curMR_paths[3])

        curMR = nib.load(curMR_path).get_fdata()
        preMR = nib.load(preMR_path).get_fdata()
        preGT = nib.load(preGT_path).get_fdata()
        curGT = nib.load(curGT_path).get_fdata()

        curGT_np_est = register_and_transform_ants(preMR_path, curMR_path, preGT_path)

        
        caseID = curMR_paths[0].split('/')[1].split('_mr.nii.gz')[0] # like: 70125695/70125695_20240828_mr.nii.gz

        ## normalization
        curMR = (curMR - curMR.min()) / (curMR.max() - curMR.min() + 1e-10)
        preMR = (preMR - preMR.min()) / (preMR.max() - preMR.min() + 1e-10)
        # curMR = (curMR - np.mean(curMR)) / np.std(curMR)

        sample = {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID, "curGT_est": curGT_np_est}
        if self.transform:
            sample = self.transform(sample) #?? why lack of curGT_est

        return sample, curGT_np_est

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']

        # pad the sample if necessary
        if curGT.shape[0] <= self.output_size[0] or curGT.shape[1] <= self.output_size[1] or curGT.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - curGT.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - curGT.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - curGT.shape[2]) // 2 + 3, 0)
            curMR = np.pad(curMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            preMR = np.pad(preMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            preGT = np.pad(preGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = curMR.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        curGT = curGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        curMR = curMR[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        preGT = preGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        preMR = preMR[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID}


class RandomCrop(object):
    """
    Crop randomly the curMR in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']

        # pad the sample if necessary
        if curGT.shape[0] <= self.output_size[0] or curGT.shape[1] <= self.output_size[1] or curGT.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - curGT.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - curGT.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - curGT.shape[2]) // 2 + 3, 0)
            curMR = np.pad(curMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            curGT = np.pad(curGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            preMR = np.pad(preMR, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            preGT = np.pad(preGT, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

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
        preGT = preGT[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        preMR = preMR[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']
        k = np.random.randint(0, 4)
        curMR = np.rot90(curMR, k)
        curGT = np.rot90(curGT, k)
        axis = np.random.randint(0, 2)
        curMR = np.flip(curMR, axis=axis).copy()
        curGT = np.flip(curGT, axis=axis).copy()

        preMR = np.rot90(preMR, k)
        preGT = np.rot90(preGT, k)
        axis = np.random.randint(0, 2)
        preMR = np.flip(preMR, axis=axis).copy()
        preGT = np.flip(preGT, axis=axis).copy()


        return {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']
        noise = np.clip(self.sigma * np.random.randn(curMR.shape[0], curMR.shape[1], curMR.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        curMR = curMR + noise
        preMR = preMR + noise
        return {'curMR': curMR, 'preMR': preMR, 'preGT': preGT, "curGT": curGT, "caseID": caseID}


class CreateOnehotcurGT(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        curMR, preMR, preGT, curGT, caseID = sample['curMR'], sample['preMR'], sample['preGT'], sample['curGT'], sample['caseID']
        onehot_curGT = np.zeros((self.num_classes, curGT.shape[0], curGT.shape[1], curGT.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_curGT[i, :, :, :] = (curGT == i).astype(np.float32)
        return {'curMR': curMR, 'curGT': curGT,'onehot_curGT':onehot_curGT}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        curMR = sample['curMR']
        curMR = curMR.reshape(1, curMR.shape[0], curMR.shape[1], curMR.shape[2]).astype(np.float32)
        preMR = sample['preMR']
        preMR = preMR.reshape(1, preMR.shape[0], preMR.shape[1], preMR.shape[2]).astype(np.float32)
        preGT = sample['preGT']
        preGT = preGT.reshape(1, preGT.shape[0], preGT.shape[1], preGT.shape[2]).astype(np.float32)


        if 'onehot_curGT' in sample:
            return {'curMR': torch.from_numpy(curMR), 'curGT': torch.from_numpy(sample['curGT']).long(),
                    'onehot_curGT': torch.from_numpy(sample['onehot_curGT']).long(),
                    'caseID': sample['caseID'], 'curGT_est':sample['curGT_est']}
        else:
            return {'curMR': torch.from_numpy(curMR), 'preMR': torch.from_numpy(preMR), 
                     'preGT': torch.from_numpy(preGT).long(), 'curGT': torch.from_numpy(sample['curGT']).long(),
                     'caseID': sample['caseID'], 'curGT_est':sample['curGT_est']}


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