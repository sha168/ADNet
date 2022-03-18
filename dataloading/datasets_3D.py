import torch
from torch.utils.data import Dataset
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
from .dataset_specifics import *
from monai.transforms.spatial.dictionary import Rand3DElasticd
from collections import defaultdict


class TestDataset(Dataset):

    def __init__(self, args):

        # reading the paths
        if args.dataset == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'cmr_MR_normalized/image*'))
        elif args.dataset == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/image*'))
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        # remove test fold!
        self.FOLD = get_folds(args.dataset)
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args.fold]]

        # split into support/query
        self.support_dir = self.image_dirs[-1]
        self.image_dirs = self.image_dirs[:-1]  # remove support
        self.label = None

        # evaluation protocol
        self.EP1 = args.EP1

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):

        img_path = self.image_dirs[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img - img.mean()) / img.std()
        img = np.stack(1 * [img], axis=0)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == self.label)

        sample = {'id': img_path}

        # Evaluation protocol 1.
        if self.EP1:
            idx = lbl.sum(axis=(1, 2)) > 0
            sample['image'] = torch.from_numpy(img[idx])
            sample['label'] = torch.from_numpy(lbl[idx])

        # Evaluation protocol 2 (default).
        else:
            sample['image'] = torch.from_numpy(img)
            sample['label'] = torch.from_numpy(lbl)

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')

        img_path = self.support_dir
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img - img.mean()) / img.std()
        img = np.stack(1 * [img], axis=0)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == label)

        sample = {}
        if all_slices:

            sample['image'] = torch.from_numpy(img)[None]
            sample['label'] = torch.from_numpy(lbl)[None]

            # target = np.where(lbl.sum(axis=(-2, -1)) > 0)[0]
            # mask = np.zeros(lbl.shape) == 1
            # mask[target.astype('float').mean().astype('int')] = True
            # sample['label'] = torch.from_numpy((mask*1)*lbl)[None]

        else:
            # select N labeled slices
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            idx = lbl.sum(axis=(1, 2)) > 0
            idx_ = self.get_support_index(N, idx.sum())

            sample['image'] = torch.from_numpy(img[:, idx][:, idx_])[None]
            sample['label'] = torch.from_numpy(lbl[idx][idx_])[None]

        return sample


class TrainDataset(Dataset):

    def __init__(self, args):
        self.n_shot = args.n_shot
        self.n_way = args.n_way
        self.n_query = args.n_query
        self.n_sv = args.n_sv
        self.max_iter = args.max_iterations
        self.min_size = args.min_size
        self.max_slices = args.max_slices

        # reading the paths (leaving the reading of images into memory to __getitem__)
        if args.dataset == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'cmr_MR_normalized/image*'))
        elif args.dataset == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/image*'))
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, 'supervoxels_' + str(args.n_sv), 'super*'))
        self.sprvxl_dirs = sorted(self.sprvxl_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        # remove test fold!
        self.FOLD = get_folds(args.dataset)
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args.fold]]
        self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if idx not in self.FOLD[args.fold]]
        self.N = len(self.image_dirs)

        # read images
        self.images = {}
        self.sprvxls = {}
        self.valid_spr_slices = {}
        for image_dir, sprvxl_dir in zip(self.image_dirs, self.sprvxl_dirs):
            img = sitk.ReadImage(image_dir)
            self.res = img.GetSpacing()
            img = sitk.GetArrayFromImage(img)
            self.images[image_dir] = torch.from_numpy(img)
            spr = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir)))
            self.sprvxls[sprvxl_dir] = spr

            unique = list(torch.unique(spr))
            unique.remove(0)
            self.valid_spr_slices[image_dir] = []
            for val in unique:
                spr_val = (spr == val)

                n_slices = min(spr_val.shape[0], self.max_slices)
                sample_list = []
                for r in range(spr_val.shape[0] - (n_slices - 1)):
                    sample_idx = torch.arange(r, r + n_slices).tolist()
                    candidate = spr_val[sample_idx]
                    if candidate.sum() > self.min_size:
                        sample_list.append(sample_idx)
                if len(sample_list) > 0:
                    self.valid_spr_slices[image_dir].append((val, sample_list))

        # set transformation details
        rad = 5 * (np.pi / 180)
        self.rand_3d_elastic = Rand3DElasticd(
            keys=("img", "seg"),
            mode=("bilinear", "nearest"),
            sigma_range=(5, 5),
            magnitude_range=(0, 0),
            prob=1.0,  # because probability controlled by this class
            rotate_range=(rad, rad, rad),
            shear_range=(rad, rad, rad),
            translate_range=(5, 5, 1),
            scale_range=((-0.1, 0.2), (-0.1, 0.2), (-0.1, 0.2)),
            as_tensor_output=True,
            device='cpu')

    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = torch.rand(1) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * torch.pow(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def __getitem__(self, idx):

        # sample patient idx
        pat_idx = random.choice(range(len(self.image_dirs)))

        # get image/supervoxel volume from dictionary
        img = self.images[self.image_dirs[pat_idx]]
        sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]]

        # normalize
        img = (img - img.mean()) / img.std()

        # sample supervoxel
        valid = self.valid_spr_slices[self.image_dirs[pat_idx]]
        cls_idx, candidates = valid[random.randint(0, len(valid) - 1)]

        sprvxl = 1 * (sprvxl == cls_idx)

        sup_lbl = torch.clone(sprvxl)
        qry_lbl = torch.clone(sprvxl)

        sup_img = torch.clone(img)
        qry_img = torch.clone(img)

        # gamma transform
        if np.random.random(1) > 0.5:
            qry_img = self.gamma_tansform(qry_img)
        else:
            sup_img = self.gamma_tansform(sup_img)

        # geom transform
        if np.random.random(1) > 0.5:
            res = self.rand_3d_elastic({"img": qry_img.permute(1, 2, 0),
                                        "seg": qry_lbl.permute(1, 2, 0)})

            qry_img = res["img"].permute(2, 0, 1)
            qry_lbl = res["seg"].permute(2, 0, 1)

            # support not tformed
            constant_s = random.randint(0, len(candidates) - 1)
            idx_s = candidates[constant_s]

            k = 50
            constant_q = constant_s + random.randint(-min(constant_s, k), min(len(candidates) - constant_s - 1, k))
            idx_q = candidates[constant_q]

        else:
            res = self.rand_3d_elastic({"img": sup_img.permute(1, 2, 0),
                                        "seg": sup_lbl.permute(1, 2, 0)})

            sup_img_ = res["img"].permute(2, 0, 1)
            sup_lbl_ = res["seg"].permute(2, 0, 1)

            constant_q = random.randint(0, len(candidates) - 1)
            idx_q = candidates[constant_q]

            k = 50
            constant_s = constant_q + random.randint(-min(constant_q, k), min(len(candidates) - constant_q - 1, k))
            idx_s = candidates[constant_s]
            if sup_lbl_[idx_s].sum() > 0:
                sup_img = sup_img_
                sup_lbl = sup_lbl_

        sup_lbl = sup_lbl[idx_s]
        qry_lbl = qry_lbl[idx_q]

        sup_img = sup_img[idx_s]
        qry_img = qry_img[idx_q]

        b = 215
        k = 0
        horizontal_s, vertical_s = sample_xy(sup_lbl, k=k, b=b)
        horizontal_q, vertical_q = sample_xy(qry_lbl, k=k, b=b)

        sup_img = sup_img[:, horizontal_s:horizontal_s + b, vertical_s:vertical_s + b]
        sup_lbl = sup_lbl[:, horizontal_s:horizontal_s + b, vertical_s:vertical_s + b]
        qry_img = qry_img[:, horizontal_q:horizontal_q + b, vertical_q:vertical_q + b]
        qry_lbl = qry_lbl[:, horizontal_q:horizontal_q + b, vertical_q:vertical_q + b]

        sample = {'support_images': torch.stack(1 * [sup_img], dim=0),
                  'support_fg_labels': sup_lbl[None],
                  'query_images': torch.stack(1 * [qry_img], dim=0),
                  'query_labels': qry_lbl}

        return sample
