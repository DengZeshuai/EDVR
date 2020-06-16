import glob
import imageio
import os.path as osp
import torch
import torch.utils.data as data
import random
import numpy as np
import data.util as util


class MM522Dataset(data.Dataset):
    def __init__(self, args, name='MM522', train=True):
        super(MM522Dataset, self).__init__()
        self.args = args
        self.scale = args['scale']
        self.train = train
        self.n_frames = args['N_frames']
        self.patch_size = args['LQ_size']
        self.lr_paths, self.gt_paths = [], []
        self.videos_name, self.idx_maxidx = [], []
        self.set_paths(osp.join(args['dataroot_GT'], name))

    def __getitem__(self, index):
        lrs, gt, video_name, filename = self.read_data(index)
        if self.train:
            lrs, gt = self.random_crop(lrs, gt, self.patch_size, self.scale)
            lrs.append(gt)
            rlt = util.augment(lrs)
            # rlt = [util.rgb2ycbcr(img, only_y=True) for img in rlt]
            lrs, gt = rlt[:-1], rlt[-1]
        else:
            lrs.append(gt)
            # rlt = [util.rgb2ycbcr(img, only_y=True) for img in lrs]
            lrs, gt = rlt[:-1], rlt[-1]
        lrs, gt = self.to_tensor(lrs, gt)
        # return lrs, gt, video_name, filename
        key = '{}_{}'.format(video_name,filename)
        return {'LQs': lrs, 'GT': gt, 'key': key}
    
    def __len__(self):
        return len(self.gt_paths)

    def set_paths(self, data_dir):
        if self.train:
            folders_lr = osp.join(data_dir, 
                            'video_training_dataset', '*', '*', 'blur4')
            folders_gt = osp.join(data_dir, 
                            'video_training_dataset', '*', '*', 'truth')
        else:
            folders_lr = osp.join(data_dir, 'video_eval', '*', '*', 'blur4')
            folders_gt = osp.join(data_dir, 'video_eval', '*', '*', 'truth')
        subfolders_lr = sorted(glob.glob(folders_lr))
        subfolders_gt = sorted(glob.glob(folders_gt))
        for subfolder_lr, subfolder_gt in zip(subfolders_lr, subfolders_gt):
            video_name = osp.basename(osp.dirname(subfolder_gt))
            lr_paths = sorted(glob.glob(subfolder_lr + '/*'))
            gt_paths = sorted(glob.glob(subfolder_gt + '/*'))
            max_idx = len(gt_paths)
            assert max_idx == len(lr_paths), \
                'Different number of images in lr and gt folders'
            self.lr_paths.extend(lr_paths)
            self.gt_paths.extend(gt_paths)
            self.videos_name.extend([video_name] * max_idx)
            for i in range(max_idx):
                self.idx_maxidx.append('{}/{}'.format(i, max_idx))

    def read_data(self, index):
        video_name = self.videos_name[index]
        idx, max_idx = self.idx_maxidx[index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        indices = util.index_generation(idx, max_idx, self.n_frames)
        
        lr_path = osp.dirname(self.lr_paths[index])
        filename = osp.basename(self.lr_paths[index])
        lrs = []
        for i in indices:
            lr = util.read_img(None, osp.join(lr_path, '{:03d}.png'.format(i)))
            lrs.append(lr)
        gt = util.read_img(None, self.gt_paths[index])
        return lrs, gt, video_name, filename
    
    def to_tensor(self, lrs, gt):
        """lrs, gt """
        lrs = np.stack(lrs, axis=0)
        # HWC to CHW, numpy to tensor
        lrs = torch.from_numpy(
            np.ascontiguousarray(lrs.transpose((0, 3, 1, 2)))).float()
        gt = torch.from_numpy(
            np.ascontiguousarray(gt.transpose((2, 0, 1)))).float()
    
        return lrs, gt
    
    def random_crop(self, lrs, gt, crop_size=32, scale=4):
        H, W, C = lrs[0].shape  # lr size
        GT_size = crop_size * scale
        # randomly crop
        rnd_h = random.randint(0, max(0, H - crop_size))
        rnd_w = random.randint(0, max(0, W - crop_size))
        rnd_h_gt, rnd_w_gt = int(rnd_h * scale), int(rnd_w * scale)
        
        lrs = [lr[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :] for lr in lrs]
        gt = gt[rnd_h_gt:rnd_h_gt + GT_size, rnd_w_gt:rnd_w_gt + GT_size, :]
        return lrs, gt

    def augment(self, imgs, hflip=True, rot=True):
        """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        return [_augment(img) for img in imgs]