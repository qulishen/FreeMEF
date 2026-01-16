import os
import re
from glob import glob
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from FreeMEF.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from FreeMEF.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from FreeMEF.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from basicsr.utils.registry import DATASET_REGISTRY
import random
import numpy as np
import torch
import cv2

class DeblurPairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(DeblurPairedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class Kalantari_Dataset(data.Dataset):
    """单帧HDR重建数据集。

    目录结构要求：
        root/
            Label/1.JPG ... n.JPG         # GT
            1/*.JPG ... 1/*.jpg           # 多张不同曝光的LDR
            2/*.JPG ...                   # 以此类推

    Args:
        opt (dict):
            dataroot_hdr (str): SICE根目录，包含各子文件夹和Label。
            dataroot_gt (str, optional): Label路径，若不填则默认 root/Label。
            io_backend (dict): IO后端设置。
            gt_size (int): 训练裁剪尺寸。
            geometric_augs (bool): 是否做几何增强。
            scale (int): 缩放倍率。
            phase (str): train 或 val。
            mean/std: 归一化参数。
    """

    def __init__(self, opt):
        super(Kalantari_Dataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.hdr_root = opt['dataroot_hdr']
        self.label_root = opt.get('dataroot_gt', os.path.join(self.hdr_root, 'Label'))

        subfolders = [
            d for d in sorted(os.listdir(self.hdr_root))
            if os.path.isdir(os.path.join(self.hdr_root, d)) and d.lower() != 'label'
        ]

        self.samples = []
        for folder in subfolders:
            folder_path = os.path.join(self.hdr_root, folder)
            exposure_paths = sorted(
                glob(os.path.join(folder_path, '*.JPG')) +
                glob(os.path.join(folder_path, '*.jpg')) +
                glob(os.path.join(folder_path, '*.png'))
            )
            if len(exposure_paths) == 0:
                continue

            gt_candidates = [
                os.path.join(self.label_root, f'{folder}.JPG'),
                os.path.join(self.label_root, f'{folder}.jpg'),
                os.path.join(self.label_root, f'{folder}.png'),
            ]
            gt_path = next((p for p in gt_candidates if os.path.isfile(p)), None)
            if gt_path is None:
                raise FileNotFoundError(f'未找到 {folder} 对应的GT（Label/{folder}.JPG 或 .jpg/.png）')

            self.samples.append({'exposures': exposure_paths, 'gt_path': gt_path})

        if len(self.samples) == 0:
            raise RuntimeError(f'在 {self.hdr_root} 下未找到可用的HDR样本。')

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        sample = self.samples[index % len(self.samples)]

        gt_path = sample['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception(f"gt path {gt_path} not working")

        # 选择基准帧：按文件序号排序后取中间帧作为主LDR
        def _extract_index(p):
            name = os.path.basename(p)
            nums = re.findall(r'\d+', name)
            return int(nums[-1]) if nums else 0

        exposures_sorted = sorted(sample['exposures'], key=_extract_index)
        base_idx = (len(exposures_sorted) - 1) // 2 
        lq_path = exposures_sorted[base_idx]
        gt_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        eps = 1e-6

        # 只按需加载，避免一次性把所有other帧读入内存导致显存/内存暴涨
        try:
            base_bytes = self.file_client.get(lq_path, 'hdr')
            lq_main = imfrombytes(base_bytes, float32=True)
        except Exception:
            raise Exception(f"ldr path {lq_path} not working")

        if lq_main is None:
            raise RuntimeError(f"未找到主LDR帧: {lq_path}")

        exposures_no_base = [p for p in exposures_sorted if p != lq_path]
        # 随机三种选帧策略：others数量可以为1、2或4
        target_others = random.choice([1, 2, 4])

        def _fetch_frame(idx):
            # idx >=0: 从最小开始；idx <0: 从最大开始
            real_idx = idx if idx >= 0 else len(exposures_no_base) + idx
            if 0 <= real_idx < len(exposures_no_base):
                sel_path = exposures_no_base[real_idx]
                try:
                    other_bytes = self.file_client.get(sel_path, 'hdr')
                    return imfrombytes(other_bytes, float32=True)
                except Exception:
                    raise Exception(f"ldr path {sel_path} not working")
            return np.zeros_like(lq_main)

        def _apply_brightness(x, factor):
            # 简单乘法调整亮度并裁剪
            return np.clip(x.astype(np.float32) * factor, 0.0, 1.0)

        lq_sequence = []
        base_pos = 0  # lq_sequence中base的位置

        if target_others == 1:
            # 只取最小曝光帧，base随机提亮
            base_pos = 0
            lq_main = _apply_brightness(lq_main, random.uniform(1.5, 2.0))
            lq_sequence = [lq_main, _fetch_frame(0)]
        elif target_others == 2:
            # 与原策略一致：最小曝光和最大曝光
            base_pos = 0
            lq_sequence = [lq_main, _fetch_frame(0), _fetch_frame(-1)]
        else:
            # 取最小、最大曝光各一帧，增加亮度增强/降低的版本
            frame_min = _fetch_frame(0)
            frame_max = _fetch_frame(-1)
            darker_min = _apply_brightness(frame_min, random.uniform(0.5, 0.8))
            brighter_max = _apply_brightness(frame_max, random.uniform(1.2, 1.6))
            base_pos = 2  # 放在中间
            lq_sequence = [darker_min, frame_min, lq_main, frame_max, brighter_max]

        # target_others==1/2 时补零，保证 others 总长度一致（4）
        pad_needed = 3 if target_others == 1 else (2 if target_others == 2 else 0)
        if pad_needed > 0:
            lq_sequence += [np.zeros_like(lq_main) for _ in range(pad_needed)]

        lq_others = [f for i, f in enumerate(lq_sequence) if i != base_pos]
        others_count = target_others
        
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            # 统一padding到最小可裁剪尺寸
            h, w, _ = lq_main.shape
            h_pad = max(0, gt_size - h)
            w_pad = max(0, gt_size - w)
            if h_pad > 0 or w_pad > 0:
                pad_fn = lambda x: cv2.copyMakeBorder(x, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
                img_gt = pad_fn(img_gt)
                lq_main = pad_fn(lq_main)
                lq_others = [pad_fn(img) for img in lq_others]

            # 保持所有帧同一裁剪
            lq_list = lq_sequence
            img_gt, lq_list = paired_random_crop(img_gt, lq_list, gt_size, scale, gt_path)
            if not isinstance(lq_list, list):
                lq_list = [lq_list]
            # if self.geometric_augs:
            #     aug_list = random_augmentation(*([img_gt] + lq_list))
            #     img_gt, lq_list = aug_list[0], aug_list[1:]
        else:
            lq_list = lq_sequence

        tensors = img2tensor([img_gt] + lq_list, bgr2rgb=True, float32=True)
        img_gt = tensors[0]
        lq_tensors = tensors[1:]

        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            for t in lq_tensors:
                normalize(t, self.mean, self.std, inplace=True)

        img_lq = lq_tensors[base_pos]
        others_list = [t for i, t in enumerate(lq_tensors) if i != base_pos]
        if len(others_list) == 0:
            others = torch.zeros((0, img_lq.shape[0], img_lq.shape[1], img_lq.shape[2]), dtype=img_lq.dtype)
        else:
            others = torch.stack(others_list, dim=0)  # (K,C,H,W)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'others': others,
            'others_count': others_count
        }

    def __len__(self):
        return len(self.samples)



class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type  = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None        

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)


            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value])/255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:            
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test/255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                            bgr2rgb=False,
                            float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)
            
            # flip, rotation            
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
