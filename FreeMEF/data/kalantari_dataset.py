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


