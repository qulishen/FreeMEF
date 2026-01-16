## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

import random
import cv2
from natsort import natsorted
from glob import glob
from FreeMEF.archs.HDR_S1_arch import HDR_S1
from skimage import img_as_ubyte
from basicsr.utils import imwrite as save_img
from pdb import set_trace as stx
from FreeMEF.archs.FreeMEF_arch import FreeMEF
parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')
# 
parser.add_argument('--input_dir', default='FreeMEF/datasets/SICE/input_resize_2frame', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='FreeMEF/experiments/train_FreeMEF_on_Kalantari/models/net_g_last.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='Kalantari', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--tile', default=0, type=int, help='Tile size for sliding-window inference. 0 disables tiling.')
parser.add_argument('--tile_stride', default=0, type=int, help='Stride for sliding-window inference. 0 defaults to tile size.')
parser.add_argument('--opt', default='FreeMEF/options/FreeMEF.yml', type=str)
parser.add_argument('--arch', default='FreeMEF', type=str)
args = parser.parse_args()

####### Load yaml #######
yaml_file = args.opt
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
arch_map = {
    'FreeMEF': FreeMEF,
}
if args.arch not in arch_map:
    raise ValueError(f"Unknown architecture: {args.arch}")
model_restoration = arch_map[args.arch](**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


def tile_inference(model, inp, others, others_count, factor, tile_size=512, tile_stride=512):
    """Sliding-window inference to节省显存; 简单双线性融合避免边界缝."""
    if tile_size <= 0:
        return model(inp, others, others_count)
    if tile_stride <= 0:
        tile_stride = tile_size

    _, _, H, W = inp.shape
    device = inp.device
    out = torch.zeros((1, 3, H, W), device=device)
    weight = torch.zeros_like(out)

    for y0 in range(0, H, tile_stride):
        y1 = min(y0 + tile_size, H)
        for x0 in range(0, W, tile_stride):
            x1 = min(x0 + tile_size, W)

            in_patch = inp[:, :, y0:y1, x0:x1]
            oth_patch = others[:, :, :, y0:y1, x0:x1]

            hp, wp = in_patch.shape[2], in_patch.shape[3]
            Hp = ((hp + factor) // factor) * factor
            Wp = ((wp + factor) // factor) * factor
            padhp = Hp - hp
            padwp = Wp - wp

            in_patch = F.pad(in_patch, (0, padwp, 0, padhp), 'reflect')
            oth_flat = oth_patch.view(-1, oth_patch.shape[2], oth_patch.shape[3], oth_patch.shape[4])
            oth_flat = F.pad(oth_flat, (0, padwp, 0, padhp), 'reflect')
            oth_patch = oth_flat.view(1, others.shape[1], others.shape[2], Hp, Wp)

            with torch.no_grad():
                out_patch = model(in_patch, oth_patch, others_count)
            out_patch = out_patch[:, :, :hp, :wp]

            out[:, :, y0:y1, x0:x1] += out_patch
            weight[:, :, y0:y1, x0:x1] += 1

    return out / weight.clamp(min=1e-8)


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir)
os.makedirs(result_dir, exist_ok=True)

inp_root = args.input_dir
# 每个子文件夹视作一个序列，按帧数固定选择主辅帧
seq_dirs = [d for d in natsorted(glob(os.path.join(inp_root, '*'))) if os.path.isdir(d)]

with torch.no_grad():
    for seq_dir in tqdm(seq_dirs):
        frame_paths = natsorted(glob(os.path.join(seq_dir, '*.*')))
        num_frames = len(frame_paths)
        if num_frames < 2:
            continue

        # 按帧数固定选择主帧和辅帧
        base_frame = None
        selected_others = []
        if num_frames == 2:
            base_frame = frame_paths[-1]
            selected_others = [frame_paths[0]]
        elif num_frames == 3:
            base_frame = frame_paths[1]
            selected_others = [frame_paths[0], frame_paths[-1]]
        elif num_frames == 5:
            base_frame = frame_paths[2]
            # selected_others = [frame_paths[0], frame_paths[1], frame_paths[-2],frame_paths[-1]]
            # selected_others = [frame_paths[-1], frame_paths[-2], frame_paths[1],frame_paths[0]]
            selected_others = [frame_paths[0], frame_paths[-1], frame_paths[1],frame_paths[-2]]

        else:
            # 其他长度时回退为以中间帧为主，其余为辅
            mid = num_frames // 2
            base_frame = frame_paths[mid]
            selected_others = [p for i, p in enumerate(frame_paths) if i != mid]

        if base_frame is None or not selected_others:
            continue

        seq_name = os.path.basename(seq_dir)
        os.makedirs(result_dir, exist_ok=True)

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(cv2.imread(base_frame))/255.
        # img = np.float32(cv2.resize(cv2.imread(base_frame), (1000,1000)))/255.
        
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()
        h_in, w_in = input_.shape[2], input_.shape[3]

        others_list = []
        for of in selected_others:
            oimg = np.float32(cv2.imread(of))/255.
            # 若尺寸不同，先对齐到主帧尺寸
            if oimg.shape[0] != h_in or oimg.shape[1] != w_in:
                oimg = cv2.resize(oimg, (w_in, h_in), interpolation=cv2.INTER_CUBIC)
            oimg = torch.from_numpy(oimg).permute(2,0,1)
            others_list.append(oimg)

        if not others_list:
            continue
        others_ = torch.stack(others_list, dim=0).unsqueeze(0).cuda()  # (1,K,C,H,W)
        others_count = torch.tensor([len(others_list)], device=others_.device)
        print(others_count)
        # 若开启切片，则走滑窗推理；否则保持整图推理
        if args.tile > 0:
            restored = tile_inference(
                model_restoration, input_, others_, others_count, factor,
                tile_size=args.tile, tile_stride=args.tile_stride if args.tile_stride > 0 else args.tile
            )
        else:
            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
            others_flat = others_.view(-1, others_.shape[2], others_.shape[3], others_.shape[4])
            others_flat = F.pad(others_flat, (0,padw,0,padh), 'reflect')
            H_pad, W_pad = others_flat.shape[2], others_flat.shape[3]
            others_ = others_flat.view(1, len(others_list), others_.shape[2], H_pad, W_pad)
            if 'SSM' or 'CNN' in args.arch:
                restored = model_restoration(input_,others_,others_count)
            elif 'AFUNet' in args.arch or 'HDRTransformer' in args.arch or 'SAFNet' in args.arch or 'SCTNet' in args.arch in args.arch:
                others_ = others_.squeeze(0)
                if num_frames == 2:
                    restored = model_restoration(torch.cat([others_[0].unsqueeze(0),input_,input_],dim=1))
                else:
                    restored = model_restoration(torch.cat([others_[0].unsqueeze(0),input_,others_[-1].unsqueeze(0)],dim=1))
            else:
                others_ = others_.squeeze(0)
                if num_frames == 2:
                    restored = model_restoration(torch.cat([others_[0].unsqueeze(0),input_],dim=1))
                elif num_frames == 3:
                    restored = model_restoration(torch.cat([others_[0].unsqueeze(0),input_,others_[-1].unsqueeze(0)],dim=1))
                elif num_frames == 5:
                    restored = model_restoration(torch.cat([others_[0].unsqueeze(0),others_[1].unsqueeze(0),input_,others_[-2].unsqueeze(0),others_[-1].unsqueeze(0)],dim=1))
                else:
                    restored = model_restoration(torch.cat([others_[0].unsqueeze(0),input_,others_[-1].unsqueeze(0)],dim=1))
            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        # 每个子文件夹只输出一张结果图，命名为“子文件夹序号.png”
        save_path = os.path.join(result_dir, f"{seq_name}.png")
        save_img(img_as_ubyte(restored), save_path)
