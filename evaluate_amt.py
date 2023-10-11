#from AMT import load_amts
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import importlib
from AMT.utils.utils import read, img2tensor
from os import path as osp
from pathlib import Path

#from AMT.networks.blocks.ifrnet import 

def segment_img(img1, img2):
    '''
    return a array of segment images
    '''
    H = img1.size(2)  # Assuming the images are of shape (batch_size, channels, height, width)

    # Calculate heights of each segment
    mid = H // 2
    h1 = int(mid * 1/9)
    h2 = int(mid * 2/9)
    h3 = int(mid * 1/3)
    h4 = int(mid * 1/3)
    
    img1_0_10 = img1[:, :, mid-h1:mid+h1, :]
    img2_0_10 = img2[:, :, mid-h1:mid+h1, :]
    img1_10_30 = torch.cat((img1[:, :, mid-h1-h2:mid-h1, :], img1[:, :, mid+h1:mid+h1+h2, :]), dim=2)
    img2_10_30 = torch.cat((img2[:, :, mid-h1-h2:mid-h1, :], img2[:, :, mid+h1:mid+h1+h2, :]), dim=2)
    img1_30_60 = torch.cat((img1[:, :, mid-h1-h2-h3:mid-h1-h2, :], img1[:, :, mid+h1+h2:mid+h1+h2+h3, :]), dim=2)
    img2_30_60 = torch.cat((img2[:, :, mid-h1-h2-h3:mid-h1-h2, :], img2[:, :, mid+h1+h2:mid+h1+h2+h3, :]), dim=2)
    img1_60_90 = torch.cat((img1[:, :, mid-h1-h2-h3-h4:mid-h1-h2-h3, :], img1[:, :, mid+h1+h2+h3:mid+h1+h2+h3+h4, :]), dim=2)
    img2_60_90 = torch.cat((img2[:, :, mid-h1-h2-h3-h4:mid-h1-h2-h3, :], img2[:, :, mid+h1+h2+h3:mid+h1+h2+h3+h4, :]), dim=2)
  
    seg_img1 = [img1_0_10, img1_10_30, img1_30_60, img1_60_90]
    seg_img2 = [img2_0_10, img2_10_30, img2_30_60, img2_60_90]
  
    return seg_img1, seg_img2

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1 / mse)

def seg_psnr(img1, img2):
    seg_img1, seg_img2 = segment_img(img1, img2)
    psnrs = [calculate_psnr(seg1, seg2) for seg1, seg2 in zip(seg_img1, seg_img2)]
    return psnrs

@torch.no_grad()
def validate_360vds(root='.', ckpt_path=None,DEVICE_IDS = [2]):
    device = torch.device(f'cuda:{DEVICE_IDS[0]}' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path)
    module_path = 'AMT.networks.AMT-S' 
    cls_name = 'Model'
    params = {
        'corr_radius': 3,
        'corr_lvls': 4,
        'num_flows': 3   
    }
    modelclass = getattr(importlib.import_module(module_path), cls_name)
    model = modelclass(corr_radius=params['corr_radius'], corr_lvls=params['corr_lvls'], num_flows=params['num_flows'])
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    total_psnr = 0
    count = 0

    with open(osp.join(root, 'tri_testlist.txt'), 'r') as fr:
        file_list = fr.readlines()
    
    segment_psnrs = {
        "Segment 1 (0-10%)": [],
        "Segment 2 (10-30%)": [],
        "Segment 3 (30-60%)": [],
        "Segment 4 (60-90%)": []
    }

    pbar = tqdm(file_list, total=len(file_list))
    for name in pbar:
        name = str(name).strip()
        if len(name) <= 1:
            continue
        dir_path = osp.join(root, 'sequences', name)
        I0 = img2tensor(read(osp.join(dir_path, 'im1.png'))).to(device)
        I1 = img2tensor(read(osp.join(dir_path, 'im2.png'))).to(device)
        I2 = img2tensor(read(osp.join(dir_path, 'im3.png'))).to(device)
        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)

        I1_pred = model(I0, I2, embt, scale_factor=1.0, eval=True)['imgt_pred']
        
        seg_I1, seg_I1_pred = segment_img(I1, I1_pred)

        current_psnr = calculate_psnr(I1, I1_pred).item()

        segment_keys = ["Segment 1 (0-10%)", "Segment 2 (10-30%)", "Segment 3 (30-60%)", "Segment 4 (60-90%)"]


        for i, (s_I1, s_I1_pred) in enumerate(zip(seg_I1, seg_I1_pred)):
            segment_psnr = calculate_psnr(s_I1, s_I1_pred)
            segment_name = segment_keys[i]  # 使用 segment_keys 列表来获取正确的 segment_name
            segment_psnrs[segment_name].append(segment_psnr.item())

        total_psnr += current_psnr
        count += 1

    average_total_psnr = total_psnr / count if count != 0 else 0


    results = {
        'Segment 1 (0-10%)': segment_psnrs['Segment 1 (0-10%)'],
        'Segment 2 (10-30%)': segment_psnrs['Segment 2 (10-30%)'],
        'Segment 3 (30-60%)': segment_psnrs['Segment 3 (30-60%)'],
        'Segment 4 (60-90%)': segment_psnrs['Segment 4 (60-90%)'],
        'metric': ['PSNR'] * len(segment_psnrs['Segment 1 (0-10%)']),
        'mode': ['Mode'] * len(segment_psnrs['Segment 1 (0-10%)'])
    }
    average_segment_psnrs = {}



    data = pd.DataFrame(results)
    print(data)

    for segment_name, psnr_values in segment_psnrs.items():
    
        average_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0 
   
        average_segment_psnrs[segment_name] = average_psnr


    for segment_name, average_psnr in average_segment_psnrs.items():
        print(f"Average PSNR for {segment_name}: {average_psnr}")

    print(f"Average Total PSNR: {average_total_psnr}")

    filepath = f"/home/public/test/final.csv" 
    
    return {
    'psnrs': average_total_psnr, 
    'data': data, 
    'filepath': filepath 
    }

    

if __name__ == '__main__':
    
    import sys
    
    #raft = load_amts.load(path_root="RAFT")
    validate_360vds(root='/home/public/wym/vimeostyle_360VDS', ckpt_path='/home/u222080208/CODE/A/AMT/pretrained_weight/amt-s.pth')