import os
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from pathlib import Path
from typing import *
from augmentor import FlowAugmentor, SparseFlowAugmentor
import numpy as np
from PIL import Image
from AMT.utils.utils import read

try:
    from RAFT.core.utils.utils import InputPadder
    from utils import ReadData
    from equi_utils import rotate_PIL, rotate_eq, getRandomRotationConfig, flow_rotation, get3DFlow
except:
    from flow360.RAFT.core.utils.utils import InputPadder
    from flow360.utils import ReadData
    from flow360.equi_utils import rotate_PIL, rotate_eq, getRandomRotationConfig, flow_rotation, get3DFlow


def random_resize(img0, imgt, img1, flow, p=0.1):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
    return img0, imgt, img1, flow

def random_crop(img0, imgt, img1, flow, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[x:x+h, y:y+w, :]
    imgt = imgt[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    flow = flow[x:x+h, y:y+w, :]
    return img0, imgt, img1, flow

def random_reverse_channel(img0, imgt, img1, flow, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]
    return img0, imgt, img1, flow

def random_vertical_flip(img0, imgt, img1, flow, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
        flow = flow[::-1]
        flow = np.concatenate((flow[:, :, 0:1], -flow[:, :, 1:2], flow[:, :, 2:3], -flow[:, :, 3:4]), 2)
    return img0, imgt, img1, flow

def random_horizontal_flip(img0, imgt, img1, flow, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
        flow = flow[:, ::-1]
        flow = np.concatenate((-flow[:, :, 0:1], flow[:, :, 1:2], -flow[:, :, 2:3], flow[:, :, 3:4]), 2)
    return img0, imgt, img1, flow

def random_rotate(img0, imgt, img1, flow, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
        flow = flow.transpose((1, 0, 2))
        flow = np.concatenate((flow[:, :, 1:2], flow[:, :, 0:1], flow[:, :, 3:4], flow[:, :, 2:3]), 2)
    return img0, imgt, img1, flow

def random_reverse_time(img0, imgt, img1, flow, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img1
        img1 = img0
        img0 = tmp
        flow = np.concatenate((flow[:, :, 2:4], flow[:, :, 0:2]), 2)
    return img0, imgt, img1, flow


class Flow360Loader(Dataset):
    def __init__(self, 
                 root_path = Path('/data/keshav/flow360/FLOW360'), 
                 mode = 'train', 
                 items = ['frame1','frame2', 'fflow', 'fflow3d'],
                 transform = {'resize' : None, 'rotation': False, 'seed' : 360}) -> None:
        """Create a dataset for flow360, with given transform and mode
        See following root_path format
        root_path/
            ├── train
            │   ├── 001
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ├── 002
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ...
            │   .   
            │   .   
            ├── val
            │   ├── 001
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ├── 002
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ...
            │   .   
            │   .   
            ├── test
            │   ├── 001
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ├── 002
            │   │   ├── bflows
            │   │   ├── fflows
            │   │   └── frames
            │   ...
            │   .   
            │   .  
        Args:
            root_path ([str], optional): [root_path as defined above]. Defaults to Path('/data/keshav/flow360/FLOW360').
            mode (str, optional): ['train', 'test', or 'val']. Defaults to 'train'.
            transform (bool, optional): [dictionary specifying resize and rotation parameter]. Defaults to {'resize' : None, 'rotation': False, 'seed': 360}.
                - resize(tuple(int(height),int(width))): resize the image and  2dflow,3dflow in given dimension
                - rotation(boolean): rotate image and 3dflow in random rotation
                - seed(int): seed to control consistent random rotation, defaults to 360
        """
        super(Flow360Loader, self).__init__()
        assert mode in ['train','val','test'], f"{mode} not in Options ['train','val','test']"
        assert set(items).issubset(set(('frame1','frame2','fflow','bflow','fflow3d','bflow3d'))), f"items : {items} not available, Valid options: ('frame1','frame2','fflow','bflow','fflow3d','bflow3d')"
        self.mode = mode
        self.items = items
        self.transform = transform
        self.datalist = ReadData(root_path=root_path, mode=mode, items = items, resize = self.transform.get('resize'))
        self.augmentor = FlowAugmentor(crop_size=(320,640))
    
    def __len__(self) -> int:
        return len(self.datalist)
    
    def transform_frame(self, frame, rotcon = None)->torch.Tensor:
        if self.transform.get('rotation'):
            frame = ToTensor()(rotate_PIL(frame, rots = rotcon))
        else:
            frame = ToTensor()(frame)
        
        return frame
    
    def transform_3dflow(self, flow, rotcon = None)-> torch.Tensor:
        flow3d = get3DFlow(flow.unsqueeze(0))
        if self.transform.get('rotation'):
            flow3d = flow_rotation(flow3d, rots = rotcon)
        return flow3d[0].permute(2,0,1)
    
    def __getitem__(self, index) -> Any:
        data = self.datalist[index]
        context = {}
        rotcon = None
        
        frame1 = data.frame1
        frame2 = data.frame2
        
        if self.mode == 'train':
            frame1 = np.asarray(frame1)
            frame2 = np.asarray(frame2)
            frame1, frame2, _ = self.augmentor(frame1, frame2, None, spatial = False)
            frame1 = Image.fromarray(frame1)
            frame2 = Image.fromarray(frame2)
        
        # if self.transform.get('rotation'):
        rotcon1 = getRandomRotationConfig(batch = 1)
        rotcon2 = getRandomRotationConfig(batch = 1)
        
        context['pitch'] = rotcon1[0].get('pitch')
        context['yaw'] = rotcon1[0].get('yaw')
        context['roll'] = rotcon1[0].get('roll')
        
        context['pitch_'] = rotcon2[0].get('pitch')
        context['yaw_'] = rotcon2[0].get('yaw')
        context['roll_'] = rotcon2[0].get('roll')
        
        context['rot'] = rotcon1
        context['rot_'] = rotcon2
        
        if 'frame1' in self.items:
            context['frame1'] = self.transform_frame(frame1, rotcon = rotcon)
        
        if 'frame2' in self.items:
            context['frame2'] = self.transform_frame(frame2, rotcon = rotcon)
        
        if 'fflow' in self.items:
            fflow = -data.fflow
            if 'fflow3d' in self.items:
                context['fflow3d'] = self.transform_3dflow(fflow, rotcon = rotcon)
                
            context['fflow'] = fflow.permute(2,0,1)
        
        if 'bflow' in self.items:
            bflow = data.bflow
            if 'bflow3d' in self.items:
                context['bflow3d'] = self.transform_3dflow(bflow, rotcon = rotcon)
            
            context['bflow'] = bflow.permute(2,0,1)
        
        context['valid'] = ((context['fflow'][0].abs() < 1000) & (context['fflow'][1].abs() < 1000)).float()
        
        return context
    

class AllLoader(object):
    def __init__(self, 
                 dataset_dir = Path('/data/keshav/flow360/FLOW360'), 
                 modes = ['train','test','val'], 
                 items = ['frame1','frame2', 'fflow', 'fflow3d'],
                 transform = {'resize' : (320, 640), 'rotation': False, 'seed' : 360},
                 train_batch_size = 8,
                 val_batch_size = 8,
                 test_batch_size = 8,
                 train_num_workers = 0,
                 val_num_workers = 0,
                 test_num_workers = 0,
                 train_shuffle = True,
                 test_shuffle = False,
                 val_shuffle = False,
                 drop_last = True,
                 ) -> None:
        super().__init__()
        self.drop_last = drop_last
        #self.root_path = root_path
        self.modes = modes
        self.items = items
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.test_num_workers = test_num_workers
        self.train_shuffle = train_shuffle
        self.test_shuffle = test_shuffle
        self.val_shuffle = val_shuffle
        
        if 'train' in modes:
            self.train = Vimeo90K_Train_Dataset(self.dataset_dir)
        
        #if 'val' in modes:
            #self.val = Flow360Loader(root_path, mode='val', items = items, #transform=transform)
        
        #if 'test' in modes:
            #self.test = Flow360Loader(root_path, mode='test', items = items, transform=transform)
    
    def loadtrain(self):
        assert 'train' in self.modes, "Train dataset not found"
        return DataLoader(self.train, 
                          shuffle=self.train_shuffle, 
                          num_workers = self.train_num_workers,
                          batch_size=self.train_batch_size,
                          #prefetch_factor=self.train_num_workers * 2,
                          drop_last=self.drop_last)
    
    def loadval(self):
        assert 'val' in self.modes, "Val dataset not found"
        return DataLoader(self.val, 
                          shuffle=self.val_shuffle, 
                          num_workers = self.val_num_workers,
                          batch_size=self.val_batch_size,
                          #prefetch_factor=self.val_num_workers * 2
                          )
    
    def loadtest(self):
        assert 'test' in self.modes, "Test dataset not found"
        return DataLoader(self.test, 
                          shuffle=self.test_shuffle, 
                          num_workers = self.test_num_workers,
                          batch_size=self.test_batch_size,
                          #prefetch_factor=self.test_num_workers * 2
                          )


class Vimeo90K_Train_Dataset(Dataset):
    def __init__(self, 
                 dataset_dir='data/vimeo_triplet', 
                 flow_dir=None, 
                 augment=True, 
                 crop_size=(224, 224)):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.crop_size = crop_size
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        self.flow_t0_list = []
        self.flow_t1_list = []
        if flow_dir is None:
            flow_dir = 'flow'
        with open(os.path.join(dataset_dir, 'tri_trainlist.txt'), 'r') as f:
            for i in f:
                name = str(i).strip()
                if(len(name) <= 1):
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))
                self.flow_t0_list.append(os.path.join(dataset_dir, flow_dir, name, 'flow_t0.flo'))
                self.flow_t1_list.append(os.path.join(dataset_dir, flow_dir, name, 'flow_t1.flo'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])
        flow_t0 = read(self.flow_t0_list[idx])
        flow_t1 = read(self.flow_t1_list[idx])
        flow = np.concatenate((flow_t0, flow_t1), 2).astype(np.float64)

        rotcon1 = getRandomRotationConfig(batch = 1)
        rotcon2 = getRandomRotationConfig(batch = 1)

        pitch = rotcon1[0].get('pitch')
        yaw = rotcon1[0].get('yaw')
        roll = rotcon1[0].get('roll')
        
        pitch_ = rotcon2[0].get('pitch')
        yaw_ = rotcon2[0].get('yaw')
        roll_ = rotcon2[0].get('roll')

        rot = rotcon1
        rot_ = rotcon2

        if self.augment == True:
            img0, imgt, img1, flow = random_resize(img0, imgt, img1, flow, p=0.1)
            img0, imgt, img1, flow = random_crop(img0, imgt, img1, flow, crop_size=self.crop_size)
            img0, imgt, img1, flow = random_reverse_channel(img0, imgt, img1, flow, p=0.5)
            img0, imgt, img1, flow = random_vertical_flip(img0, imgt, img1, flow, p=0.3)
            img0, imgt, img1, flow = random_horizontal_flip(img0, imgt, img1, flow, p=0.5)
            img0, imgt, img1, flow = random_rotate(img0, imgt, img1, flow, p=0.05)
            img0, imgt, img1, flow = random_reverse_time(img0, imgt, img1, flow, p=0.5)
                
        
        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return {'frame1': img0.float(), 'imgt': imgt.float(), 'frame2': img1.float(), 'flow': flow.float(), 'embt': embt, 'pitch': pitch, 'yaw': yaw,'roll': roll,'pitch_': pitch_, 'yaw_': yaw_,'roll_': roll_,'rot': rot,'rot_': rot_}
