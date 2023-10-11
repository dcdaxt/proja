from utils import AverageMeter
from dataloader import AllLoader
from RAFT import load_raft
from AMT import load_amts
from losses.loss import Loss ,TernaryLoss ,MultipleFlowLoss ,CharbonnierLoss
import torch.nn as nn
import torch
import cv2
import torch.optim as optim
from simsiam import Siam360
import os
from tqdm import tqdm
from equi_utils import rotate_eq, rotate_PIL, flow_rotation, getRandomRotationConfig
from evaluate_amt import validate_360vds
import numpy as np
from pathlib import Path
import utils
import random
from ktnfyraft import getKTNisedRaft
from sys import exit
# MAX_FLOW = 300
# NUM_EPOCH = 100
# TRAIN_BATCH = 16
# VAL_BATCH = 16
# CLIP_GRAD = True
# DEVICE_IDS = [0,1,2,3,4,5,6,7]
# TRAIN = True
# USE_DENISTY_MASK = False
# MODEL_PATH = 'cache/_smallermotion_cache_final_version001.pt'
# MODEL_NAME = f"_smallermotion_cache_final_version001{'_WEIGHTED' if USE_DENISTY_MASK else ''}"
# MODEL_SAVE = f'cache/{MODEL_NAME}.pt'
# BENCHMARK = False
# LOAD = True
# FINETUNE = False
# SWITCH_ROTATION = False
# DOUBLE_ROTATION = False


def fetch_optimizer(model, epochs, steps_per_epoch, init_lr = 0.00002, wdecay = 0.00005, epsilon = 1e-8):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=wdecay, eps=epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = init_lr, epochs = epochs, steps_per_epoch = steps_per_epoch,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class EarlyStopping():
    """
    credit: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
    

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass



def run(args):
    init_lr = 0.05 * args.train_batch / 256
    early_stopping = EarlyStopping(patience = args.patience, min_delta = args.min_delta)
    loader = AllLoader(dataset_dir = '/home/public/wym/vimeostyle_360VDSO/',modes = ['train'], train_batch_size=args.train_batch, val_batch_size=args.val_batch)
    train_loader = loader.loadtrain()
    
    if args.ktn:    
        model = Siam360(getKTNisedRaft().train(True), finetune = args.finetune)
    else:
        model = Siam360(flow_encoder = load_amts.load(ckpt_path="/home/u222080208/CODE/A/AMT/pretrained_weight/amt-s.pth").train(True), finetune = args.finetune)
        
    
    def train(loader, criterion, init_lr, epoch, print_freq = 10):
        optimizer, scheduler = fetch_optimizer(model, epochs = args.num_epoch, steps_per_epoch = len(loader) + args.train_batch, init_lr = init_lr)
        scaler = GradScaler(enabled = args.grad_scaler)
        
        if not args.finetune:    
            losses_sim = AverageMeter("Loss", ':.4f')
        
        losses = AverageMeter("Loss", ':.4f')
        loss_vfi_meter  = AverageMeter("Loss VFI", ':.4f')
        
        dataloader = tqdm(loader)
        
        dataloader.set_description_str(f"[TRAINING]")
        
        for i, data in enumerate(dataloader):
            if utils.stop(write = False):
                break
            optimizer.zero_grad()

            
            frame1  = data['frame1']
            frame2  = data['frame2']
            imgt = data['imgt']
            embt = data['embt']
            
            pitch = data['pitch'].cuda()
            yaw = data['yaw'].cuda()
            roll = data['roll'].cuda()
            
            pitch_ = data['pitch_'].cuda()
            yaw_ = data['yaw_'].cuda()
            roll_ = data['roll_'].cuda()
            
            
            flow = data['flow']


            
    
            
            # rotconfigs
            rot_ = [{'pitch':p, 'roll':r, 'yaw':y} for p,r,y in  zip(pitch_.tolist(), roll_.tolist(), yaw_.tolist())]
            rot = [{'pitch':p, 'roll':r, 'yaw':y} for p,r,y in  zip(pitch.tolist(), roll.tolist(), yaw.tolist())]
            
            frame1_ = rotate_eq(frame1, rots = rot_, mode = "bilinear")
            frame2_ = rotate_eq(frame2, rots = rot_, mode = "bilinear")
            
            if args.double_rotation:
                frame1 = rotate_eq(frame1, rots = rot, mode = "bilinear")
                frame2 = rotate_eq(frame2, rots = rot, mode = "bilinear")
                flowgt = rotate_eq(flowgt, mode = "bilinear", rots = rot, map_range=True, map_min_src=flowgt.min(), map_max_src = flowgt.max(), map_min_des=0, map_max_des=1)
                
            
            
            frame1 = frame1.cuda()
            frame2 = frame2.cuda()
            imgt = imgt.cuda()
            flow = flow.cuda()
            frame1_ = frame1_.cuda()
            frame2_ = frame2_.cuda()
            
            #regularization
            if not args.finetune:    
                l1_regularization, l2_regularization = torch.tensor(0).float().cuda(), torch.tensor(0).float().cuda()
            
            if args.finetune:
                try:    
                    frame_predictions = model(frame1, frame2, frame1_, frame2_)
                except Exception as e:
                    print(e)
                    exit()
                    
                    
            else:
                try:
                    if args.switch_rotation:
                        prob = random.random()
                        if prob>0.5:
                            p1, p2, z1_prj, z2_prj, flow_predictions = model(x1 = frame1_, 
                                                                x2 = frame2_, 
                                                                x3 = frame1, 
                                                                x4 = frame2, 
                                                                rots = True, 
                                                                rots_ = False, 
                                                                pitch = pitch_, 
                                                                yaw = yaw_, 
                                                                roll = roll_, 
                                                                pitch_ = None, 
                                                                roll_ = None, 
                                                                yaw_ = None,)
                        else:
                            p1, p2, z1_prj, z2, flow_predictions = model(x1 = frame1, 
                                                                x2 = frame2, 
                                                                x3 = frame1_, 
                                                                x4 = frame2_, 
                                                                rots = False, 
                                                                rots_ = True, 
                                                                pitch = None, 
                                                                yaw = None, 
                                                                roll = None, 
                                                                pitch_ = pitch_, 
                                                                roll_ = roll_, 
                                                                yaw_ = yaw_,)
                    else:    
                        p1, p2, z1, z2, imgt_pred = model(x1 = frame1, 
                                                                x2 = frame2, 
                                                                x3 = frame1_, 
                                                                x4 = frame2_,
                                                                embt = embt, 
                                                                rots = False, 
                                                                rots_ = True, 
                                                                pitch = None, 
                                                                yaw = None, 
                                                                roll = None, 
                                                                pitch_ = pitch_, 
                                                                roll_ = roll_, 
                                                                yaw_ = yaw_,)
                    
                except Exception as e:
                    print(e)
                    exit()
            if not args.finetune:    
                for param in model.module.parameters():
                    l1_regularization += torch.norm(param, 1)**2
                    l2_regularization += torch.norm(param, 2)**2
            loss_weight = 1.0
            keys = ['imgt_pred', 'imgt']
            l_rec = CharbonnierLoss(loss_weight = loss_weight,keys = keys)
            l_ter = TernaryLoss(loss_weight = loss_weight,keys = keys)
            
            
            if not args.finetune:
                
                l_r =  l_rec(imgt_pred = imgt_pred, imgt = imgt)
                l_t = l_ter(imgt_pred = imgt_pred, imgt = imgt)
                l_r = l_r.item()
                l_t = l_t.item()
                loss_vfi = l_r + l_t 
                loss_sim = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                loss = loss_vfi + loss_vfi + 1e-10*l1_regularization + 1e-5*l2_regularization *0.1
            else:
                loss = loss_vfi
            if not args.finetune:    
                losses_sim.update(loss_sim.item(), frame1.size(0))
            
            loss_vfi_meter.update(loss_vfi, frame1.size(0))
            losses.update(loss.item(), frame1.size(0))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            if i%print_freq == 0:
                if not args.finetune:    
                    dataloader.set_postfix_str(f"Loss: {losses.avg:.4f} | FlowLoss: {loss_vfi_meter.avg:.4f} | SimLoss: {losses_sim.avg:.4f} | epoch: {epoch}")
                else:
                    dataloader.set_postfix_str(f"Loss: {losses.avg:.4f} | epoch: {epoch}")
        return losses.avg
    
    if not args.benchmark:
        if args.load:
            if Path(args.model_path).exists():
                try:
                    model.load_state_dict(torch.load(args.model_path))
                    model.eval()
                    print(f"model succesfully loaded from {args.model_path}")
                except Exception as e:
                    print(e)
                    torch.cuda.empty_cache()
                    exit()
        
        criterion = nn.CosineSimilarity(dim=1).cuda()
        model = model.cuda()
        model = nn.DataParallel(model, device_ids = args.gpus)
        psnr = 100
        
        for epoch in range(args.num_epoch):
            try:
                model.eval()
                print(f'begin eval')
                results_ = validate_360vds(root='/home/public/wym/vimeostyle_360VDS', ckpt_path='/home/u222080208/CODE/A/AMT/pretrained_weight/amt-s.pth')
                print(f'finish eval')
                psnr_ = results_['psnrs']
                data = results_['data']
                filename = results_['filepath']
                if psnr_ < psnr:
                    psnr = psnr_
                    if args.train:
                        print(f"results saved at {filename}")
                        data.to_csv(filename, index = None)
                        
                        print(f"Model saved with current psnr {psnr_}")
                        print(f'begin modle save') 
                        torch.save(model.module.state_dict(), args.model_save)
                        print(f'finish modle save') 
            except Exception as e:
                torch.cuda.empty_cache()
                print(e)
                exit()
        
            if (not args.train) or args.benchmark:
                break
            
            try:
                print(f'begin train')   
                model.train(True)
                loss_ = train(train_loader, criterion, init_lr, epoch)
                early_stopping(loss_)
                if early_stopping.early_stop:
                    break
                print(f'finish train')  
                
            except Exception as e:
                torch.cuda.empty_cache()
                print(e)
                exit()
            
            if utils.stop():
                break