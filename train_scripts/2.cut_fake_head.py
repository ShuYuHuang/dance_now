import sys,os
from pathlib import Path
sys.path.append(str(Path("../")))
from utils import loaders,model_body,nnmodels,create_head

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
<<<<<<< HEAD
from imageio import imread,imsave

import numpy as np
import cv2
=======
from skimage.io import imread,imsave

import numpy as np
>>>>>>> 72dfc472d73fe993ea067735b47f10c67b2b0a65
import functools

NUM_GPU=torch.cuda.device_count()
#-------------------------------------Loader Building---------------------------------



GAN_DIM=24+5+5+1
HEAD_GAN_DIM=14+1
<<<<<<< HEAD
BODY_SIZE=512
HALF_HEAD=loaders.HEAD_SIZE//2
LABEL_DIR="../data/soldier/train_label/"
FAKE_BODY_DIR="../rslt/soldier/fake_img/"
FAKE_HEAD_DIR="../rslt/soldier/fake_headimg/"
=======
HALF_HEAD=loaders.HEAD_SIZE//2
LABEL_DIR="../data/anime/train_label/"
FAKE_BODY_DIR="../rslt/anime/fake_img/"
FAKE_HEAD_DIR="../rslt/anime/fake_headimg/"
>>>>>>> 72dfc472d73fe993ea067735b47f10c67b2b0a65
os.makedirs(FAKE_BODY_DIR, exist_ok = True)
os.makedirs(FAKE_HEAD_DIR, exist_ok = True)

train_set=loaders.CostumImFolder(None,[LABEL_DIR],ifbody=True,ifhead=False)
print(train_set.transform)
train_loader=DataLoader(train_set, batch_size=8, shuffle=False,num_workers = 4*NUM_GPU,pin_memory=True)

def norm_0to1(inp):
    return (inp-inp.min())/(inp.max()-inp.min())
#-------------------------------------Model Building---------------------------------
norm_layer0 = functools.partial(nn.InstanceNorm2d, affine=False)
G_body=torch.load("../model_body/netGbody_struct.pth").cuda()
<<<<<<< HEAD
G_body.load_state_dict(torch.load("../model_body/netGbody_S_run360.pt"))
G_body = nn.DataParallel(G_body)

=======
G_body.load_state_dict(torch.load("../model_body/netGbody_run220.pt"))

G_body = nn.DataParallel(G_body)
>>>>>>> 72dfc472d73fe993ea067735b47f10c67b2b0a65
#-------------------------------------Model Inference, save image---------------------------------

with torch.no_grad():
    for imname,lbl_sample,_,head_center,_ in train_loader:
<<<<<<< HEAD
        out_img = G_body(torch.tensor(lbl_sample,dtype=torch.float32, device=torch.device('cuda:0')))
        head_center=head_center.int()
        for ii,img_t in enumerate(out_img):
            body_img = norm_0to1(img_t.cpu().numpy().transpose(1,2,0))*255
            
            cv2.imwrite(os.path.join(FAKE_BODY_DIR,f"{imname[ii]}.png"),cv2.cvtColor(body_img, cv2.COLOR_RGB2BGR))
            if (head_center[ii,1]-HALF_HEAD>=0 and\
               head_center[ii,1]+HALF_HEAD<BODY_SIZE and\
               head_center[ii,0]-HALF_HEAD>=0 and\
               head_center[ii,0]+HALF_HEAD<BODY_SIZE):
                head_img=body_img[head_center[ii,1]-HALF_HEAD:\
                                  head_center[ii,1]+HALF_HEAD,\
                                  head_center[ii,0]-HALF_HEAD:\
                                  head_center[ii,0]+HALF_HEAD,:]
            cv2.imwrite(os.path.join(FAKE_HEAD_DIR,f"{imname[ii]}.png"),cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR))
=======
        out_img = G_body(torch.tensor(lbl_sample,dtype=torch.float32, device=torch.device('cuda:0')))  
        for ii,img_t in enumerate(out_img):
            body_img = norm_0to1(img_t.cpu().numpy().transpose(2,1,0))*255
            head_img=body_img[head_center[1]-HALF_HEAD:head_center[1]+HALF_HEAD,\
                               head_center[0]-HALF_HEAD:head_center[0]+HALF_HEAD,:]
            imsave(os.path.join(FAKE_BODY_DIR,f"{imname[ii]}.png"),body_img)
            imsave(os.path.join(FAKE_HEAD_DIR,f"{imname[ii]}.png"),head_img)
>>>>>>> 72dfc472d73fe993ea067735b47f10c67b2b0a65
            
          
        
#-------------------------------------Cut Head ---------------------------------
<<<<<<< HEAD
=======
#head_dataset=create_head.create([FAKE_BODY_DIR],
#                                 [LABEL_DIR],
#                               FAKE_HEAD_DIR)
#head_loader=DataLoader(head_dataset, batch_size=4,\
#                        shuffle=False,num_workers = 4*NUM_GPU,pin_memory=True)
#
#for head_array,head_center in head_loader:
#    continue
>>>>>>> 72dfc472d73fe993ea067735b47f10c67b2b0a65
