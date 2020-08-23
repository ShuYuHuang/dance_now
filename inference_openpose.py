# import sys,os
# from pathlib import Path
# sys.path.append(str(Path("./")))
import torch
from torch import nn
#from torch.autograd import Variable
from torchvision import transforms as ts
from utils import loaders,nnmodels
import numpy as np

import functools
##------------------------------------------Parameters-------------------------------------------------------####
HALF_HEAD=loaders.HEAD_SIZE//2
##------------------------------------------Functions--------------------------------------------------------####
def to_4d(inp):
    return np.expand_dims(inp,0)
def patch_head(body_img,head_img,head_center):
    body_img[:,:,head_center[1]-HALF_HEAD:head_center[1]+HALF_HEAD,head_center[0]-HALF_HEAD:head_center[0]+HALF_HEAD]=head_img
    return torch.squeeze(body_img,0)
def norm_0to1(inp):
    return (inp-inp.min())/(inp.max()-inp.min())
##------------------------------------------Execution--------------------------------------------------------####
def main(label_path="./data/anime/train_label/img_1_0_keypoints.json"):
    
    print("1.Load Models")#------------------------------------------------------------------------------------
    G_body=torch.load("./model_body/netGbody_struct.pth").cuda()
    G_body.load_state_dict(torch.load("./model_body/netGbody_run220.pt"))
    G_face=torch.load("./model_face/netGface_struct.pth").cuda()
    G_face.load_state_dict(torch.load("./model_face/netGface_run90.pt"))
    
    print("2.Load Data")#-------------------------------------------------------------------------------------
    lbl_sample,head_mtx,head_center = loaders.read_label(label_path,ifhead=True,ifbody=True)
    with torch.no_grad():
        
        print("3.Stick2body Generation")#-----------------------------------------------------------------------
        body_img=G_body(torch.tensor(to_4d(lbl_sample), device=torch.device('cuda:0')))
        print("Output body only")#------------------------------------------------------------------------------
        body_pil=ts.ToPILImage()(norm_0to1(torch.squeeze(body_img,0).cpu()))
        body_pil.save("./body.png")
        
        print("4.Face Enhance")#--------------------------------------------------------------------------------
        fake_head=body_img[:,:,head_center[1]-HALF_HEAD:head_center[1]+HALF_HEAD,\
                               head_center[0]-HALF_HEAD:head_center[0]+HALF_HEAD]
        head_lbl=torch.tensor(to_4d(head_mtx),dtype=torch.float32, device=torch.device('cuda:0'))
        head_input=torch.cat((fake_head, head_lbl), dim=1)
        head_buff=G_face(head_input)
        head_enhance=head_buff+fake_head
        
        anime=patch_head(body_img,head_enhance,head_center).cpu()
        
        print("Over, output image")#----------------------------------------------------------------------------
        anime_pil=ts.ToPILImage()(norm_0to1(anime))
        anime_pil.save("./cat_head_body.png")
    
main()