import sys,os
from pathlib import Path
sys.path.append(str(Path("./")))     
import tensorflow as tf
import torch
from torch import nn
#from torch.autograd import Variable
from imageio import imread,imwrite
from utils import loaders,nnmodels,net
from utils import integration as intg
import numpy as np
import cv2
import functools
#################################################################################################################
##------------------------------------------Parameters-------------------------------------------------------####
#################################################################################################################
HALF_HEAD=loaders.HEAD_SIZE//2
TRIM_SIZE=7;
TRIM_EROTION=1
CUDA_DEVICE=torch.device("cuda:0")
MODEL_DIR="./model"
#################################################################################################################
##------------------------------------------Functions--------------------------------------------------------####
#################################################################################################################
def to_4d(inp):
    return np.expand_dims(inp,0)
def patch_head(body_img,head_img,head_center):
    body_img[:,:,head_center[1]-HALF_HEAD:head_center[1]+HALF_HEAD,head_center[0]-HALF_HEAD:head_center[0]+HALF_HEAD]=head_img
    return torch.squeeze(norm_0to1(body_img),0)
def norm_0to1(inp):
    return (inp-inp.min())/(inp.max()-inp.min())
#################################################################################################################
##------------------------------------------Execution--------------------------------------------------------####
#################################################################################################################
def main():
    input_img="./test/test_img/test_img.png"
    label_path="./test/test_label/test_img_keypoints.json"
    
    ###########################################################################################################
    print("1.Load Models")#------------------------------------------------------------------------------------
    ###########################################################################################################
    G_body=torch.load(os.path.join(MODEL_DIR,"netGbody_struct.pth")).cuda()
    G_body.load_state_dict(torch.load(os.path.join(MODEL_DIR,"netGbody_run220.pt")))
    G_face=torch.load(os.path.join(MODEL_DIR,"netGface_struct.pth")).cuda()
    G_face.load_state_dict(torch.load(os.path.join(MODEL_DIR,"netGface_run410.pt")))
    Mask_Model = net.DeepLabModel(os.path.join(MODEL_DIR,"model_xception65_coco_voc_trainval.tar.gz"))
    Mat_model = net.VGG16(1).cuda(0)
    Mat_model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"VGG16_weight.pt")))
    
    ###########################################################################################################
    print("2.Load Data")#-------------------------------------------------------------------------------------
    ###########################################################################################################
    lbl_sample,head_mtx,head_center = loaders.read_label(label_path,ifhead=True,ifbody=True)
    with torch.no_grad():
        
        ###########################################################################################################
        print("3.Stick2body Generation")#-----------------------------------------------------------------------
        ###########################################################################################################
        body_img=G_body(torch.tensor(to_4d(lbl_sample), device=CUDA_DEVICE))
        #anime=torch.squeeze(norm_0to1(body_img),0).detach().cpu().numpy().transpose(1,2,0)
        #imwrite("./cat_head_body.png",anime)
        
        ###########################################################################################################
        print("4.Face Enhance")#--------------------------------------------------------------------------------
        ###########################################################################################################
        fake_head=body_img[:,:,head_center[1]-HALF_HEAD:head_center[1]+HALF_HEAD,\
                               head_center[0]-HALF_HEAD:head_center[0]+HALF_HEAD]
        head_lbl=torch.tensor(to_4d(head_mtx),dtype=torch.float32, device=CUDA_DEVICE)
        head_input=torch.cat((fake_head, head_lbl), dim=1)
        head_buff=G_face(head_input)
        head_enhance=head_buff+fake_head
        body_img[:,:,head_center[1]-HALF_HEAD:head_center[1]+HALF_HEAD,head_center[0]-HALF_HEAD:head_center[0]+HALF_HEAD]=head_enhance
        
        ###########################################################################################################
        print("5.Masking inference Mask")#--------------------------------------------------------------------------------
        ###########################################################################################################
        # extract to Numpy for Tensorflow
        animeimg=norm_0to1(body_img).detach().cpu().numpy().transpose(0,2,3,1)
        # ----------------gen raw mask-----------------------------------------------------------------------------------------
        inf_mask = np.sign(Mask_Model.run(animeimg*255),dtype=np.float32)
        
        ###########################################################################################################
        print("6.Enhance inference mask")#--------------------------------------------------------------------------------
        ###########################################################################################################
        # ----------------gen tri map----------------------------------------------------------------------------------------
        tri_inf=intg.gen_trimap(inf_mask, size=TRIM_SIZE, erosion=TRIM_EROTION)
        tensor_tri_inf=torch.tensor(tri_inf,device=CUDA_DEVICE,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        inf_input = torch.cat((body_img, tensor_tri_inf), 1)
        pred_mattes, _ = Mat_model(inf_input)
        
        alpha=pred_mattes[0].data.detach().cpu().numpy().transpose(1,2,0)>0.9
        
        ###########################################################################################################
        print("7.Masking photo")#--------------------------------------------------------------------------------
        ###########################################################################################################
        photo_input=np.expand_dims(imread(input_img),0)
        photo_mask = np.sign(Mask_Model.run(photo_input),dtype=np.float32)
        tatal_mask=np.expand_dims(1-np.logical_or(photo_mask,inf_mask),2)
        b_img=intg.edge_blur(photo_input[0],np.repeat(1-alpha,3,axis=2).astype(np.float32),np.repeat(tatal_mask,3,axis=2))/255
        ###########################################################################################################
        print("8.Combine Masked Photo and Masked inference image")#--------------------------------------------------------------------------------
        ###########################################################################################################
        good_img=b_img*(1-alpha)+animeimg[0]*alpha
        ###########################################################################################################
        print("9.Refine Image Edge")#--------------------------------------------------------------------------------
        ###########################################################################################################
        print(tatal_mask.shape,alpha.shape)
        smooth_mask=intg.blur_fun((1-np.logical_xor(tatal_mask,1-alpha)).astype(np.float32))
        
        
        fine_img=good_img*np.expand_dims(smooth_mask,2)
        
        print("Over, output image")#----------------------------------------------------------------------------
        imwrite("./cat_head_body.png",fine_img)
    
main()