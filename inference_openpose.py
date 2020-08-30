import sys,os
from pathlib import Path
sys.path.append(str(Path("./")))

from torchvision import transforms as ts
import torch
from torch import nn
import torch.nn.functional as F

#from torch.autograd import Variable
from imageio import imread,imwrite
from utils import loaders,nnmodels,BASNet
from utils import integration as intg

import numpy as np
import cv2
import functools
import time


#################################################################################################################
##------------------------------------------Parameters-------------------------------------------------------####
#################################################################################################################
HALF_HEAD=loaders.HEAD_SIZE//2
TRIM_SIZE=7;
TRIM_EROTION=1
CUDA_DEVICE=torch.device("cuda:0")
MODEL_DIR="../model"
CHAR_SET=["malcom","soldier","girl"]
MODEL_SET=dict()
MODEL_SET["malcom"]=["netGbody_run220.pt","netGface_run410.pt"]
MODEL_SET["soldier"]=["netGbody_S_run360.pt","netGface_S_run200.pt"]
MODEL_SET["girl"]=["netGbody_G_run170.pt"]

label_path="../test/test_label/test_img_keypoints.json"
params = dict()
params["model_folder"] = "./models/"
params["output_resolution"] = "512x512"
params["write_json"] = "./temp/test_label/"
params["num_gpu"] = int(1)
params["face"] = True
params["hand"] = True
numberGPUs = int(params["num_gpu"])

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
global t,now
t=0
now=time.perf_counter()
# #------------------------------------------------------------------------------------------------------------
# print("0-1.Load Openpose")#------------------------------------------------------------------------------------
# #------------------------------------------------------------------------------------------------------------
# opWrapper = op.WrapperPython()
# opWrapper.configure(params)
# opWrapper.start()

#------------------------------------------------------------------------------------------------------------
print("0-2.Load Body Models")#---------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
G_body=dict()
G_face=dict()
for char in CHAR_SET:
    G_body[char]=torch.load(os.path.join(MODEL_DIR,"netGbody_struct.pth")).cuda()
    G_body[char].load_state_dict(torch.load(os.path.join(MODEL_DIR,MODEL_SET[char][0])))
    if char!="girl":
        G_face[char]=torch.load(os.path.join(MODEL_DIR,"netGface_struct.pth")).cuda()
        G_face[char].load_state_dict(torch.load(os.path.join(MODEL_DIR,MODEL_SET[char][1])))
#------------------------------------------------------------------------------------------------------------
print("0-3.Load Mask Models")#---------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
Mask_Model = BASNet.BASNet(3,1).cuda()
Mask_Model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"basnet.pth")))
t,now=time.perf_counter()-now,time.perf_counter()    
print('{:4f} {:4f}'.format(t,now))

#################################################################################################################
##------------------------------------------Execution--------------------------------------------------------####
#################################################################################################################

def transfer(input_img="../test/test_img/test_img.png",char="malcom"):
    global t,now
    global G_body,G_face,Mask_Model,Mat_model
#     ###########################################################################################################
#     print("1.Generate Stick")#-------------------------------------------------------------------------------------
#     ###########################################################################################################
#     # Crop and resize image
#     resized_img=loaders.__scale_width(imread(input_img), loaders.BODY_SIZE)
    
#     #imwrite("./temp/test_img/test_img.png",resized_img)
#     # Gen Stick
    
#     #os.system("openpose.bin  --image_dir ./temp/test_img/ --output_resolution 512x512 --write_json ./temp/test_label/ --display 0 --hand --face --render_pose 0")
#     datums = []
#     datum = op.Datum()
#     datum.cvInputData = resized_img
#     datums.append(datum)
#     opWrapper.waitAndEmplace([datums[-1]])
#     datum = datums[0]
#     opWrapper.waitAndPop([datum])
    
#     openpose_time = time.time()
#     print(datum.poseKeypoints.shape)
    ###########################################################################################################
    print("1.Load Data")#-------------------------------------------------------------------------------------
    ###########################################################################################################
    t,now=time.perf_counter()-now,time.perf_counter()
    lbl_sample,head_mtx,head_center = loaders.read_label(label_path,ifhead=True,ifbody=True)
    #lbl_sample,head_mtx,head_center = loaders.label2feature(datum,ifhead=True,ifbody=True)
    with torch.no_grad():
        t,now=time.perf_counter()-now,time.perf_counter()
        print('{:4f} {:4f}'.format(t,now))
        ###########################################################################################################
        print("2.Stick2body Generation")#-----------------------------------------------------------------------
        ###########################################################################################################
        body_img=G_body[char](torch.tensor(to_4d(lbl_sample), device=CUDA_DEVICE))
        t,now=time.perf_counter()-now,time.perf_counter()
        print('{:4f} {:4f}'.format(t,now))
        ###########################################################################################################
        print("4.Face Enhance")#--------------------------------------------------------------------------------
        ###########################################################################################################
        if char!="girl":
            fake_head=body_img[:,:,head_center[1]-HALF_HEAD:head_center[1]+HALF_HEAD,\
                                   head_center[0]-HALF_HEAD:head_center[0]+HALF_HEAD]
            head_lbl=torch.tensor(to_4d(head_mtx),dtype=torch.float32, device=CUDA_DEVICE)
            head_input=torch.cat((fake_head, head_lbl), dim=1)
            #----------------------------------------------------------------------------------------------------------
            head_buff=G_face[char](head_input)
            #----------------------------------------------------------------------------------------------------------
            head_enhance=head_buff+fake_head
            body_img[:,:,head_center[1]-HALF_HEAD:head_center[1]+HALF_HEAD,head_center[0]-HALF_HEAD:head_center[0]+HALF_HEAD]=head_enhance
        t,now=time.perf_counter()-now,time.perf_counter()
        print('{:4f} {:4f}'.format(t,now))
        ###########################################################################################################
        print("5.Masking inference Mask")#--------------------------------------------------------------------------------
        ###########################################################################################################
        # extract to Numpy for Tensorflow
        
        animeimg=norm_0to1(body_img).detach().cpu().numpy().transpose(0,2,3,1)
        imwrite("./aniimg.png",animeimg[0])
        # ----------------gen raw inference mask-----------------------------------------------------------------------------------------

        d1_inf,_,_,_,_,_,_,_=Mask_Model(body_img)
        alpha =intg.infout_transform(d1_inf)
        t,now=time.perf_counter()-now,time.perf_counter()
        print('{:4f} {:4f}'.format(t,now))
        ###########################################################################################################
        print("6.Masking photo")#--------------------------------------------------------------------------------
        ###########################################################################################################
        ph_img=loaders.cv2_loader(input_img)
        photo_input=intg.ph_transform(ph_img).unsqueeze(0).cuda()
        
        d1_pho,_,_,_,_,_,_,_=Mask_Model(photo_input)
        photo_mask =intg.phout_transform(d1_pho)
        t,now=time.perf_counter()-now,time.perf_counter()
        print('{:4f} {:4f}'.format(t,now))
        ###########################################################################################################
        print("7.Photo inpaiting")#--------------------------------------------------------------------------------
        ###########################################################################################################
        for _ in range(3):
            photo_mask=cv2.erode(photo_mask,kernel=loaders.KERNEL_FACEEDGE)
            photo_mask=cv2.dilate(photo_mask,kernel=loaders.KERNEL_FACEEDGE)
            
        
        good_bg=cv2.inpaint(ph_img,photo_mask,3,cv2.INPAINT_NS)/255
#         tatal_mask=np.expand_dims(1-np.logical_or(photo_mask,alpha),2)
        alpha=np.expand_dims(alpha,2)
#         print(np.repeat(1-alpha,3,axis=2).shape)
#         print(np.repeat(tatal_mask,3,axis=2).shape)
#         b_img=intg.edge_blur(ph_img,np.repeat(1-alpha,3,axis=2),np.repeat(tatal_mask,3,axis=2))/255
        t,now=time.perf_counter()-now,time.perf_counter()
        print('{:4f} {:4f}'.format(t,now))
        ###########################################################################################################
        print("8.Combine Masked Photo and Masked inference image")#--------------------------------------------------------------------------------
        ###########################################################################################################
        good_img=good_bg*(1-alpha)+animeimg[0]*alpha
        ###########################################################################################################
        print("9.Refine Image Edge")#--------------------------------------------------------------------------------
        ###########################################################################################################
#         print(tatal_mask.shape,alpha.shape)
        
#         smooth_mask=intg.blur_fun((1-np.logical_xor(tatal_mask,1-alpha)).astype(np.float32))
        
#         fine_img=good_img*np.expand_dims(smooth_mask,2)
        
        t,now=time.perf_counter()-now,time.perf_counter()
        print('{:4f} {:4f}'.format(t,now))
        print("Over, output image")#----------------------------------------------------------------------------
        imwrite("./cat_head_body.png",good_img)
