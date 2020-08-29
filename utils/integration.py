import cv2
import torch, sys, cv2, os
sys.path.insert(1, 'pytorch_deep_image_matting/core')
import numpy as np
from torchvision import transforms as ts


######################################################################
# Parameters
######################################################################

cuda = torch.cuda.is_available()
BACKGROUND_SIZE = 512
TRIM_KERNEL=np.array([[1]], dtype=np.uint8)
EROTION_KERNEL= np.ones((3,3), np.uint8)  
EDGE_BLUR_KERNEL=np.array([[0, 0, 1, 1 ,1, 0, 0],
                 [0, 0, 1, 1, 1, 0, 0],
                 [1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 0, 0],
                 [0, 0, 1, 1 ,1, 0, 0]], dtype=np.uint8)
# EDGE_BLUR_KERNEL=np.array([[0, 0, 1, 0, 0],
#                              [0, 1, 1, 1, 0],
#                              [1, 1, 1, 1, 1],
#                              [0, 1, 1, 1, 0],
#                              [0, 0, 1, 0, 0]], dtype=np.uint8)
SMOOTH_KER_SIZE=31
######################################################################
# Img Pre-Process functions
######################################################################


def __scale_width(img, target_width):
    shape_dst = np.min(img.shape[:2])
    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2

    img = img[oh:oh + shape_dst, ow:ow + shape_dst]
    return cv2.resize(img, (target_width, target_width))
 
ph_transform = ts.Compose([
         ts.Lambda(lambda im: __scale_width(im,256)),
         ts.ToTensor(),
         torch.FloatTensor,
         ts.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
         ])

phout_transform = ts.Compose([
         ts.Lambda(lambda tr: (tr>0.9).squeeze().detach().cpu().numpy()),
         ts.Lambda(lambda im: cv2.resize(im, (512,512)))
         ])
infout_transform = ts.Compose([
         ts.Lambda(lambda tr: (tr>0.9).squeeze().detach().cpu().numpy()),
         ])
######################################################################
# Img Post-Process functions
######################################################################

def blur_fun(im):
    return cv2.GaussianBlur(im,(SMOOTH_KER_SIZE,SMOOTH_KER_SIZE),0)

def edge_blur(img,mask,mask2):
    # convert to 0~1
    mask=(mask/mask.max()).round()
    # create eroded mask+edge mask
    fatmask=cv2.erode(mask,EDGE_BLUR_KERNEL,iterations=2)
    gradient = cv2.morphologyEx(fatmask, cv2.MORPH_GRADIENT, EDGE_BLUR_KERNEL,iterations=3)
    # Edge Image bluring
    b_img=blur_fun(img*gradient)
    # combine 2 images
    return img*(mask2==1)+b_img*(mask2!=1)





