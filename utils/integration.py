import cv2
import tensorflow as tf
import torch, sys, cv2, os
sys.path.insert(1, 'pytorch_deep_image_matting/core')
import numpy as np
import time


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



###### --------------------Create trimap------------------------------######

def gen_trimap(in_mask, size, erosion=False):
    pixels = 2*size + 1;                                     ## Double and plus 1 to have an odd-sized kernel

    if erosion is not False:
        erosion = int(erosion)                 ## Design an odd-sized erosion kernel
        msk = cv2.erode(in_mask, EROTION_KERNEL, iterations=erosion)  ## How many erosion do you expec
        #image = np.where(image > 0, 255, image)                       ## Any gray-clored pixel becomes white (smoothing)
        # Error-handler to prevent entire foreground annihilation
        if cv2.countNonZero(msk) == 0:
            print("ERROR: foreground has been entirely eroded");
            sys.exit();
    dilation  = cv2.dilate(msk, TRIM_KERNEL, iterations = 1)

    dilation  = np.where(dilation == 1., 0.5, dilation) 	## WHITE to GRAY
    remake    = np.where(dilation != 0.5, 0., dilation)		## Smoothing
    remake    = np.where(msk > 0.5, 0.7, dilation)		## mark the tumor inside GRAY
    remake    = np.where(remake < 0.5, 0., remake)		## Embelishment
    remake    = np.where(remake > 0.7, 0., remake)		## Embelishment
    remake    = np.where(remake == 0.7, 1., remake)		## GRAY to WHITE

    #############################################
    # Ensures only three pixel values available #
    # TODO: Optimization with Cython            #
    #############################################    
    remake[np.logical_and(remake != 0. ,remake != 1.)]=0.5
    print("generate trimap(size: " + str(size) + ", erosion: " + str(erosion) + ")")
    return remake

 

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





