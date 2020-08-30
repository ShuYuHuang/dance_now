import torch
from torchvision import transforms as ts
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import StandardTransform
from torch.utils.data import Dataset
import os

import cv2
from cv2 import imread
import numpy as np
from typing import Any, Callable, List, Optional, Tuple
import json


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',".json"
]
HEAD_SIZE=64
HEAD_DIM=14+1
BODY_SIZE=512
BODY_DIM=24+5+5+1
HAND_DIM=5
############--------------------------Crop Image---------------------------################
def __scale_width(img, target_width):
    shape_dst = np.min(img.shape[:2])
    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2

    img = img[oh:oh + shape_dst, ow:ow + shape_dst]
    return cv2.resize(img, (target_width, target_width))

############--------------------------Image Preprocess---------------------------################
transform_dict = {
        'body': ts.Compose(
        [ts.Lambda(lambda im: __scale_width(im, BODY_SIZE)),   
         ts.ToTensor(),
         torch.FloatTensor,
         ts.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
         ]),
        'head': ts.Compose(
        [
         ts.Lambda(lambda im: __scale_width(im, HEAD_SIZE)),
         ts.ToTensor(),
         torch.FloatTensor,
         ts.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
         ]),
        'tgthead': ts.Compose(
        [
         ts.Lambda(lambda im: __scale_width(im, HEAD_SIZE)),
         ts.ToTensor(),
         torch.FloatTensor,
         ts.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
         ])}

############--------------------------Skeleton List---------------------------################
bodylink_list=[
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14),
    (0,15),(0,16),(15,17),(16,18),
    (14,19),(19,20),(14,21),
    (11,22),(22,23),(11,24)
]
handlink_list=[
    (0,1),(0,5),(0,9),(0,13),(0,17),
    (1,2),(5,6),(9,10),(13,14),(17,18),
    (2,3),(6,7),(10,11),(14,15),(18,19),
    (3,4),(7,8),(8,12),(15,16),(19,20)
]

############--------------------------DILATION KERNEL---------------------------################
KERNEL=np.array([[0, 0, 1, 0, 0],
                 [0, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 0],
                 [0, 0, 1, 0, 0]], dtype=np.uint8)
KERNEL_HAND=np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=np.uint8)
KERNEL_FACEEDGE=np.array([[0, 0, 1, 1 ,1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1 ,1, 0, 0]], dtype=np.uint8)
############--------------------------Make or Get Functions---------------------------################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def cv2_loader(path):
    if not os.path.exists(path):
        print(path,"not exist")
    imgg=imread(path)
    if not imgg.any():
        print(path,"not good")
        os.system(f"rm -f {path}")
        
    return cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)

def make_dataset(dir_img, dir_lbl,dir_tgt=None):
    images = []
    label=[]
    prefix=[]
    target=[]
    for ii,dir in enumerate(dir_lbl):
        assert os.path.isdir(dir), '%s is not a valid dirsectory' % dir
        #print(os.walk(dir))
        if dir_img is not None:
            for rooti,_,fnames in os.walk(dir_img[ii]):
                for fname in sorted(fnames):
                    if is_image_file(fname) and not "checkpoint" in fname:
                        img_suffix=fname[-4:]
                        break
                break
        if dir_tgt:
            for roott,_,_ in os.walk(dir_tgt[ii]):
                break
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname) and not "checkpoint" in fname:
                    path = os.path.join(root, fname)
                    
                    if dir_tgt:
                        path2=os.path.join(rooti,f"{fname[:-15]}{img_suffix}")
                        path3=os.path.join(roott,f"{fname[:-15]}{img_suffix}")
                        if not os.path.exists(path2) or not os.path.exists(path3):
                            os.system(f"rm -f {path}")
                            os.system(f"rm -f {path2}")
                            os.system(f"rm -f {path3}")
                        else:
                            label.append(path)
                            images.append(path2)
                            target.append(path3)
                            prefix.append(fname[:-15])
                            
                    else:
                        if dir_img is not None:
                            path2=os.path.join(rooti,f"{fname[:-15]}{img_suffix}")
                            if os.path.exists(path2):
                                images.append(path2)
                                label.append(path)
                                prefix.append(fname[:-15])
                        else:
                            label.append(path)
                            prefix.append(fname[:-15])
                else:
                    print(f"{fname} not imported")

    return images,label,prefix,target

def read_label(fname,ifhead,ifbody):
    with open(fname) as f:
        data = json.load(f)
    Mtx=np.zeros((BODY_DIM,BODY_SIZE,BODY_SIZE),dtype=np.float32)
    head_mtx=np.zeros((HEAD_DIM,HEAD_SIZE,HEAD_SIZE),dtype=np.float32)
    head_cent=np.array([0,0])  
    head_inside=False
    for jj in range(len(data["people"])):
        face=np.array(data["people"][jj]["face_keypoints_2d"])# *70 points
        face[face>511]=511
        face=[ int(x) for x in face ]
        head_cent=np.array([face[3*30],face[3*30+1]],dtype=np.int)
        if ifhead:
            # ----Handeling Head Keypoints and Face extraction 
            

            if face is not None:
                n_face=face.__len__()//3
                face=[ int(x) for x in face ]
                head_cent=np.array([face[3*30],face[3*30+1]],dtype=np.int)
                for ii in range(0,n_face):
                    if face[3*ii+1]-head_cent[1]+HEAD_SIZE//2>=0 and\
                       face[3*ii+1]-head_cent[1]+HEAD_SIZE//2<HEAD_SIZE and\
                       face[3*ii]-head_cent[0]+HEAD_SIZE//2>=0 and\
                       face[3*ii]-head_cent[0]+HEAD_SIZE//2<HEAD_SIZE:
                        head_mtx[ii%14,face[3*ii+1]-head_cent[1]+HEAD_SIZE//2\
                                      ,face[3*ii]-head_cent[0]+HEAD_SIZE//2]=1
        if ifbody:
            #Handling pose and hand gesture
            pose=np.array(data["people"][jj]["pose_keypoints_2d"])# *25 points
            hand_l=np.array(data["people"][jj]["hand_left_keypoints_2d"])# *25 points
            hand_r=np.array(data["people"][jj]["hand_right_keypoints_2d"])
            pose[pose>511]=511
            hand_l[hand_l>511]=511
            hand_r[hand_r>511]=511
            if pose is not None:
                #n_pose=pose.__len__()//3
                pose=[ int(x) for x in pose ]
                ii=0
                for p1,p2 in bodylink_list:
                    #dist=int(abs(complex(pose[3*p1+1]-pose[3*p2+1],pose[3*p1]-pose[3*p2])))+1
                    dist=int(max(abs(pose[3*p1+1]-pose[3*p2+1]),abs(pose[3*p1]-pose[3*p2])))+1
                    xline=np.linspace(pose[3*p1+1],pose[3*p2+1],dist).astype(int)
                    yline=np.linspace(pose[3*p1],pose[3*p2],dist).astype(int)
                    Mtx[ii,xline,yline]=1
                    ii=ii+1
            if hand_l is not None:

                hand_l=[ int(x) for x in hand_l ]
                ii=0
                for p1,p2 in handlink_list:
                    dist=int(abs(complex(hand_l[3*p1+1]-hand_l[3*p2+1],hand_l[3*p1]-hand_l[3*p2])))+1
                    #dist=int(max(abs(hand_l[3*p1+1]-hand_l[3*p2+1]),abs(hand_l[3*p1]-hand_l[3*p2])))+1
                    xline=np.linspace(hand_l[3*p1+1],hand_l[3*p2+1],dist).astype(int)
                    yline=np.linspace(hand_l[3*p1],hand_l[3*p2],dist).astype(int)
                    Mtx[BODY_DIM-HAND_DIM*2-1+ii%5,xline,yline]=1
                    ii+=1

            if hand_r is not None:
                hand_r=[ int(x) for x in hand_r ]
                ii=0
                for p1,p2 in handlink_list:
                    dist=int(abs(complex(hand_r[3*p1+1]-hand_r[3*p2+1],hand_r[3*p1]-hand_r[3*p2])))+1
                    #dist=int(max(abs(hand_r[3*p1+1]-hand_r[3*p2+1]),abs(hand_r[3*p1]-hand_r[3*p2])))+1
                    xline=np.linspace(hand_r[3*p1+1],hand_r[3*p2+1],dist).astype(int)
                    yline=np.linspace(hand_r[3*p1],hand_r[3*p2],dist).astype(int)
                    for kk,xx in enumerate(xline):
                        Mtx[BODY_DIM-HAND_DIM-1+ii%5,xx,yline[kk]]=1
                    ii+=1
            
    if ifhead:
         #-------------Bluring
        for ii in range(HEAD_DIM-1):
            head_mtx[ii,...]=cv2.GaussianBlur(head_mtx[ii,...],(3,3),0)
        head_mtx[HEAD_DIM-1,...]=np.any(head_mtx[:HEAD_DIM-1,...],0)
        head_mtx[HEAD_DIM-1,...]=1-head_mtx[HEAD_DIM-1,...]
            
        for _ in range(3):
            head_mtx[HEAD_DIM-1,...]=cv2.erode(head_mtx[HEAD_DIM-1,...],KERNEL_FACEEDGE)
        for _ in range(2):
            head_mtx[HEAD_DIM-1,...]=cv2.dilate(head_mtx[HEAD_DIM-1,...],KERNEL_FACEEDGE)
        #-------------Blur Background
        head_mtx[HEAD_DIM-1,...]=cv2.GaussianBlur(head_mtx[HEAD_DIM-1,...],(5,5),0)
    if ifbody:    
        for ii in range(BODY_DIM-HAND_DIM*2-1):
            Mtx[ii,...]=cv2.dilate(Mtx[ii,...],KERNEL)
        for ii in range(BODY_DIM-HAND_DIM*2,BODY_DIM-1):
            Mtx[ii,...]=cv2.dilate(Mtx[ii,...],KERNEL_HAND)
        
        Mtx[BODY_DIM-1,...]=np.any(Mtx[:BODY_DIM-1,...],0)
        Mtx[BODY_DIM-1,...]=1-Mtx[BODY_DIM-1,...]
        
    return Mtx,head_mtx,head_cent

def label2feature(datum,ifhead,ifbody):
    
    Mtx=np.zeros((BODY_DIM,BODY_SIZE,BODY_SIZE),dtype=np.float32)
    head_mtx=np.zeros((HEAD_DIM,HEAD_SIZE,HEAD_SIZE),dtype=np.float32)
    head_cent=np.array([0,0])  
    head_inside=False
    for jj in range(datum.poseKeypoints.shape[0]):
        face=datum.faceKeypoints[jj]# *70 points
        face[face>511]=511
        face=face.astype('int32')
        head_cent=np.array([face[30,0],face[30,1]],dtype=np.int)
        if ifhead:
            # ----Handeling Head Keypoints and Face extraction 

            if face is not None:
                n_face=face.__len__()
                for ii in range(0,n_face):
                    if face[ii,1]-head_cent[1]+HEAD_SIZE//2>=0 and\
                       face[ii,1]-head_cent[1]+HEAD_SIZE//2<HEAD_SIZE and\
                       face[ii,0]-head_cent[0]+HEAD_SIZE//2>=0 and\
                       face[ii,0]-head_cent[0]+HEAD_SIZE//2<HEAD_SIZE:
                        head_mtx[ii%14,face[ii,1]-head_cent[1]+HEAD_SIZE//2\
                                      ,face[ii,0]-head_cent[0]+HEAD_SIZE//2]=1
        if ifbody:
            #Handling pose and hand gesture
            pose=datum.poseKeypoints[jj]# *25 points
            hand_l=datum.handKeypoints[0][jj]# *21 points
            hand_r=datum.handKeypoints[1][jj]
            pose[pose>511]=511
            hand_l[hand_l>511]=511
            hand_r[hand_r>511]=511
            if pose is not None:
                #n_pose=pose.__len__()//3
                pose=pose.astype('int32')
                ii=0
                for p1,p2 in bodylink_list:
                    #dist=int(abs(complex(pose[p1,1]-pose[p2,1],pose[p1,0]-pose[p2,0])))+1
                    dist=int(max(abs(pose[p1,1]-pose[p2,1]),abs(pose[p1,0]-pose[p2,0])))+1
                    xline=np.linspace(pose[p1,1],pose[p2,1],dist).astype(int)
                    yline=np.linspace(pose[p1,0],pose[p2,0],dist).astype(int)
                    Mtx[ii,xline,yline]=1
                    ii=ii+1
            if hand_l is not None:
                hand_l=hand_l.astype('int32')
                ii=0
                for p1,p2 in handlink_list:
                    dist=int(abs(complex(hand_l[p1,1]-hand_l[p2,1],hand_l[p1,0]-hand_l[p2,0])))+1
                    #dist=int(max(abs(hand_l[p1,1]-hand_l[p2,1]),abs(hand_l[p1,0]-hand_l[p2,0])))+1
                    xline=np.linspace(hand_l[p1,1],hand_l[p2,1],dist).astype(int)
                    yline=np.linspace(hand_l[p1,0],hand_l[p2,0],dist).astype(int)
                    Mtx[BODY_DIM-HAND_DIM*2-1+ii%5,xline,yline]=1
                    ii+=1

            if hand_r is not None:
                hand_r=hand_r.astype('int32')
                ii=0
                for p1,p2 in handlink_list:
                    dist=int(abs(complex(hand_r[p1,1]-hand_r[p2,1],hand_r[p1,0]-hand_r[p2,0])))+1
                    #dist=int(max(abs(hand_r[p1,1]-hand_r[p2,1]),abs(hand_r[p1,0]-hand_r[p2,0])))+1
                    xline=np.linspace(hand_r[p1,1],hand_r[p2,1],dist).astype(int)
                    yline=np.linspace(hand_r[p1,0],hand_r[p2,0],dist).astype(int)
                    for kk,xx in enumerate(xline):
                        Mtx[BODY_DIM-HAND_DIM-1+ii%5,xx,yline[kk]]=1
                    ii+=1
        #-------------Gen Background
    if ifhead:
         #-------------Bluring
        for ii in range(HEAD_DIM-1):
            head_mtx[ii,...]=cv2.GaussianBlur(head_mtx[ii,...],(3,3),0)
        head_mtx[HEAD_DIM-1,...]=np.any(head_mtx[:HEAD_DIM-1,...],0)
        head_mtx[HEAD_DIM-1,...]=1-head_mtx[HEAD_DIM-1,...]
            
        for _ in range(3):
            head_mtx[HEAD_DIM-1,...]=cv2.erode(head_mtx[HEAD_DIM-1,...],KERNEL_FACEEDGE)
        for _ in range(2):
            head_mtx[HEAD_DIM-1,...]=cv2.dilate(head_mtx[HEAD_DIM-1,...],KERNEL_FACEEDGE)
        #-------------Blur Background
        head_mtx[HEAD_DIM-1,...]=cv2.GaussianBlur(head_mtx[HEAD_DIM-1,...],(5,5),0)
    if ifbody:    
        for ii in range(BODY_DIM-HAND_DIM*2-1):
            Mtx[ii,...]=cv2.dilate(Mtx[ii,...],KERNEL)
        
        for ii in range(BODY_DIM-HAND_DIM*2,BODY_DIM-1):
            Mtx[ii,...]=cv2.dilate(Mtx[ii,...],KERNEL_HAND)
        
        Mtx[BODY_DIM-1,...]=np.any(Mtx[:BODY_DIM-1,...],0)
        Mtx[BODY_DIM-1,...]=1-Mtx[BODY_DIM-1,...]
        
    return Mtx,head_mtx,head_cent
############--------------------------Loader Class---------------------------################

class CostumImFolder(Dataset):
    def __init__(self,
                img_root: List[str]=None,
                label_root: List[str]=None,
                target_root: List[str]=None,
                transforms: Optional[Callable]=None,
                transform: Optional[Callable]=None,
                target_transform: Optional[Callable]=None,
                loader=cv2_loader,
                ifbody=True,
                ifhead=False
                ) -> None:
        #for ii,root in enumerate(img_root):
        #    if isinstance(img_root,torch._six.string_classes):
        #        img_root[ii]=os.path.expanduser(root)
        #for ii,root in enumerate(label_root):
        #    if isinstance(label_root,torch._six.string_classes):
        #        label_root[ii]=os.path.expanduser(root)
            
        self.img_root = img_root
        self.label_root = label_root
        self.target_root=target_root
        self.ifbody=ifbody
        self.ifhead=ifhead
        self.loader= loader
        has_transforms = transforms is not None
        
        has_separate_transform = transform is not None  or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")
        self.transforms=transforms
        # for backwards-compatibility
        if transform is not None:
            self.transform = transform
        else:
            if ifbody and not ifhead:
                self.transform =transform_dict["body"]
            elif ifhead and not ifbody:
                self.transform =transform_dict["head"]
                self.target_transform=transform_dict["tgthead"]
            else:
                raise ValueError("Need to be only head or only body image")
        if self.target_root is None:
            self.samples_img,self.label,self.imname,_=make_dataset(self.img_root,self.label_root)
        else:
            self.samples_img,self.label,_,self.target_img=make_dataset(self.img_root,\
                                                                           self.label_root,self.target_root)
        
        
        if self.img_root is not None and len(self.samples_img) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.img_root)
            if IMG_EXTENSIONS is not None:
                msg += "Supported extensions are: {}".format(",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)
        if len(self.label) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.label_root)
            if IMG_EXTENSIONS is not None:
                msg += "Supported extensions are: {}".format(",".join(IMG_EXTENSIONS))
            raise RuntimeError(msg)
        
        
    def __getitem__(self,index:int) ->Any:
        
        lbl_path= self.label[index]
        if self.img_root is not None:
            im_path= self.samples_img[index]
            im_sample = self.loader(im_path)
            im_sample = self.transform(im_sample)
        else:
            im_sample=self.imname[index]
        im_target=None
        #print(im_path,tgt_path)
        if self.ifbody or self.ifhead:
            lbl_sample,head_mtx,head_center = read_label(lbl_path,self.ifhead,self.ifbody)
            if self.target_root is not None:
                tgt_path=self.target_img[index]
                im_target = self.loader(tgt_path)
                im_target = self.target_transform(im_target)
            else:
                im_target=0
            return im_sample,lbl_sample,head_mtx,head_center,im_target
            #return im_sample,lbl_sample,head_mtx,head_center,im_target,im_path,lbl_path,tgt_path
        else:
            raise ValueError("Need to be only head or only body image")
            
    def __len__(self) ->int:
        return len(self.label)
        
    def __repr__(self) ->str:
        head="Dataset"+self.__class__.__name__
        body= ["Data amount:{}".format(self.__len__())]
        if self.img_root is not None:
            body.append("im folder location:{}".format(self.img_root))
            
        if self.label_root is not None:
            body.append("im folder location:{}".format(self.label_root))
        body+=self.extra_repr().splitlines()
        if hasattr(self,"transforms") and self.transforms is not None:
            body +=[repr(self.transforms)]
        lines=[head]+[" " * 3+line for line in body]
        return "\n".join(lines)
    def _format_transform_repr(self,transform: Callable,head:str) ->List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])
    def extra_repr(self) -> str:
        return ""
    