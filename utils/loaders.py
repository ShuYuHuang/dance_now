import torch
from torchvision import transforms as ts
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import StandardTransform
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import numpy as np
from typing import Any, Callable, List, Optional, Tuple
import json
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',".json"
]
transform_dict = {
        'src': ts.Compose(
        [ts.Resize((512,512)),
         ts.ToTensor(),
         ts.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ts.Lambda(lambda x: (x-x.min())/(x.max()-x.min()))
         ])}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(diri,dirl):
    images = []
    label=[]
    for ii,dir in enumerate(diri):
        assert os.path.isdir(dir), '%s is not a valid dirsectory' % dir
        #print(os.walk(dir))
        for rootl,_,_ in os.walk(dirl[ii]):
            break
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname) and not "checkpoint" in fname:
                    path = os.path.join(root, fname)
                    images.append(path)
                    path2=os.path.join(rootl,f"{fname[:-4]}_keypoints.json")
                    label.append(path2)
                else:
                    print(f"{fname} not imported")

    return images,label
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def read_label(fname):

    KERNEL=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    with open(fname) as f:
        data = json.load(f)
    Mtx=np.zeros((512,512))
    for jj in range(len(data["people"])):
        face=np.array(data["people"][jj]["face_keypoints_2d"])
        pose=np.array(data["people"][jj]["pose_keypoints_2d"])
        hand_l=np.array(data["people"][jj]["hand_left_keypoints_2d"])
        hand_r=np.array(data["people"][jj]["hand_right_keypoints_2d"])

        face[face>511]=511
        pose[pose>511]=511
        hand_l[hand_l>511]=511
        hand_r[hand_r>511]=511
        kk=0
        if face is not None:
            n_face=face.__len__()//3
            face=[ int(x) for x in face ]
            kk+=1
            for ii in range(n_face):
                Mtx[face[3*ii+1],face[3*ii]]=kk
        if pose is not None:
            n_pose=pose.__len__()//3
            pose=[ int(x) for x in pose ]
            for ii in range(n_pose):
                kk+=1
                Mtx[pose[3*ii+1],pose[3*ii]]=kk

        if hand_l is not None:
            n_lhand=hand_l.__len__()//3
            hand_l=[ int(x) for x in hand_l ]
            kk+=1
            for ii in range(n_lhand):
                Mtx[hand_l[3*ii+1],hand_l[3*ii]]=kk

        if hand_r is not None:
            n_rhand=hand_r.__len__()//3
            hand_r=[ int(x) for x in hand_r ]
            kk+=1
            for ii in range(n_rhand):
                Mtx[hand_r[3*ii+1],hand_r[3*ii]]=kk
            
    return np.expand_dims(cv2.dilate(Mtx,KERNEL),0)


class CostumImFolder(Dataset):
    def __init__(self,
                img_root: List[str],
                target_root: List[str],
                transforms: Optional[Callable]=None,
                transform: Optional[Callable]=transform_dict["src"],
                target_transform: Optional[Callable]=None,
                loader=default_loader,
                ) -> None:
        #for ii,root in enumerate(img_root):
        #    if isinstance(img_root,torch._six.string_classes):
        #        img_root[ii]=os.path.expanduser(root)
        #for ii,root in enumerate(target_root):
        #    if isinstance(target_root,torch._six.string_classes):
        #        target_root[ii]=os.path.expanduser(root)
            
        self.img_root = img_root
        self.target_root = target_root
        
        has_transforms = transforms is not None
        has_separate_transform = transform is not None  or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")
        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform
        self.loader= loader
        #if has_separate_transform:
        #    transforms = StandardTransform(transform, target_transform)
        self.transforms=transforms
        samples_img,target=make_dataset(self.img_root,self.target_root)
        if len(samples_img) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.img_root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        if len(target) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.target_root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        self.samples_img=samples_img
        self.target=target
        
        
    def __getitem__(self,index:int) ->Any:
        im_path= self.samples_img[index]
        tgt_path= self.target[index]
        
        im_sample = self.loader(im_path)
        tgt_sample = read_label(tgt_path)
        if self.transforms is not None:
            im_sample = self.transforms.transform(im_sample)
            tgt_sample=self.transforms.target_transform(tgt_sample)
        elif self.target_transform is not None and self.transform is not None:
            im_sample = self.transform(im_sample)
            tgt_sample=self.target_transform(tgt_sample)
        elif self.transform is not None:
            im_sample = self.transform(im_sample)
            
            #tgt_sample=self.transform(tgt_sample)
            
            
        return im_sample, tgt_sample , im_path , tgt_path

            
            
    def __len__(self) ->int:
        return len(self.samples_img)
        
    def __repr__(self) ->str:
        head="Dataset"+self.__class__.__name__
        body= ["Data amount:{}".format(self.__len__())]
        if self.img_root is not None:
            body.append("im folder location:{}".format(self.img_root))
            
        if self.target_root is not None:
            body.append("im folder location:{}".format(self.target_root))
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
    