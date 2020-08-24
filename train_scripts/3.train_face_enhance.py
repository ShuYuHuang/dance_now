import sys,os
from pathlib import Path
sys.path.append(str(Path("../")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils import loaders,model_face

NUM_GPU=torch.cuda.device_count()
EPOCHS=500
NITER=20
OUT_DIR="../model_face"
#-------------------------------------Loader Building---------------------------------

train_set=loaders.CostumImFolder(["../rslt/anime/fake_headimg/"],
                                 ["../data/anime/train_label/"],
                                 ["../data/anime/train_headimg/"],ifbody=False,ifhead=True)

train_loader=DataLoader(train_set, batch_size=10, shuffle=True,num_workers = 4*NUM_GPU,pin_memory=True)

print(NUM_GPU)
print(train_set.transform)
GAN_DIM=24+5+5+1
HEAD_GAN_DIM=14+1

#-------------------------------------Model Building---------------------------------
big_model=model_face.Pix2PixHDModel(HEAD_GAN_DIM,3).cuda()
#-------------------------------------Model Training---------------------------------
os.makedirs(f"{OUT_DIR}/", exist_ok = True)
torch.save(big_model.facenetG.module,f"{OUT_DIR}/netGface_struct.pth")
torch.save(big_model.facenetD.module,f"{OUT_DIR}/netDface_struct.pth")


for epoch in  range(EPOCHS) :
    for in_img,_,head_mtx,_,tgt_img  in train_loader:
        ############### Forward ####################
        
        losses, out_img = big_model(torch.tensor(head_mtx, device=torch.device('cuda:0'))
                                  ,torch.tensor(in_img, device=torch.device('cuda:0'))
                                  ,torch.tensor(tgt_img, device=torch.device('cuda:0')), infer=False)
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(big_model.loss_names, losses))
        
        # calculate final loss scalar
        loss_D = (loss_dict['D_face_fake'] + loss_dict['D_face_real']) * 0.5
        
        loss_G = loss_dict['G_GAN_Face']+loss_dict['G_VGG_face']

    if epoch > NITER and epoch%20==19:
        big_model.update_learning_rate()
        
    print(f"epoch {epoch}/{EPOCHS} over, loss_G,loss_D= {loss_G},{loss_D}")

    if epoch%10==9:
        torch.save(big_model.state_dict(), f"{OUT_DIR}/GAN_run{epoch+1}.pt")
        torch.save(big_model.facenetG.module.state_dict(), f"{OUT_DIR}/netGface_run{epoch+1}.pt")
        torch.save(big_model.facenetD.module.state_dict(), f"{OUT_DIR}/netDface_run{epoch+1}.pt")
    elif epoch<10:
        if epoch==4:
            torch.save(big_model.facenetG.module.state_dict(), f"{OUT_DIR}/netGface_run{epoch+1}.pt")
            torch.save(big_model.facenetD.module.state_dict(), f"{OUT_DIR}/netDface_run{epoch+1}.pt")
        torch.save(big_model.state_dict(), f"{OUT_DIR}/GAN_run{epoch+1}.pt")
