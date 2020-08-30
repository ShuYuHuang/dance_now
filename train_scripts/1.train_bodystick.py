import sys,os
from pathlib import Path
sys.path.append(str(Path("../")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils import loaders,model_body


NUM_GPU=torch.cuda.device_count()
EPOCHS=500
NITER=20

IMG_ROOT="../data/girl"
OUT_DIR="../model_body"
os.makedirs(OUT_DIR, exist_ok = True)
#-------------------------------------Loader Building---------------------------------
train_set=loaders.CostumImFolder([IMG_ROOT+"/train_img/"],
                                 [IMG_ROOT+"/train_label/"],ifbody=True,ifhead=False)

train_loader=DataLoader(train_set, batch_size=8, shuffle=True,num_workers = 4*NUM_GPU,pin_memory=True)

print(train_set.transform)
GAN_DIM=24+5+5+1
HEAD_GAN_DIM=14+1

#-------------------------------------Model Building---------------------------------
big_model=model_body.Pix2PixHDModel(GAN_DIM,3).cuda()
#-------------------------------------Model Training---------------------------------
# big_model.load_state_dict(torch.load("../model_body/GANbody_S_run10.pt"))
torch.save(big_model.netG.module,f"{OUT_DIR}/netGbody_struct.pth")
torch.save(big_model.netD.module,f"{OUT_DIR}/netDbody_struct.pth")


for epoch in  range(EPOCHS) :
    for in_img,lbl_sample,_,_,_ in train_loader:
        ############### Forward ####################
        losses, out_img = big_model(torch.tensor(lbl_sample,dtype=torch.float32, device=torch.device('cuda:0'))
                                  ,torch.tensor(in_img,dtype=torch.float32, device=torch.device('cuda:0'))
                                  , torch.tensor(0,dtype=torch.float32, device=torch.device('cuda:0')), infer=False)
        
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        
        loss_dict = dict(zip(big_model.loss_names, losses))

        
        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = (loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0))
        
        ############### Backward Pass ####################
        # update generator weights
        big_model.optimizer_G.zero_grad()
        loss_G.backward()
        big_model.optimizer_G.step()
        
        
        # update discriminator weights
        big_model.optimizer_D.zero_grad()
        loss_D.backward()
        big_model.optimizer_D.step()
        
    if epoch > NITER and epoch%50==49:
        big_model.update_learning_rate()
        
    print(f"epoch {epoch}/{EPOCHS} over, loss_G,loss_D= {loss_G},{loss_D}")
    #plt.figure(figsize=(20,40))
    #plt.subplot(1,3,1)
    #plt.imshow(in_img[0,...].detach().cpu().numpy().transpose(1,2,0))
    #plt.subplot(1,3,2)
    #plt.imshow(tgt_stick[0,...].detach().cpu().numpy().transpose(1,2,0))
    #plt.subplot(1,3,3)
    #plt.imshow(y[0,...].detach().cpu().numpy().transpose(1,2,0))
    #plt.show()
    if epoch%10==9:
        torch.save(big_model.state_dict(), f"{OUT_DIR}/GANbody_G_run{epoch+1}.pt")
        torch.save(big_model.netG.module.state_dict(), f"{OUT_DIR}/netGbody_G_run{epoch+1}.pt")
        torch.save(big_model.netD.module.state_dict(), f"{OUT_DIR}/netDbody_G_run{epoch+1}.pt")
    elif epoch<10:
        torch.save(big_model.state_dict(), f"{OUT_DIR}/GANbody_G_run{epoch+1}.pt")
