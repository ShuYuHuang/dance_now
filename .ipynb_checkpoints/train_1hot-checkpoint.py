
from utils import loaders,model_fun
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import numpy as np
#-------------------------------------Loader Building---------------------------------

train_set=loaders.CostumImFolder(["./data/anime/train_img/"],
                                 ["./data/anime/train_label/"])
train_loader=DataLoader(train_set, batch_size=8, shuffle=True)

print(train_set.transform)
print(train_set.target_transform)

#-------------------------------------Model Building---------------------------------
GPU_ID=[0]
torch.cuda.empty_cache()
big_model=model_fun.Pix2PixHDModel(28+1,3,GPU_ID)                    
                                 
#-------------------------------------Model Training---------------------------------

import tqdm 
#import matplotlib.pyplot as plt
EPOCHS=40
NITER=20
best_loss_D=np.Inf
for epoch in  range(EPOCHS) :
    for in_img,tgt_stick,cc,dd in train_loader:
        ############### Forward ####################
        torch.cuda.empty_cache()
        losses, out_img = big_model(Variable(tgt_stick).cuda(GPU_ID[0]), Variable(torch.tensor(0)).cuda(GPU_ID[0]), 
            Variable(in_img).cuda(GPU_ID[0]), Variable(torch.tensor(0)).cuda(GPU_ID[0]), infer=False)
        
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        
        loss_dict = dict(zip(big_model.loss_names, losses))

        
        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)
        
        ############### Backward Pass ####################
        # update generator weights
        torch.cuda.empty_cache()
        big_model.optimizer_G.zero_grad()
        loss_G.backward()
        big_model.optimizer_G.step()
        torch.cuda.empty_cache()
        # update discriminator weights
        big_model.optimizer_D.zero_grad()
        loss_D.backward()
        big_model.optimizer_D.step()
        
    if epoch > NITER:
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
    if epoch<3:
        torch.save(big_model.state_dict(), f"./GAN_run{epoch}.pt")
    elif loss_D.detach().cpu().numpy()<best_loss_D:
        best_loss_D=loss_D.detach().cpu().numpy()
        torch.save(big_model.state_dict(), f"./GAN_run{epoch}.pt")
