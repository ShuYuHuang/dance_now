from utils import loaders,model_fun
from torch.utils.data import DataLoader


train_set=loaders.CostumImFolder(["./data/anime/train_img/"],
                                 ["./data/anime/train_label/"])
train_loader=DataLoader(train_set, batch_size=16, shuffle=True)

print(train_set.transform)
print(train_set.target_transform)

#-------------------------------------Model Building---------------------------------
import torch
import torch.nn as nn
torch.cuda.empty_cache()
criterion = nn.MSELoss()
G=model_fun.GlobalGenerator(input_nc=137,output_nc=3,n_blocks=5).cuda()
opt_G = torch.optim.Adam(G.parameters(), lr=1e-5, betas=(0.5, 0.999))
model_fun.weights_init(G)
G.train()
                                 
                                 
#-------------------------------------Model Training---------------------------------
import tqdm 
#import matplotlib.pyplot as plt

for epoch in  tqdm.tqdm(range(10)) :
    for in_img,tgt_stick,cc,dd in train_loader:
        torch.cuda.empty_cache()
        out_img=G(tgt_stick.cuda())
        torch.cuda.empty_cache()

        # optimizer

        loss_G = criterion(in_img,out_img.cuda())

        G.zero_grad()
        loss_G.backward()
        opt_G.step()
    print(f"epoch {epoch} over, loss= {loss_G}")
    
    print(out_stick.shape)
    print(in_img.shape)
    print(tgt_stick.shape)
    torch.save(G.state_dict(), f"./skeleton_model_{epoch}.pt")
