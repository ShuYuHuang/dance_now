import sys
from pathlib import Path
sys.path.append(str(Path("../")))
from utils import create_head
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

NUM_GPU=torch.cuda.device_count()
head_dataset=create_head.create(["../data/girl/train_img/"],
                                 ["../data/girl/train_label/"],
                               "../data/girl/train_headimg/")
head_loader=DataLoader(head_dataset, batch_size=16,\
                        shuffle=False,num_workers = 4*NUM_GPU,pin_memory=True)

for head_array,head_center in head_loader:
    continue 