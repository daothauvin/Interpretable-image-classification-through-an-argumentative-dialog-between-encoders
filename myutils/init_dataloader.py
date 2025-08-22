from .dataset.cub2011 import MyCub2011 
from .dataset.flowers import Flowers
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as T
import os
import pandas as pd


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class DivideBy255(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 255.0
    
def get_names(args):
    class_names = pd.read_csv(os.path.join(args.data_path, 'attributes.txt'),
                                  sep=' ', names=['att_name'], usecols=[1])
    attribute_names = class_names['att_name'].to_list()

    class_names = pd.read_csv(os.path.join(args.data_path, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
                                  
    label_name = class_names['class_name'].to_list()
    return attribute_names, label_name
    
def get_dataloader(args):
    '''
     Load dataloaders
    '''
    dataset_name = args.dataset

    sim_preprocess = args.sim_preprocess
    att_preprocess = args.att_preprocess

    if dataset_name == "flowers":
        train_dataset = Flowers(args.data_path, args.attribute_file,  train=True, download=False,sim_transform=sim_preprocess, att_transform = att_preprocess) 
        test_dataset = Flowers(args.data_path, args.attribute_file, train=False, download=False,sim_transform=sim_preprocess, att_transform = att_preprocess) 

    if dataset_name == "CUB":

        train_dataset = MyCub2011(args.data_path, train=True, download=False,sim_transform=sim_preprocess, att_transform = att_preprocess) 
        test_dataset = MyCub2011(args.data_path, train=False, download=False,sim_transform=sim_preprocess, att_transform = att_preprocess) 

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 6)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 6) 
    # set label names and attribute names
    setattr(args,"label_names",train_dataset.label_names)
    setattr(args,"attribute_names",train_dataset.attribute_names)

    return train_dataloader, test_dataloader