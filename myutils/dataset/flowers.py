import numpy as np
import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
import cv2
from torchvision.datasets import VisionDataset, Flowers102
from torchvision.datasets.folder import default_loader
from torchvision import transforms

import ast

    
class Flowers(VisionDataset):
  def __init__(self, root, att_file, train=True, sim_transform=None, att_transform=None, target_transform=None, download=False):
    '''
      root: root of the dataset
      att_file: file stocking the attributes
    '''
    super(Flowers, self).__init__(root, transform=sim_transform, target_transform=target_transform)
    self.loader = default_loader
    self.root = root
    self.base_folder = "flowers"

    self.sim_transform = sim_transform
    self.att_transform = att_transform

    if train:
       train = "train"
    else:
       train = "test"
       

    self.data = Flowers102(
        root=root,
        download=False,
        split=train
    )   

    
    self.label_names = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 
                       'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 
                       'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 
                       'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 
                       'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 
                       'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 
                       'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 
                       'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 
                       'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 
                       'buttercup', 'oxeye daisy','common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 
                       'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia', 
                       'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 
                       'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 
                       'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 
                       'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 
                       'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 
                       'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 
                       'trumpet creeper', 'blackberry lily']
    self.get_attribute_names(att_file)
    
    
  def get_attribute_names(self, att_file):
    '''
      att_file: file containing attributes
    '''
    self.atts_by_label = dict()
    self.attribute_names = []
    with open(att_file,"r") as gfg_file:
      ini_list = gfg_file.read()

    
    # Converting string to list
    res = ast.literal_eval(ini_list)

    for x in res:
       label_n, attributes = x.split(":")

       self.atts_by_label[label_n] = attributes.split(", ")
       self.attribute_names += attributes.split(", ") 

  # get path of an image
  def get_path(self,i):
        return self.data._image_files[i].__str__()

  def __getitem__(self, index):

    img, target = self.data[index]
    certainty = torch.ones(len(self.attribute_names))*4
    atts = torch.zeros(len(self.attribute_names))
    true_atts = self.atts_by_label[self.label_names[target]]
    for a in true_atts:
       atts[self.attribute_names.index(a)] = 1


    sample = { 
            'sim_img' : self.sim_transform(img),
            'att_img' : self.att_transform(img),
            'target': target,
            'attributes': atts,
            'att_certainty': certainty,
            'index': index,
            'image_path': self.data._image_files[index].__str__()
        }
    return sample

  def __len__(self):
    return len(self.data)