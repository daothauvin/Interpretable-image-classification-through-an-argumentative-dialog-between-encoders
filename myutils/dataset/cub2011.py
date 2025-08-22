import os
from PIL import Image
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive
import torch
import sys
import re
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class MyCub2011(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, sim_transform=None, att_transform=None, target_transform=None, download=False):
        super(MyCub2011, self).__init__(root, transform=sim_transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        self.sim_transform = sim_transform
        self.att_transform = att_transform
        
        
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')


        class_names = pd.read_csv(os.path.join(self.root, 'attributes.txt'),
                                  sep=' ', names=['att_name'], usecols=[1])

        self._load_metadata()
        self.attribute_names = class_names['att_name'].to_list()

        self.stats_att = torch.zeros((len(self.label_names),len(self.attribute_names)))
 
        with open(os.path.join(self.root, 'CUB_200_2011','attributes', 'class_attribute_labels_continuous.txt'),"r") as f:
            for i,l in enumerate(f):
                for j,v in enumerate(l.split()):
                    self.stats_att[i,j] = float(v)
        
        # fuse attributes that appears for same classes 
        
        self.to_fuse = dict()
        # If two concepts always present/absent on the same class, set their values alike
        for i in range(len(self.attribute_names)):
            for j in range(len(self.attribute_names)):
                if i < j and self.attribute_names[i].split("::")[0] == self.attribute_names[j].split("::")[0]:
                    
                    # search if always the same value
                    same_v = torch.logical_or(
                        torch.logical_and(self.stats_att[:,i] > 10.,self.stats_att[:,j] > 10.),
                        torch.logical_and(self.stats_att[:,i] <= 10.,self.stats_att[:,j] <= 10.)
                    )
                    
                    if torch.sum(same_v) == same_v.shape[0]:
                        # add to fuse
                        if self.to_fuse.get(i) is None:
                            self.to_fuse[i] = [j]
                        else:
                            self.to_fuse[i].append(j)
        
        
        # get attribute values and uncertainty
        with open(os.path.join(self.root, 'CUB_200_2011','attributes', 'image_attribute_labels.txt'),"r") as f:
            atts_d = dict()
            for l in f: # for a data
                l = re.sub(' +', ' ',l)
                sample = [int(x) for x in l.split()[:4]]
                att_id = sample[1]-1
                # set attributes value of the data
                if atts_d.get(sample[0]) is None: 
                    atts_d[sample[0]] = [[0 for _ in range(len(self.attribute_names))],[0 for _ in range(len(self.attribute_names))]]
                atts_d[sample[0]][0][att_id] = sample[2] # binary value
                if sample[2]:
                    # change to fuse attributes
                    for a in self.to_fuse.get(sample[1]-1,[]):
                        atts_d[sample[0]][0][a] = 1
                atts_d[sample[0]][1][att_id] = sample[3] # attributes certainty

        self.atts = pd.DataFrame([[k] + v for k,v in atts_d.items()],columns=['img_id','atts','m'])
        self.data = self.data.merge(self.atts, on='img_id')

    def _load_metadata(self):
        # images
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        # labels
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])

        # train test split
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        # merge everything
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')


        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        
                                  
        self.label_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        
    
    def get_path(self,i):
        sample = self.data.iloc[i]
        return os.path.join(self.root, self.base_folder, sample.filepath)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                return False
        return True
    
    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)

        target_num = (sample.target - 1).item()  # Targets start at 1 by default, so shift to 0

        target = torch.zeros(len(self.label_names),dtype=torch.float)
        target[target_num] = 1

        img = self.loader(path)

        atts = torch.tensor(sample.atts,dtype=torch.float)

        certainty = torch.tensor(sample.m,dtype=torch.float)

        # remove attributes with less than 5% of chance of occuring for class     
        # atts[self.stats_att[target_num] < 5.0] = 0
        # add attributes with more than 80% of chance of occuring for class
        # atts[self.stats_att[target_num] > 80.0] = 1
        # atts[217:221] = 0 # remove size (not really visible in the image)
        
        sample = { 
            'sim_img' : self.sim_transform(img),
            'att_img' : self.att_transform(img),
            'target': target,
            'attributes': atts,
            'att_certainty': certainty,
            'index': idx,
            'image_path': path
        }

        return sample

