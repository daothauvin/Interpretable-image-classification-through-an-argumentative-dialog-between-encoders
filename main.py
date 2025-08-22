import argparse
import yaml
import numpy as np
import torch
import random
import torch.nn as nn
import os
from scipy.spatial.distance import cdist
from torchvision import transforms

from myutils.init_dataloader import get_dataloader
from myutils.save_features import save_features

from agentA import AgentA
from agentP import AgentP
from myutils.agentA.attribute_selection.basic_att_selection import Attribute_Selector
from myutils.agentP.counter_att.basic_counter_att import BasicCounterAttack
from myutils.agentP.label_selector.basic_label_selection import *
from myutils.agentP.proto_selector.basic_proto_selector import BasicProtoSelection

from analyse_results import accuracy

from test import test
from analyse_results import *


def parse_args():
    # fmt: off

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")    
    parser.add_argument("--config-file", type=str, default="",
        help="File containing configuration")
    parser.add_argument("--dataset", type=str, default="CUB",
        help="dataset to use (CUB or flowers)")
    parser.add_argument("--batch-size", type=int, default=20,
        help="batch size for similarity and attribute encoders")
    parser.add_argument("--similarity-encoder", type=str, default="DINO",
        help="encoder of agent P (DINO, DINOv2)")
    parser.add_argument("--attribute-encoder", type=str, default="CLIP",
        help="encoder of agent A (CLIP, CLIPFT: a CLIP finetuned for attribute detection on CUB)")
    
    parser.add_argument("--features", type=str, default="features.pt",
        help="file to save/load prototypes features encoded wih similarity and attribute encoder")
    
    parser.add_argument("--compute-features", type=bool, default=False,
        help="to compute new features (even if already saved)")
    
    parser.add_argument("--data-path", type=str, default="",
        help="repertory to load data")
    
    parser.add_argument("--visualize", type=str, default='',
        help="to save dialogues visualisation (all => every data, error => only missclassified data)")
    parser.add_argument("--save-name", type=str, default="results/results.pt",
        help="file name to save metrics")
    parser.add_argument("--thresholds", type=list, default=[10,10],
        help="thresholds for attribute detection")
    
    parser.add_argument("--K", type=int, default=5,
        help="Number of similar prototypes")
    
    parser.add_argument("--distance-metrics",type=list, default="cosine", # "euclidean", "canberra", "sqeuclidean", "jensenshannon", "cosine","cityblock", "correlation"
        help = "distance to use between encodings (see scipy.spatial.distance documentation)")

    args = parser.parse_args()

    class DotDict(dict):
        """dot.notation access to dictionary attributes (Thomas Robert)"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    # config file  modifications
    if args.config_file is not None:
        with open(args.config_file, 'r') as stream:
            opt = yaml.load(stream,Loader=yaml.Loader)
        conf = DotDict(opt)
    for k,v in conf.items():
        setattr(args, k.replace("-","_"), v)
    # fmt: on
    return args



if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)


    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    setattr(args, "device", device)
    setattr(args, "dir_path", os.path.dirname(os.path.realpath(__file__)))
    
    if not os.path.exists(os.path.dirname(args.save_name)):
        os.makedirs(os.path.dirname(args.save_name))

    '''
        Select similarity encoder and set similarity encoder preprocessing 
        We also set: 
            sim_num_ftrs -> the output size of the encoder
    '''
    if args.similarity_encoder == "DINO":
        setattr(args, "sim_num_ftrs",  384)#328)#768)
        image_size = 256 
        sim_enc = torch.hub.load('facebookresearch/dino:main', 'dino_vits8') 
    if args.similarity_encoder == "DINOv2":
        setattr(args, "sim_num_ftrs", 768) 
        image_size = 266 
        sim_enc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    sim_preprocess = transforms.Compose([
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])

    sim_inverse_preprocess = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    setattr(args,"sim_inverse_preprocess",sim_inverse_preprocess)
    setattr(args,"sim_preprocess",sim_preprocess)


    # multi-gpu and cuda
    sim_enc = nn.DataParallel(sim_enc)
    sim_enc = sim_enc.eval()
    sim_enc = sim_enc.to(device)

    '''
        Set attribute encoder preprocessing
    '''
    if args.attribute_encoder == "CLIP":
        from myutils.agentA.attribute_selection.attribute_predictors.attribute_clip import CLIP_Att_Detector
        att_preprocess = CLIP_Att_Detector.get_preprocess()

    elif args.attribute_encoder == "CLIPFT":
        from myutils.agentA.attribute_selection.attribute_predictors.attribute_clip_ft import CLIPFT_Att_Detector
        att_preprocess = CLIPFT_Att_Detector.get_preprocess()

    setattr(args,"att_preprocess",att_preprocess)


    train_dataloader, test_dataloader = get_dataloader(args)
    
        
    # Select attribute encoder

    if args.attribute_encoder == "CLIP":
        att_enc = CLIP_Att_Detector(args)
    elif args.attribute_encoder == "CLIPFT":
        att_enc = CLIPFT_Att_Detector(args)

    # Load or save prototype features
    features_path = os.path.join("features",args.features)

    if not args.compute_features and os.path.isfile(features_path):
        train_data = torch.load(features_path, weights_only=False)
    else:
        train_data = save_features(train_dataloader, args ,sim_enc, att_enc)



    # init agent A
    a_s = Attribute_Selector(args)

    a_s.init_thresholds(train_data["attributes"], train_data['att_latent_space'])

    A = AgentA(args, train_data["attributes"], train_data['att_latent_space'],a_s, att_enc)



    # define distance between encodings
    def dist(x,x_i,metric,**kwargs): 
        return torch.from_numpy(cdist(x,x_i,metric=metric,**kwargs))
    distance = lambda x,y: dist(x,y,args.distance_metrics) 


    # init agent P
    c_a = BasicCounterAttack(args)
    p_a = BasicLabelSelection(args)
    s_a = BasicProtoSelection(args)
    P = AgentP(train_data["sim_latent_space"], train_data["label"],train_data["attributes"],c_a,s_a,p_a, sim_enc, distance)

    # test procedure
    results = test(args, train_dataloader,test_dataloader,A,P)

    print("Accuracy")
    print("\tdialogue:",float(accuracy(results["dialogue prediction"],results["VT"])))
    print("\tK-NN:",float(accuracy(results["agent P prediction"],results["VT"])))

    torch.save(results,args.save_name)








