from tqdm import tqdm
import torch
import os

def save_features(train_dataloader, args, sim_enc = None, att_enc = None):
    '''
        train_dataloader: dataloader containing data to encode
        args: arguments given in entry of the code
        sim_enc: similarity encoder
        att_enc: attributes encoder
        Return: dictionnary containing the features encoded with sim_enc (latent_space), its label, its attributes 
    '''
    if not os.path.exists("features"):
            os.makedirs("features")
    print('Save Prototype Encodings...')
    nb_atts = len(args.attribute_names)

    train_data = dict()
    train_data["sim_latent_space"] = torch.zeros((len(train_dataloader.dataset),args.sim_num_ftrs))
    train_data["att_latent_space"] = torch.zeros((len(train_dataloader.dataset),nb_atts))
    train_data["label"] = torch.zeros(len(train_dataloader.dataset))
    train_data["attributes"] = torch.zeros((len(train_dataloader.dataset),nb_atts))
    train_data["indexe"] = torch.zeros(len(train_dataloader.dataset))
    train_data['attributes_certainty'] = torch.zeros((len(train_dataloader.dataset),nb_atts))
    

    cur_id = 0
    for sample in tqdm(train_dataloader):
        x_s = sample["sim_img"]
        x_a = sample["att_img"]
        y = sample["target"]
        a = sample["attributes"]
        a_c = sample["att_certainty"]
        if sim_enc is not None:
          with torch.no_grad():
                train_data["sim_latent_space"][cur_id:cur_id+x_s.shape[0]] = sim_enc(x_s.to(args.device)).cpu()
        if att_enc is not None:
            with torch.no_grad():
                train_data["att_latent_space"][cur_id:cur_id+x_a.shape[0]] = att_enc(x_a.to(args.device), args.attribute_names).cpu()

        if len(y.shape) == 2:
                y = torch.argmax(y,dim = 1)
        train_data["label"][cur_id:cur_id+x_s.shape[0]] = y
        train_data["indexe"][cur_id:cur_id+x_s.shape[0]] = sample["index"]
        train_data["attributes"][cur_id:cur_id+x_s.shape[0]] = a
        train_data["attributes_certainty"][cur_id:cur_id+x_s.shape[0]] = a_c

        cur_id += x_s.shape[0] 
    torch.save(train_data,os.path.join("features",args.features))
    return train_data