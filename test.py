import torch
from tqdm import tqdm
from label_prediction import *
from myutils.plot import plot_dialogue_tree, save_dialogue
from analyse_results import accuracy
from PIL import Image  
from torchvision.transforms.functional import to_pil_image
from scipy.spatial.distance import cdist


def test(args, train_dataloader,test_dataloader,A,P):

    '''
        args: arguments given in entry of the code
        train_dataloader:
        test_dataloader:
        A: agent A
        P: agent P
    '''

    nb_att = len(args.attribute_names)
    nb_label = len(args.label_names)
    
    # things to save
    dial_pred = []
    vt = []
    att = []
    agentP_pred = []
    lengths = []
    strength = None
    all_img_link = []
    dialogs = []
    idxs = []

    
    # stop_value = 0
    with torch.no_grad():   
            
        for e, sample in enumerate(tqdm(test_dataloader)):
            #if e != 49:
            #    continue
            x_sim = sample["sim_img"]
            x_att = sample["att_img"]
            y = sample["target"]
            a = sample["attributes"]
            c_a = sample["att_certainty"]
            idx = sample["index"]
            idxs += idx
            img_id = sample["image_path"]
            all_img_link += img_id
             

            

            # create history of dialogue and agent knowledge
            h = [[[] for _ in range(x_sim.shape[0])],[[] for _ in range(x_sim.shape[0])],[[] for _ in range(x_sim.shape[0])]] 
            
            # advance dialogues until they stop all
            max_length = -1
            while max_length < max([len(h_x) for h_x in h[0]]):

                max_length = max([len(h_x) for h_x in h[0]])
                (h[0], h[2]) = P.answer(x_sim,h[0],h[2])
                (h[0], h[1]) = A.answer(x_att,h[0],h[1])

  
            dialogs += h[0] # save dialogues

            # predictions 
            
            cur_strength = dialogue_label_prediction(h[0],nb_label)
            cur_labels = torch.argmax(cur_strength,dim=1)

            # prediction with K-Nearest neighbors
            distances = P.d(h[2][1],P.p)
            K_nearest = P.new_proto.get_knn_batch(h[2][1],distances)
            cur_labels_P = torch.argmax(knn_label_prediction(K_nearest,P.y_p,nb_label),dim=1)
            

            # stock prediction
            agentP_pred += cur_labels_P
            dial_pred += cur_labels
            # score strenght for different classes
            if strength is None:
                strength = cur_strength
            else:
                strength = torch.cat([strength,cur_strength],dim=0)

            # store labels and attributes
            if len(y.shape) == 2:
                y = torch.argmax(y,dim=1)
            vt += y.tolist()
            att += a.tolist()

            # store lenghts of dialogue
            lengths += [len(h_x) for h_x in h[0]]

            if args.visualize == 'all':
                i_print = torch.ones(cur_labels.shape)
            elif args.visualize == 'error':
                i_print = torch.where(cur_labels==y,0,1)
            else:
                i_print = torch.zeros(cur_labels.shape)
            
            
           

            print("Accuracy:")
            print("\tdialogue:",float(accuracy(torch.tensor(dial_pred),torch.tensor(vt))))
            print("\tagent P (K-nearest neighbors):",float(accuracy(torch.tensor(agentP_pred),torch.tensor(vt))))
            
            # plot dialogues and save them
            for i in torch.flatten(torch.nonzero(i_print)):

                image_path = sample['image_path'][i]
                y = sample['target'][i]
                try: 
                    _ = len(y) # just verify if list or single element
                    y = torch.argmax(y)
                except:
                    _ = 1

                plot_dialogue_tree(args, h[0][i], train_dataloader.dataset,save_name = "images/"+str(e)+"_"+str(i.item())+"tree_dialogue.png",image_path = image_path, y = y)
                
                # if you want to save images
                # to_pil_image(args.sim_inverse_preprocess(x_sim[i])).save("images/"+str(e)+"_"+str(i.item())+"image_to_class_"+ str(idx[i.item()].item()) +".png")
                
                save_dialogue(args, h[0][i], "images/"+str(e)+"_"+str(i.item())+"dialogue.txt")

    results = {"dialogue prediction":torch.tensor(dial_pred), "VT": torch.tensor(vt), "agent P prediction": torch.tensor(agentP_pred), # predictions
                "lengths": lengths, "strength": strength, "image links": all_img_link, 
                "dialogs": dialogs, "indexes": idxs}
    return results