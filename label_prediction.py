import torch
def dialogue_label_prediction(ds,nb_label):
    '''
        Predict a label according to the dialogue
        ds: dialogues
        nb_label: number of labels
        Resumts: A score for each class
    '''
    pred = torch.zeros((len(ds),nb_label)) - 1000 # set value of not proposed labels to -1000
    more = 0.01 # add small value to prioritize the first proposed labels
    for i, h in enumerate(ds):
        l = None
        for arg in h:
            # search for new label
            if arg[1] == "Propose":

                # change current label
                l = arg[2][1]
                # set its value to 0 instead of -1000
                pred[i][l] =0.0
                # add the small value
                pred[i][l] += more
                more -= 0.001 # reduce the value for new classes
            # count the number of arguments
            if arg[1] == "Argue" and arg[0][0] == 'A':
                pred[i][l] -= 1
            elif arg[1] == "Argue" and arg[0][0] == 'B':
                pred[i][l] += 1
            elif arg[1] == "Argue":
                raise 
            
    return pred

def knn_label_prediction(K_nearest,labels,nb_label):
    '''
        Count the number of classes in the K nearest
        K_nearest: the list of the K_nearest prototypes for each input x
        labels: the labels of the protoptyes
        nb_label: the number of prototypes
        Results: A score for each class

    '''
    pred = torch.zeros((len(K_nearest),nb_label))
    for i in range(K_nearest.shape[0]):
        for j in K_nearest[i]:
            pred[i][int(labels[int(j)])] += 1
    return pred
