import torch
class BasicCounterAttack:
    '''
        Class to select when to counter an attribute with a prototype
    '''
    def __init__(self, args):
        '''
            args: arguments given in entry of the code
        '''
        self.K = args.K # number of similar prototypes

    def __call__(self, x_a, p_a, distances):
        '''
            x_a: attribute to counter for the image to classify
            p_a: attribute to counter for the prototypes
            distances: distance between the image to classify and prototypes
            Return: The prototype to counter argument with (-1 if no counter)
        '''

        # get K nearest
        K_nearest = torch.argsort(distances)[:self.K]
        # get the attribute of the K nearest
        attributes = p_a[K_nearest]

        # observe if the attribute is present/absent in every similar prototypes

        if sum(attributes) == self.K*(1-x_a):
            return K_nearest[0].item()
        return -1