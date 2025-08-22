class AgentA:
    def __init__(self,args,a_p, a_e, a_s, f_a): 
        super().__init__()
        '''
            args: arguments given in entry of the code
            a_p: attributes of prototypes. 
            a_e: attribute encoding of prototypes
            a_s: attribute selection (should return -1 if no selection)
            f_a: attributes dectector
        '''

        self.att_names = args.attribute_names 
        self.args = args
        self.a_s = a_s
        self.f_a = f_a
        self.a_p = a_p
        self.a_e = a_e
        
        


    def answer(self,xs,h,k, a_xs = None):
        '''
            xs: Images to classify
            h: Dialogue history
            k: Contains commitment stores and predicted attributes of the agent
            a_xs: attributes detection on the images to classify
            Return: updated history and knowledge with the answer of the agent
        '''

        # If start of the dialogue, predict attributes and add them to knowledge
        if len(k[0]) == 0:
            if a_xs is None:
                a_xs = self.f_a(xs,self.att_names)
            k = (k,a_xs)

        # for each dialogue
        for i, _ in enumerate(xs):
            # if last move a propose
            if h[i][-1][1] == "Propose": 
                h[i] += [("A","Why-Propose",h[i][-1][-1])]
                continue

            # Otherwise, find last argument about a class y
            last_arg = None
            y = None
            
            for m in h[i][::-1]: 
                if m[1] == "Argue" and m[-2][0] == "x is":
                    last_arg = m
                    y = m[-2][1]
                    break
                    
            if last_arg is None:
                raise Exception("No argue found")


            # get prototype similar according to the other agent

            p = None
            for prem in last_arg[-1]:
                if prem[0] == "x is sim to":
                    p = prem[1]
                    break

            if p is None:
                raise Exception("No prototype found")
            # search for an attribute to counter argument
            a = self.a_s(k[1][i], self.a_p[p], self.a_e[p], k[0][i])

            # If found a difference
            if a>=0:
                v_a_p = self.a_p[p,a]
                # If attribute present in prototype
                if v_a_p:
                    h[i] += [("A","Argue",("x is not",y),[("x has not",a), ("has",[p,a]),("is",[p,y])])]
                    k[0][i] += [("x has not",a)]
                # If attribute not present in prototype
                else:
                    h[i] += [("A","Argue",("x is not",y),[("x has",a), ("has not",[p,a]),("is",[p,y])])]
                    k[0][i] += [("x has",a)]
                continue
            # If no difference, Concede
            if h[i][-1][1] != "Concede":
                h[i] += [("A","Concede",("x is",y))]
        return h, k
            


