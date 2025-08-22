class AgentP:
    def __init__(self,p,y_p,a_p,counter_a,new_proto,select_label,f_model,d): 
        super().__init__()
        '''
            p: prototypes
            y_p: label prototypes 
            a_p: attributes of prototypes. 
            counter_a: to see if it counters an attribute; return -1 if not, otherwise the prototype to use for counter 

        '''
        self.p = p
        self.a_p = a_p
        self.counter_a = counter_a
        self.select_label = select_label
        self.new_proto = new_proto
        self.y_p = y_p
        self.f_model = f_model
        self.d = d
    
    


    def answer(self,xs,h,k):
        '''
            xs: images to classify
            h: dialogue history stocking move as a triplet (agent, locution, argument)
            k: knowledge including
                1. the attributes provided by the other agent,
                2. similarity encoding of prototypes
                3. the distances between prototypes similarity encoding and the image to classify encoding
        '''
        # calculate similarity encoding of image to classify and distances with prototypes at the start of the dialogue
        if len(k[0]) == 0:
            f_x = self.f_model(xs).cpu().detach()
            distance = self.d(f_x,self.p)
            k = (k,f_x,distance)
        f_x = k[1] # similarity encodings
        d = k[2] # distance between similarity encodings
        k = k[0] # attributes provided by the other agent
        
        
        last_arg = None
        a_present = True
        
        
        for i, x in enumerate(xs):
            l = None
            # first label proposition
            if len(h[i]) == 0:
                y = self.select_label(self.a_p,self.y_p,k[i],d[i])
                h[i] += [("B","Propose",("x is",y))]
                k[i] += [("x is",y)]
                continue
            
            
            if h[i][-1][1] == "Argue":
                last_arg = h[i][-1]
                # find the last argument of agent A and if the attribute is detected present/absent
                for prem in last_arg[-1]:
                    # search for attribute in premisses
                    if prem[0] == "has" or prem[0] == "has not":
                        
                        a = prem[-1][1]
                        if prem[0] == "has not":
                            a_present = True
                        else:
                            a_present = False
                        break
                l = last_arg[2][1]
                # Search to counter attach the argument
                p = self.counter_a(a_present,self.a_p[:,a], d[i])
                # counter attack
                if p != -1:
                    if a_present:
                        h[i] += [("B","Argue",("x has not",a),[("x is sim to",p),("has not",[p,a])])]
                        k[i] += [("x has not",a)]
                    else:
                        h[i] += [("B","Argue",("x has not",a),[("x is sim to",p),("has",[p,a])])]
                        k[i] += [("x has",a)]
                    continue
                else:
                    # if no counter attack, add to knowledge the information 
                    if a_present:
                        k[i] += [("x has",a)]
                    else:
                        k[i] += [("x has not",a)]
                
                
                
                    
            # get last label conclusion
            if l is None:
                l = h[i][-1][-1][-1]
            
            # search prototype to justify the proposition
            p = self.new_proto(self.y_p,self.a_p,k[i], d[i])
            if p != -1:
                h[i] += [("B","Argue",("x is",l),(("x is sim to",p),("is",[p,l])))]
                k[i] += [("x is sim to",p)]
                continue
            # if no argument Concede
            if h[i][-1][1] == "Argue" and h[i][-1][0][0] == 'A':
                h[i] += [("B","Concede",("x is not",l))]
            # if nothing to say about the label, propose a new label
            elif h[i][-1][1] == "Concede":
                n_l = self.select_label(self.a_p,self.y_p,k[i], d[i])
                
                if n_l != -1:
                    h[i] += [("B","Propose",("x is",n_l))]
                    k[i] += [("x is",n_l)]
        return h, (k, f_x, d)
                    
            
        
        
        
            


