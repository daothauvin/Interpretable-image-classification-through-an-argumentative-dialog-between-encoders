from PIL import Image

from graphviz import Digraph






def string_sentence(m,att_names, label_names):
    '''
        Transform a move m in a sentence
    '''
    str_sentence = m[0] + ": "
    if m[1] == "Propose":
        str_sentence += "I propose that " + string_arg(m[-1],att_names,label_names) + "."
    elif m[1] == "Why-Propose":
        str_sentence += "Why " + string_arg(m[-1],att_names,label_names) + "?"
    elif m[1] == "Concede":
        str_sentence += "Ok, " + string_arg(m[-1],att_names,label_names)
    elif m[1] == "Argue":
         str_sentence += string_arg(m[-2],att_names,label_names) + " because " + ", ".join([string_arg(arg,att_names,label_names) for arg in m[-1]]) + "."
    return str_sentence 

def string_arg(arg, att_names, label_names):
    '''
        Transform an argument arg in text
    '''
    if arg[0] == "x has":
        return "x has the attribute " + str_att(att_names[arg[1]])
    
    if arg[0] == "x has not":
        return "x has not the attribute " + str_att(att_names[arg[1]]) 
    
    if arg[0] == "has":
        return "prototype " + str(arg[1][0]) + " has the attribute " + str_att(att_names[arg[1][1]])
    
    if arg[0] == "has not":
        return "prototype " + str(arg[1][0]) + " has not the attribute " + str_att(att_names[arg[1][1]])
    
    if arg[0] == "x is":
        return "x is of label " + str_class(label_names[arg[1]])
    if arg[0] == "x is not":
        return "x is not of label " + str_class(label_names[arg[1]])
    if arg[0] == "is":
        return "prototype " + str(arg[1][0]) + " is of label " + str_class(label_names[arg[1][1]])
    if arg[0] == "x is sim to":
        return "x is similar to prototype " + str(arg[1])

def str_class(label):
    '''
        Transform a class in text
    '''
    if "_" in label:
        return  label.split(".")[1].replace("_"," ")

    return label

def str_att(att):
    '''
        Transform an attribut in text
    '''
    if "::" in att: # CUB attributes
        k,v = att.split("::")
        k = " ".join(k.split("_")[1:])
        return " ".join([v,k])
    else: # Other attributes
        return att

def str_att_2(att):
    '''
        Transform an attribut in text (multiline)
    '''
    if "::" in att:
        k,v = att.split("::")
        k = "\n".join(k.split("_")[1:])
        v = " ".join(v.split("_"))
        return "\n".join([v,k])
    else:
        return "-\n".join("\n".join(att.split(" ")).split("-"))


def get_image(idx,dataloader):
    '''
        get image from idx and dataloader
    '''
    return Image.open(dataloader.dataset.get_path(idx,dataloader))


def get_label(move):
    '''
        Recover the label of a move
    '''
    if move[1] == "Argue":
        for pre in move[3]:
            if pre[0] == "x has not" or pre[0] == "x has":
                return pre
            elif pre[0] == "x is sim to":
                return pre
    return None

def plot_graph(G, save_name):
    '''
        Plot the graph G in save_name
    '''
    G.render(save_name, format='png', cleanup=True, quiet = True)
    
def transform_label_name(label_name, space = "\n"):
    '''
        Change label for printing (separate texts by space)
    '''
    if "." in label_name:
        return label_name.replace("_",space).split(".")[1]
    if " " in label_name:
        return space.join(label_name.split(" "))
    return label_name




def plot_dialogue_tree(args, h, p_dataset, save_name, image_path, y):
    '''
        Plot dialogues in the form of a tree
        h: dialogue history
        p_dataset: dataset of prototypes
        save_name: file to save the dialogue tree
        image_path: path of the image to classify
        y: the ground truth of the image
    '''
    
    # create the graph
    G = Digraph(node_attr={'fixedsize': 'true','width': '2','height': '1.5'})
    i = 1   

    with G.subgraph() as s:
        s.attr(rankdir='TB',ranksep='0.1', nodesep='0.1')  # This ensures that the elements in the sub-graph are arranged vertically.
        # root of the graph
        G.node(str(0), label='', image=image_path, shape='square', penwidth='2', style='filled', fillcolor="orange")
        G.node('root_title', label='label: '+ transform_label_name(args.label_names[y], " "), shape='plaintext', fontname="Helvetica-Bold", fontsize='20', width='0.3', height='0.3', style='italic', fontcolor="black")
        G.edge('root_title', str(0), weight='0', style='invis',len='0')  # Relier le titre au nœud racine

    cur_label = 0

    
    for move in h:
        # color according to the agent
        color = "blue" if move[0][0] == 'A' else "orange"
        fontcolor='white' if color == "blue" else "black"

        # add label proposition
        if move[1] == "Propose":
            G.node(str(i), label = "Class\n"+ transform_label_name(args.label_names[move[2][1]]),fontsize='25', shape='square', penwidth='2', style='filled', fillcolor="orange")
            cur_label = str(i)
            G.edge(str(0), str(i), label =  "", fontsize='30', penwidth='5',arrowhead='none') 
            i += 1

        # add proposition
        elif move[1] == "Argue":
            
            label = get_label(move)
            
            # similarity argument
            if label[0] == "x is sim to":

                path = p_dataset.get_path(label[1])
                id_img = move[3][0][-1]  

                with G.subgraph() as s:
                    G.node(str(i), image = path, shape='square', penwidth='2', style='filled',fillcolor=color, label=str(id_img), labelloc = 'b', fontsize='30')
            # attribute argument
            else:
                G.node(str(i), label = label[0] + "\n" + str_att_2(args.attribute_names[label[1]]) + ' ' + str(label[1]),fontsize='25', shape='square', penwidth='2', style='filled',fillcolor=color, fontcolor = fontcolor)

            # add edge according to positif of negatif
            edge_color = 'green' if move[2][0] == "x is" else "red"
            if move[2][0] == "x is" or move[2][0] == "x is not":
                
                G.edge(cur_label, str(i), label =  "+" if move[2][0] == "x is" else "-", fontsize='30', penwidth='5', color=edge_color, font_color = edge_color,arrowhead='none')
            else:
                G.edge(str(i-1), str(i),  label = "+" if move[2][0] == "x is" else "-", fontsize='30', penwidth='5', color=edge_color, font_color = edge_color,arrowhead='none')
            i += 1
            
    # plot graph in file
    plot_graph(G, save_name)

        
def save_dialogue(args, d, save_name):
    '''
        Save Dialogue as a latex text
        args: 
        d: dialogue
        save_name: name to save dialogue
    '''
    txt = "\\noindent \\fbox{% \n \parbox{\linewidth}{ \n"

    for i, move in enumerate(d):
        if move[0][0] == 'A':
            textcolor = "abot"
            agent = "A"
        elif move[0][0] == 'B':
            textcolor = "qbot"
            agent = "P"
        txt = txt + "\n"
        txt =  txt + "\\noindent ("+str(i+1)+") \\textcolor{"+textcolor+"}{$\\bm{\mathcal{"+agent+"}}$}:" + string_sentence(move, args.attribute_names, args.label_names)[2:] + " \n"
    txt = txt + "}% \n } \n"
    with open(save_name,'w') as f:
        f.write(txt)


def get_label(move):
    if move[1] == "Argue":
        for pre in move[3]:
            if pre[0] == "x has not" or pre[0] == "x has":
                return pre
            elif pre[0] == "x is sim to":
                return pre
    return None

def plot_graph(G,node_labels, edge_labels):
    fig=plt.figure(figsize=(15,15))
    ax=plt.subplot(111)
    ax.set_aspect('equal')
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi")
    nx.draw_networkx(G,pos,labels=node_labels,node_size=[len(node_labels[i]) * 1000 if node_labels.get(i) is not None else 1000 for i in pos])
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
                  
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform

    piesize=0.2 # this is the image size
    p2=piesize/2.0
    for n in G:
        if G.nodes(data=True)[n].get("image") is not None:
            xx,yy=trans(pos[n]) # figure coordinates
            xa,ya=trans2((xx,yy)) # axes coordinates
            a = plt.axes([xa-p2,ya-p2, piesize, piesize])
            a.set_aspect('equal')
            a.imshow(G.nodes(data=True)[n]['image'])
            a.axis('off')
    ax.axis('off')

    plt.savefig('foo.png')

def plot_dialogue_tree(h,y,label_names, att_names, dataloader):
    G=nx.Graph()
    last_propose = None
    i = 1
    node_labels = dict()
    edge_labels = dict()
    
    for move in h:
        if move[1] == "Propose":
            if last_propose is not None:
                # pos = nx.nx_agraph.graphviz_layout(G, prog="twopi")
                plot_graph(G,node_labels, edge_labels)
                # plt.show()
                return
            elif move[2][1] == y:
                last_propose = y
                G.add_node(0)
                node_labels[0] = label_names[y].replace("_"," ")

        elif move[1] == "Argue":
            
            label = get_label(move)
            if label[0] == "x is sim to":
                G.add_node(i, image = get_image(label[1], dataloader))
            else:
                G.add_node(i)
                node_labels[i] = label[0] + "\n" + str_att(att_names[label[1]])
            if move[2][0] == "x is":
                G.add_edge(0, i)
            else:
                G.add_edge(i-1, i)
            edge_label = "+" if move[2][0] == "x is" else "-"
            edge_labels[(0,i)] = edge_label
            i += 1
            
            
    # pos = nx.nx_agraph.graphviz_layout(G, prog="twopi")
    plot_graph(G,node_labels, edge_labels)
    # plt.show()