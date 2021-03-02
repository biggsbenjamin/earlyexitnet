#some helper tools

from models.Branchynet import Branchynet, ConvPoolAc
import torch
import torch.nn as nn
import torch.optim as optim

#checks the shape of the input and output of the model
def shape_test(model, dims_in, dims_out, loss_function=nn.CrossEntropyLoss()):
    rand_in = torch.rand(tuple([1, *dims_in]))
    rand_out = torch.rand(tuple([*dims_out])).long()

    model.eval()
    with torch.no_grad():
        results = model(rand_in)
        if isinstance(results, list):
            losses = [loss_function(res, rand_out) for res in results ]
        else:
            losses = [loss_function(results, rand_out)]
    return losses

# Our drawing graph functions. We rely / have borrowed from the following
# python libraries:
# https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py
# https://github.com/willmcgugan/rich
# https://graphviz.readthedocs.io/en/stable/
def draw_graph(start, watch=[]):
    from graphviz import Digraph
    node_attr = dict(style='filled',
                    shape='box',
                    align='left',
                    fontsize='12',
                    ranksep='0.1',
                    height='0.2')
    graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    assert(hasattr(start, "grad_fn"))
    if start.grad_fn is not None:
        label = str(type(start.grad_fn)).replace("class", "").replace("'", "").replace(" ", "")
        print(label) #missing first item
        graph.node(label, str(start.grad_fn), fillcolor='red')

        _draw_graph(start.grad_fn, graph, watch=watch, pobj=label)#str(start.grad_fn))
        size_per_element = 0.15
    min_size = 12    # Get the approximate number of nodes and edges
    num_rows = len(graph.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    graph.graph_attr.update(size=size_str)
    graph.render(filename='net_graph.jpg')

def _draw_graph(var, graph, watch=[], seen=[], indent=".", pobj=None):
    ''' recursive function going through the hierarchical graph printing off
    what we need to see what autograd is doing.'''
    from rich import print
    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is not None:
                #if joy not in seen: #WARNING if there's looping behaviour will break

                label = str(type(joy)).replace("class", "").replace("'", "").replace(" ", "")
                label_graph = label
                colour_graph = ""
                seen.append(joy)
                if hasattr(joy, 'variable'):
                    happy = joy.variable
                    if happy.is_leaf:
                        label += " \U0001F343"
                        colour_graph = "green"
                        vv = []
                        for (name, obj) in watch:
                            if obj is happy:
                                label += " \U000023E9 " + \
                                    "[b][u][color=#FF00FF]" + name + \
                                    "[/color][/u][/b]"
                                label_graph += name
                                colour_graph = "blue"
                                break
                            vv = [str(obj.shape[x]) for x in range(len(obj.shape))]
                        label += " [["
                        label += ', '.join(vv)
                        label += "]]"
                        label += " " + str(happy.var())
                        graph.node(str(joy), label_graph, fillcolor=colour_graph)
                print(indent + label)
                _draw_graph(joy, graph, watch, seen, indent + ".", joy)
                if pobj is not None:
                    graph.edge(str(pobj), str(joy))


def vis_backprop_graphs():
    #set up the model
    model = Branchynet()
    print("Model done")


    #set loss function - og bn used "softmax_cross_entropy" unclear if this is the same
    loss_f = nn.CrossEntropyLoss() # combines log softmax and negative log likelihood
    print("Loss function set")

    #shape testing
    #print(shape_test(model, [1,28,28], [1])) #output is not one hot encoded

    bb_only=False

    lr = 0.001
    exp_decay_rates = [0.99, 0.999]
    backbone_params = [
            {'params': model.backbone.parameters()},
            {'params': model.exits[-1].parameters()}
            ]

    if bb_only:
        opt = optim.Adam(backbone_params, betas=exp_decay_rates, lr=lr)
    else:
        opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

    rand_in = torch.rand(tuple([1, *[1,28,28]]))
    rand_out = torch.rand(tuple([*[1]])).long()

    model.train()
    print("GO")

    results = model(rand_in)
    if bb_only:
        loss = loss_f(results[-1], yb)
    else:
        loss = 0.0
        for res in results:
            loss += loss_f(res, rand_out)

    opt.zero_grad()
    loss.backward(create_graph=True)

    #GRAPH TEST STUFF
    watching=[('bb-strt', model.backbone[0].weight)]
    for i,block in enumerate(model.exits[0]):
        print(i,block)
        if hasattr(block, 'weight'):
            watching.append(('ee1_'+str(i), block.weight))
            print("\t",i,block)
        elif isinstance(block, ConvPoolAc):
            for j,subblock in enumerate(block.layer):
                if hasattr(subblock, 'weight'):
                    watching.append(('ee1_'+str(i)+'_'+str(j), subblock.weight))
                    print("\t",i,j,subblock)

    print("------")

    for i,block in enumerate(model.backbone[1]):
        print(i,block)
        if hasattr(block, 'weight'):
            watching.append(('bbmain_'+str(i), block.weight))
            print("\t",i,block)
        elif isinstance(block, ConvPoolAc):
            for j,subblock in enumerate(block.layer):
                if hasattr(subblock, 'weight'):
                    watching.append(('bbmain_'+str(i)+'_'+str(j), subblock.weight))
                    print("\t",i,j,subblock)

    watching.append(('exitF', model.exits[1][-1].weight))

    draw_graph(loss, watching)

def probe_params(model):
    #probe params to double check only backbone run
    print("backbone 1st conv")
    print([param for param in model.backbone[0].parameters()])
    print("backbone last linear")
    print([param for param in model.exits[-1].parameters()])
    print("exit 1")
    print([param for param in model.exits[0].parameters()])

