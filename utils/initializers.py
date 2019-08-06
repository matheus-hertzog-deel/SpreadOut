
import math
import torch.nn as nn
import torch
from tqdm import tqdm
#from .distances import *
from .distances import kernel_EuclideanDistance, cov, _batch_mahalanobis


def initialize_model(model, mode, init_epochs = 0, gamma = None, optimizer = None):
    print("Initializing Model")
    if mode == 'ortho':
        print("Using orthogonal initialization !!")
        ortho_weights(model)
    else:
        init_w_alex(model)
    return

def init_w_alex(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)

def ortho_weights(model):
    for f in model.modules():
        if (isinstance(f, nn.Conv2d)):
            nn.init.orthogonal_(f.weight)

def euclidian_init(model, gamma, init_epochs, kernel_optimizer):
    print("Applying SpreadOut to Conv2d Layers...")
    counter = 0
    for k in model.modules():
        if isinstance(k, nn.Conv2d):
            counter += 1
            print("Initializing layer {}".format(counter))
            for i in tqdm(range(init_epochs)):
                model.train()
                kernel_optimizer.zero_grad()

                filter_loss = kernel_EuclideanDistance(k)
                filter_loss = - filter_loss
                filter_loss *= gamma

                filter_loss.backward()
                kernel_optimizer.step()
    print("SpreadOut done!")

def squared_mahalanobis_init(model, gamma, init_epochs, kernel_optimizer):
    # create list from models' conv2d layers
    layers = []
    for i,k in enumerate(model.modules()):
        if isinstance(k, nn.Conv2d):
            layers.append(k)

    ## for each layer sum distance and propagate
    for k in range(len(layers)):
        print("Spreading layer ", layers[k])

        kernel_optimizer.zero_grad()
        filter_loss = 0

        l1 = layers[k].weight

        for j in range(l1.shape[1]): # iterate over channels
            X = l1[0][j].view(1,l1[0][j].shape[1]**2)  # gets first elem
            for i,w in enumerate(l1): # iterates over filters 0-> 64
                if i == 0:
                    continue
                else:
                    y = w[0].view(1,w[j].shape[1]**2)
                    X = torch.cat((X,y))

            VI = torch.inverse(cov(X)) #inverse of covariance matrix
            for _input in X:
                filter_loss += _batch_mahalanobis(VI,_input)
                #print(_batch_mahalanobis(VI,_input))
        #print(dist)

        filter_loss.backward()
        kernel_optimizer.step()
