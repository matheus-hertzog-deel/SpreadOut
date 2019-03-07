import math
import torch.nn as nn
from .distances import kernel_EuclideanDistance


def initialize_model(model, mode, init_epochs = 0, gamma = None, optimizer = None):
    if mode == 'ED':
        init_w_alex(model)
        euclidian_init(model, gamma, init_epochs, optimizer)
    elif mode == 'ortho':
        ortho_weights(model)
    elif mode == 'None':
        init_w_alex(model)
    elif mode == 'standard':
        print("Standard Init")
        return
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
