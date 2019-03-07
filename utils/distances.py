import torch
import torchvision
import numpy as np
from numpy import linalg as la

def kernel_EuclideanDistance(kernel, check = False):
    """
    Returns the summation of distances between filters in kernel
    """
    sum_dist = 0
    for i,target in enumerate(kernel.weight):
        for it, _filter in enumerate(kernel.weight):
            if i == it:
                continue
            dist = 0
            sub = _filter - target
            dist = torch.pow(sub,2)
            dist = torch.sum(dist)
            dist = torch.sqrt(dist)
            sum_dist += dist
            if check == True:
                print('Distance between filter {} and {} is equal to: {}'.format(i,it,dist))
    return sum_dist

def ED_var(kernel, check = False):
    """
    Returns the summation of distances between filters in kernel and its variance
    """
    l = []
    sum_dist = 0
    for i,target in enumerate(kernel.weight):
        for it, _filter in enumerate(kernel.weight):
            if i == it:
                continue
            dist = 0
            sub = _filter - target
            dist = torch.pow(sub,2)
            dist = torch.sum(dist)
            dist = torch.sqrt(dist)
            l.append(dist)
            sum_dist += dist
            if check == True:
                print('Distance between filter {} and {} is equal to: {}'.format(i,it,dist))
    v = torch.stack(l)
    v = v.var()    #variance in distances
    return v, sum_dist




def _mahalanobis(X):
    V = torch.inverse(cov(X))
    if not _isPD(V):
        VI = _nearestPD(V) #nearest Positive Definite of covariance matrix
    else:
        VI = V
    total_dist = 0
    for i,v in enumerate(X):
        dist = 0
        for j,u in enumerate(X):
            if i == j:
                continue
            x = (v-u).unsqueeze(0).t()
            y = (v - u).unsqueeze(0)

            dist = (torch.mm(torch.mm(y,VI),x)) #sqrt of dist returns NaN (?)

            total_dist +=dist
            #print(dist)
    return total_dist

def _nearestPD(A):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B = (A + A.t()) / 2
    _, s, V = torch.svd(B)

    H = torch.mm(V.t(), torch.mm(torch.diag(s),V))

    A2 = (B+H) / 2
    A3 = (A2 + A2.t())/ 2

    if _isPD(A3):
        return A3
    spacing = np.spacing(la.norm(A.cpu().detach().numpy()))

    I = torch.eye(A.shape[0]).to(device)
    k = 1
    while not _isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3.cpu().detach().numpy())))
        tmp = -mineig * k**2 + spacing
        A3 += I * tmp
        #A3 += I * (-mineig * k**2 + torch.from_numpy(np.array(spacing)).float().to(device) )
        k += 1
    return A3


def _isPD(B):
    try:

        M = B.cpu().detach().numpy()
        _ = la.cholesky(M)
        return True
    except la.LinAlgError:
        return False

# Returns the covariance matrix of m
def cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def _batch_mahalanobis(L, x):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both L and x.
    """
    # TODO: use `torch.potrs` or similar once a backwards pass is implemented.
    flat_L = L.unsqueeze(0).reshape((-1,) + L.shape[-2:])
    L_inv = torch.stack([torch.inverse(Li.t()) for Li in flat_L]).view(L.shape)
    #return (x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0).sum(-1)
    return ((x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0) + 1e7).sum(-1)
