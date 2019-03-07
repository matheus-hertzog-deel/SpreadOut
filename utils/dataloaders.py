import os
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models import alexnet
from torch.utils.data import DataLoader
#from tiny_imagenet import TinyImageNet

def getMnist(batch_size,model):

    root = './data/mnist'
    if not os.path.exists(root):
        os.makedirs(root)

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
    ])

    mnist_trainset = dset.MNIST(root=root, train=True,transform = data_transform, download=True)
    mnist_testset = dset.MNIST(root=root, train=False,transform = data_transform, download=True)

    mnist_train_loader = torch.utils.data.DataLoader(
                     dataset=mnist_trainset,
                     batch_size=batch_size,
                     shuffle=True,
                     num_workers=4)
    mnist_test_loader = torch.utils.data.DataLoader(
                    dataset=mnist_testset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4)

    print('===>>> MNIST total training batch number: {}'.format(len(mnist_train_loader)))
    print('===>>> MNIST total testing batch number: {}'.format(len(mnist_test_loader)))
    return mnist_train_loader, mnist_test_loader

def getCifar10(batch_size,model):
    root = './data/cifar10'
    if not os.path.exists(root):
        os.makedirs(root)

    if model == 'alex':
        transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) )])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) )])

    trainset = dset.CIFAR10(root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testset = dset.CIFAR10(root, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    print('===>>> CIFAR10 total training batch number: {}'.format(len(trainloader)))
    print('===>>> CIFAR10 total testing batch number: {}'.format(len(testloader)))
    return trainloader, testloader

def getCifar100(batch_size,model):
    root = './data/cifar100'
    if not os.path.exists(root):
        os.makedirs(root)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if model == 'alex':
        transform = transforms.Compose([transforms.Resize(224) , transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    else:
        transform = transforms.Compose([ transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(), normalize])

    trainset = dset.CIFAR100(root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testset = dset.CIFAR100(root, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    print('===>>> CIFAR100 total training batch number: {}'.format(len(trainloader)))
    print('===>>> CIFAR100 total testing batch number: {}'.format(len(testloader)))
    return trainloader, testloader

def getStl(batch_size,model):
    root = './data/stl10'
    if not os.path.exists(root):
        os.makedirs(root)

    if model == 'alex':
        transform = transforms.Compose([transforms.Resize(224) , transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    testset = torchvision.datasets.STL10(root='./data', split='test', download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader

def getTinyImageNet(batch_size,model):
    root = './data/tiny-imagenet-200'
    train_root = os.path.join(root, 'train')
    validation_root = os.path.join(root, 'val')

    if model == 'alex':
        transform = transforms.Compose([transforms.Resize(224) , transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.Resize(224) , transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = dset.ImageFolder(train_root, transform=transform)
    valid_data = dset.ImageFolder(validation_root, transform=transform)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_data, batch_size=batch_size, num_workers=4)

    return trainloader, validloader

def getCaltech(batch_size,model):
    root = './data/256_ObjectCategories'
    train_root = os.path.join(root, 'train')
    validation_root = os.path.join(root, 'val')

    normalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

    if model =='alex':
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), normalize])

    train_data = dset.ImageFolder(train_root, transform=transform)
    valid_data = dset.ImageFolder(validation_root, transform=transform)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_data, batch_size=batch_size, num_workers=4)

    return trainloader, validloader


def getDataloaders(batch_size, dataset, model):

    if dataset == 'cifar100':
        print('Getting loaders for Cifar100 !')
        trainloader, testloader = getCifar100(batch_size,model)
        return trainloader, testloader

    elif dataset == 'cifar10':
        print('Getting loaders for Cifar10 !')
        trainloader, testloader = getCifar10(batch_size,model)
        return trainloader, testloader

    elif dataset == 'stl10':
        print('Getting loaders for Stl10 !')
        trainloader, testloader = getStl(batch_size,model)
        return trainloader, testloader

    elif dataset == 'tiny':
        print('Getting loaders for TinyImageNet !')
        trainloader, testloader = getTinyImageNet(batch_size,model)
        return trainloader, testloader

    elif dataset == 'caltech':
        print('Getting loaders for Caltech-256 !')
        trainloader, testloader = getCaltech(batch_size,model)
        return trainloader, testloader

    else:
        print('Getting loaders for MNIST !')
        trainloader, testloader = getMnist(batch_size,model)
        return trainloader, testloader

    return trainloader, testloader
