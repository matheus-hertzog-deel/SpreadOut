import os
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models import alexnet
from torch.utils.data import DataLoader
from .autoaugment import *
from .cutout import Cutout
#from tiny_imagenet import TinyImageNet

def getMnist(args):

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
                     batch_size=args.batch_size,
                     shuffle=True,
                     num_workers=4)
    mnist_test_loader = torch.utils.data.DataLoader(
                    dataset=mnist_testset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=4)

    print('===>>> MNIST total training batch number: {}'.format(len(mnist_train_loader)))
    print('===>>> MNIST total testing batch number: {}'.format(len(mnist_test_loader)))
    return mnist_train_loader, mnist_test_loader


def getCifar10(args):
    root = './data/cifar10'
    if not os.path.exists(root):
        os.makedirs(root)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    if args.augmentation:    # Order matters!
        transform.transforms.append(transforms.RandomCrop(32, padding=4))
        transform.transforms.append(transforms.RandomHorizontalFlip())
    if args.model == 'alex':
        transform.transforms.append(transforms.Resize(224))
        test_transform.transforms.append(transforms.Resize(224))
    transform.transforms.append(transforms.ToTensor())
    transform.transforms.append(normalize)
    if args.cutout:
        transform.transforms.append(Cutout(n_holes=1, length=16))

    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(normalize)
    print(transform)
    trainset = dset.CIFAR10(root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
    testset = dset.CIFAR10(root, train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)
    print('===>>> CIFAR10 total training batch number: {}'.format(len(trainloader)))
    print('===>>> CIFAR10 total testing batch number: {}'.format(len(testloader)))
    return trainloader, testloader



def getCifar100(args):
    root = './data/cifar100'
    if not os.path.exists(root):
        os.makedirs(root)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    if args.augmentation:    # Order matters!
        transform.transforms.append(transforms.RandomCrop(32, padding=4))
        transform.transforms.append(transforms.RandomHorizontalFlip())
    if args.model == 'alex':
        transform.transforms.append(transforms.Resize(224))
        test_transform.transforms.append(transforms.Resize(224))
    transform.transforms.append(transforms.ToTensor())
    transform.transforms.append(normalize)
    if args.cutout:
        transform.transforms.append(Cutout(n_holes=1, length=8))

    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(normalize)
    #transform = transforms.Compose([ transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source

    print(transform)
    trainset = dset.CIFAR100(root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testset = dset.CIFAR100(root, train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    print('===>>> CIFAR100 total training batch number: {}'.format(len(trainloader)))
    print('===>>> CIFAR100 total testing batch number: {}'.format(len(testloader)))
    return trainloader, testloader

def getStl(args):
    root = './data/stl10'
    if not os.path.exists(root):
        os.makedirs(root)

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    if args.augmentation:    # Order matters!
        transform.transforms.append(transforms.RandomCrop(96, padding=4))
        transform.transforms.append(transforms.RandomHorizontalFlip())
    if args.model == 'alex':
        transform.transforms.append(transforms.Resize(224))
        test_transform.transforms.append(transforms.Resize(224))
    transform.transforms.append(transforms.ToTensor())
    transform.transforms.append(normalize)
    if args.cutout:
        transform.transforms.append(Cutout(n_holes=1, length=16))

    print(transform)
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(normalize)

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    testset = torchvision.datasets.STL10(root='./data', split='test', download=False, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader

def getTinyImageNet(args):
    root = './data/tiny-imagenet-200'
    train_root = os.path.join(root, 'train')
    validation_root = os.path.join(root, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([])
    test_transform = transforms.Compose([])


    if args.model == 'alex':
        transform.transforms.append(transforms.Resize(224))
        test_transform.transforms.append(transforms.Resize(224))
    if args.augmentation:    # Order matters!
        transform.transforms.append(transforms.RandomCrop(224, padding=4))
        transform.transforms.append(transforms.RandomHorizontalFlip())
    transform.transforms.append(transforms.ToTensor())
    transform.transforms.append(normalize)
    if args.cutout:
        transform.transforms.append(Cutout(n_holes=1, length=16))

    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(normalize)

    train_data = dset.ImageFolder(train_root, transform=transform)
    valid_data = dset.ImageFolder(validation_root, transform=test_transform)

    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=4)

    return trainloader, validloader

def getCaltech(args):
    root = './data/256_ObjectCategories'
    train_root = os.path.join(root, 'train')
    validation_root = os.path.join(root, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    if args.model == 'alex':
        transform.transforms.append(transforms.Resize((224,224)))
        test_transform.transforms.append(transforms.Resize((224,224)))
    if args.augmentation:    # Order matters!
        transform.transforms.append(transforms.RandomCrop((224,224), padding=4))  # VERIFY ARCHITECTURES INPUT SIZE !!
        transform.transforms.append(transforms.RandomHorizontalFlip())
    transform.transforms.append(transforms.ToTensor())
    transform.transforms.append(normalize)
    if args.cutout:
        transform.transforms.append(Cutout(n_holes=1, length=16))

    print(transform)
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(normalize)

    train_data = dset.ImageFolder(train_root, transform=transform)
    valid_data = dset.ImageFolder(validation_root, transform=test_transform)

    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=4)

    return trainloader, validloader


def getDataloaders(args):

    if args.dataset == 'cifar100':
        print('Getting loaders for Cifar100 !')
        trainloader, testloader = getCifar100(args)
        return trainloader, testloader

    elif args.dataset == 'cifar10':
        print('Getting loaders for Cifar10 !')
        trainloader, testloader = getCifar10(args)
        return trainloader, testloader

    elif args.dataset == 'stl10':
        print('Getting loaders for Stl10 !')
        trainloader, testloader = getStl(args)
        return trainloader, testloader

    elif args.dataset == 'tiny':
        print('Getting loaders for TinyImageNet !')
        trainloader, testloader = getTinyImageNet(args)
        return trainloader, testloader

    elif args.dataset == 'caltech':
        print('Getting loaders for Caltech-256 !')
        trainloader, testloader = getCaltech(args)
        return trainloader, testloader

    else:
        print('Getting loaders for MNIST !')
        trainloader, testloader = getMnist(args)
        return trainloader, testloader

    return trainloader, testloader
