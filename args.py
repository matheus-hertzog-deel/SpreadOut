import argparse

def get_args():

    parser = argparse.ArgumentParser(description='Filter Desambiguation')

    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (default: 1)')

    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")

    parser.add_argument('--path', default='./log', help='Path to save model')

    parser.add_argument('--device', choices=['cuda:0','cuda:1', 'cpu'], default='cpu')

    parser.add_argument('--dataset', choices=['mnist','cifar10', 'cifar100', 'stl10', 'tiny', 'caltech'], default='cifar10', help='Choose dataset to perform train and test')

    parser.add_argument('--init_mode', choices=['ED','rv','pearson','euclidean','ortho_ed', 'ortho_edALL', 'mahalanobis', 'ortho', 'squared_mahalanobis', 'None'], default='None', help='Kernel init mode')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--augmentation', type=bool, default=False, help='augment data')

    parser.add_argument('--cutout', type=bool, default=False, help='Cutout augmentation')

    parser.add_argument('--auto_augment', type=bool, default=False, help='auto augmentation')   #https://github.com/DeepVoltaire/AutoAugment

    parser.add_argument('--init_epochs', type=int, default=10, help='Kernel Init Epochs')

    parser.add_argument('--gamma', type=float, default=1.0, help='Init coeficient')

    parser.add_argument('--seed', type=int, default=42, help='Init coeficient')

    parser.add_argument('--model', choices=['alex','resnet34','resnet50','wide_resnet28-10','wide_resnet50-2'], default='alex', help='select cnn architecture')

    parser.add_argument('--run', type=int, default=0, help='Run counter')

    parser.add_argument('--layers', help='delimited layers input', type=str, default=None)
    parser.add_argument('--layer_slice', help='delimited layers input', type=bool, default=False)

    args = parser.parse_args()
    return args
