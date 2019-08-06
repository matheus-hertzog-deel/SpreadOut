import csv
from args import *

from utils.logger import *
from utils.dataloaders import *
from utils.initializers import *
from utils.spreadout import *

from models.resnet import Bottleneck, ResNet, BasicBlock, resnet34
from models.wide_resnet import *
from torchvision.models import alexnet
from torch.optim.lr_scheduler import MultiStepLR

#from parallel import DataParallelModel, DataParallelCriterion

def learn(X, y):
    model.train()

    output = model(X)
    loss = criterion(output,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, output

def validate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: {}'.format(accuracy))
    return accuracy

def nn_train():
    model.train() #sets training mode
    for epoch in range(args.epochs):
        correct = 0
        total = 0
        ave_loss = 0
        for batch_idx, (x,target) in enumerate(trainloader):
            x = x.to(device)
            target = target.to(device)
            loss, output = learn(x, target)
            total += target.size(0)
            ave_loss = ave_loss * 0.9 + loss.item() * 0.1
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            if(batch_idx+1) % args.batch_size == 0 or (batch_idx+1) == len(trainloader):
                accuracy = 100 * correct / total
                print('==>>> epoch: {}, batch index: {}, train loss {:.6f}, accuracy: {}'.format(epoch,batch_idx+1,ave_loss,accuracy))
        acc = 100 * (correct / total)
        test_acc = validate()
        writer.writerow({'Epoch': epoch, 'train_acc': acc, 'test_acc': test_acc})
        scheduler.step(epoch)

def main():

    global args, model, epochs, device, batch_size, num_classes, PATH, writer, dist_writer
    global optimizer, criterion, scheduler, trainloader, testloader

    args = get_args()
    print(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("/////////// Found device {} //////////".format(device))
    print("//////////  Device Name: {} ////////// ".format(torch.cuda.get_device_name(device.index)))

    ## Create new path logs
    i = 0
    while os.path.exists(os.path.join("Logs", args.dataset, args.model,"GammaByLayer", args.init_mode, "DataAugment=" + str(args.augmentation),  "Gamma" + str(args.gamma) + "InitEpochs" + str(args.init_epochs) + "Lr" + str(args.lr), "Trial:" + str(args.run + i))):
        i+=1
    PATH = os.path.join("Logs", args.dataset, str(args.layers) ,args.model, args.init_mode, "DataAugment=" + str(args.augmentation) + "," +"Cutout=" + str(args.cutout) + "AutoAugment" + str(args.auto_augment),
        "Gamma" + str(args.gamma) + "InitEpochs" + str(args.init_epochs) + "Lr" + str(args.lr), "Trial:" + str(args.run + i))
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    print("SAVING INFO IN")
    print(PATH)

    ## Initialize Loggers
    fieldnames = ['Dataset','Init', 'Lr', 'DataAugment', 'Cutout', 'AutoAugment', 'Seed', 'Gamma', 'InitEpochs', 'TrainEpochs', 'Epoch', 'train_acc', 'test_acc']
    distance_logger_fields = ['Layer', 'Distance', 'Stage']
    writer = CSVLogger(fieldnames, os.path.join(PATH, "TrainLog.csv"))
    dist_writer = CSVLogger(distance_logger_fields, os.path.join(PATH,"DistanceLog.csv"))
    writer.writerow({'Dataset':args.dataset, 'Init': args.init_mode,'Lr':args.lr,'DataAugment': args.augmentation,
                     'Cutout': args.cutout,'AutoAugment': args.auto_augment, 'Seed':args.seed, 'Gamma':args.gamma, 'InitEpochs': args.init_epochs, 'TrainEpochs': args.epochs})

    ## Seed torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    forms = ['Dataset', 'Epochs', 'LearningRate', 'Init Mode', 'Training Time', 'Test Accuracy', 'Init time', 'gamma']
    fields = [args.dataset, args.epochs, args.lr, args.init_mode, args.gamma]

    ## Start loader
    trainloader, testloader = getDataloaders(args)

    ## Get num_classes
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'cifar10' or 'stl10':
        num_classes = 10
    if args.dataset == 'caltech':
        num_classes = 256
    if args.dataset == 'tiny':
        num_classes = 200

    ## Get model
    if args.model == 'alex':
        model = alexnet()
    elif args.model == 'resnet34':
        print("Loading model resnet34")
        model = resnet34(num_classes=num_classes)
    elif args.model == 'resnet50':
        print("Loading model resnet50")
        model = resnet50(num_classes=num_classes)
    elif args.model == 'wide_resnet28-10':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)
    elif args.model == 'wide_resnet50-2':
        print("building wide_resnet50")
        class WRNBottleneck(Bottleneck):
            expansion = 2
        model = ResNet(WRNBottleneck, [2,2,2,2], width_per_group=64 * 2)

    #if torch.cuda.device_count() > 1:
      #print("Let's use", torch.cuda.device_count(), "GPUs!")
      #model = nn.DataParallel(model)

    model = model.to(device)
    init_w_alex(model) # He Init
    #model = DataParallelModel(model)

    # Model Hypers
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, nesterov=True, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    spreadout_optimizer = torch.optim.SGD(model.parameters(), lr=1e-03)

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160, 250], gamma=0.2)

    initializer = Spreadout(1, args.init_mode)

    # Make a list from model's convolutional layers
    conv_layers = []
    for child in model.modules():
        if(isinstance(child, nn.Conv2d)):
            conv_layers.append(child)

    # Create a slice or a list with layers to be spread
    if args.layers is not None:
        target_layers = [int(item) for item in args.layers.split(',')]
        if args.layer_slice:
            s = slice(target_layers[0],target_layers[1])
        else:
            s = [x for x in target_layers]
    else:
        s = None

    # Log distances before spreadout
    print("Logging distances")
    dist_writer.log_distances('pre_init', conv_layers, initializer.check_fn)
    print("Done")

    if args.init_mode == 'ortho':
        initialize_model(model, args.init_mode)
        
    if not args.init_mode == 'None':
        if args.init_epochs < 0: # auto spread
            initializer(model)
        else:
            initializer.spread(model, args.init_epochs, args.gamma, s)

    print("Logging...")
    dist_writer.log_distances('pos_init',conv_layers, initializer.check_fn)
    print("Done")

    # Training
    print(":::: Starting training ::::")
    print("Dataset: ", args.dataset, "Classes: ", num_classes, "Model: ", args.model)

    nn_train()
    acc = validate()

    print("Done")
    print("Final logging...")
    dist_writer.log_distances('pos_train', conv_layers, initializer.check_fn)
    print("Done")
    torch.save(model.state_dict(), os.path.join(PATH,args.model+str(args.run)))

    dist_writer.close()
    writer.close()
    print(args)

if __name__ == "__main__":
    main()
