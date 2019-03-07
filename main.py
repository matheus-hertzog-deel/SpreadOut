import csv
from args import *

from utils.logger import *
from utils.dataloaders import *
from utils.initializers import *

from models.resnet import *
from models.wide_resnet import *
from torchvision.models import alexnet


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
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    return 100 * correct / total

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
        test_acc = validate()
        writer.writerow({'Epoch': epoch, 'train_acc': accuracy, 'test_acc': test_acc})

def main():

    global args, model, epochs, device, batch_size, PATH, writer, dist_writer
    global optimizer, criterion, trainloader, testloader

    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("/////////// Found device {} //////////".format(device))
    print("//////////  Device Name: {} ////////// ".format(torch.cuda.get_device_name(device.index)))

    # Paths for saving logs
    PATH = os.path.join("Logs", args.dataset, args.model, args.init_mode,  "Gamma" + str(args.gamma) + "InitEpochs" + str(args.init_epochs) + "Lr" + str(args.lr), "Trial:" + str(args.run))
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    print("SAVING INFO IN")
    print(PATH)

    ## Initialize Loggers
    fieldnames = ['Dataset','Init', 'Lr', 'Gamma', 'InitEpochs', 'TrainEpochs', 'Epoch', 'train_acc', 'test_acc']
    distance_logger_fields = ['Layer', 'Distance', 'Stage']
    writer = CSVLogger(fieldnames, os.path.join(PATH, "TrainLog.csv"))
    dist_writer = CSVLogger(distance_logger_fields, os.path.join(PATH,"DistanceLog.csv"))
    writer.writerow({'Dataset':args.dataset, 'Init': args.init_mode,'Lr':args.lr,'Gamma':args.gamma,'InitEpochs': args.init_epochs,'TrainEpochs': args.epochs})

    ## Seed torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    forms = ['Dataset', 'Epochs', 'LearningRate', 'Init Mode', 'Training Time', 'Test Accuracy', 'Init time', 'gamma']
    fields = [args.dataset, args.epochs, args.lr, args.init_mode, args.gamma]

    ## Start loader
    trainloader, testloader = getDataloaders(args.batch_size, args.dataset, args.model)

    ## Get num_classes
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'cifar10' or 'stl10':
        num_classes = 10
    elif args.dataset == 'caltech':
        num_classes = 256
    elif args.dataset == 'tiny':
        num_classes = 200

    # Get model
    if args.model == 'alex':
        model = alexnet()
    elif args.model == 'resnet':
        #model = ResNet101(num_classes = num_classes)
        model = resnet101()
    elif args.model == 'wide_resnet':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)
    model = model.to(device)

    ## Model Hypers
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)
    spreadout_optimizer = optim.SGD(model.parameters(), lr=args.lr)


    ## Make a list from models convolutional layers
    conv_layers = []
    for child in model.modules():
        if(isinstance(child, nn.Conv2d)):
            conv_layers.append(child)



    print("Logging distances")
    #dist_writer.log_distances('pre_init', conv_layers)
    print("Done")


    initialize_model(model, args.init_mode, args.init_epochs, args.gamma, spreadout_optimizer)


    print("Logging...")
    #dist_writer.log_distances('pos_init',conv_layers)
    print("Done")


    # Training
    print(":::: Starting training ::::")
    nn_train()
    acc = validate()
    print("Done")
    print("Final logging...")
    dist_writer.log_distances('pos_train', conv_layers)
    print("Done")
    torch.save(model.state_dict(), os.path.join(PATH,args.model+str(args.run)))

    dist_writer.close()
    writer.close()

if __name__ == "__main__":
    main()
