import torch
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from tqdm import tqdm
import PIL
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader


def mkds(args, dataset):
    fulldataset = []
    num_data = 1000
    classes = dataset.classes
    class_per_data = num_data/len(classes)
    num_original = int((1 - args.rate) * class_per_data)
    num_generated = int(args.rate * class_per_data)
    print('num of original data per class : ',num_original)
    print('num of generated data per class : ',num_generated)
    for i in range(len(classes)):
        count = 0
        for j in range(len(dataset)):
            if dataset.targets[j] == i:
                fulldataset.append(dataset[j])
                count = count + 1
            if count >= num_original:
                break
    if args.dataset == 'mnist':
        for i in range(len(classes)):
            for j in range(int(num_generated)):
                transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()])
                img = transform(PIL.Image.open(f'./mnist/generated_dataset/{i}/MNIST-{i}-{j}.png'))
                fulldataset.append((img,i))
    elif args.dataset == 'cifar10':
        for i in range(len(classes)):
            for j in range(int(num_generated)):
                transform = transforms.Compose([transforms.ToTensor()])
                img = transform(PIL.Image.open(f'./cifar10/generated_dataset/{classes[i]}/{j}.png'))
                fulldataset.append((img,i))
    return fulldataset
    
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100
    if args.dataset == "mnist": #generated 데이터 32x32 resize
        from mnist import LeNet5
        transform = transforms.Compose([transforms.ToTensor()])
        traindataset = datasets.MNIST('../data', train=True, download=True, transform= transform)
        testdataset = datasets.MNIST('../data', train=False, download=True, transform= transform) 
    elif args.dataset =="cifar10":
        from cifar10 import LeNet5
        transform = transforms.Compose([transforms.ToTensor()])
        traindataset = datasets.CIFAR10('../data', train=True, download=True, transform= transform)
        testdataset = datasets.CIFAR10('../data', train=False, download=True, transform= transform) 
    
    traindataset = mkds(args, traindataset)
    model = LeNet5.LeNet5().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loader = DataLoader(traindataset, batch_size=16, shuffle= True)
    test_loader = DataLoader(testdataset, batch_size = 16)
    model.train()
    total_loss = []
    testing_accuracy = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        loss = 0
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            train_loss = criterion(output, targets)
            train_loss.backward()
            optimizer.step()
            loss = loss + train_loss
        pbar.set_description(f'Train Epoch: {epoch}/{epochs} Loss : {loss:.6f}')
        total_loss.append(loss.cpu().detach().numpy())
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
            testing_accuracy.append(accuracy)
    plt.plot(total_loss, color = '#1f77b4')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(f'./result/{args.dataset}-{args.rate}-training_loss.png', format='png')
    plt.clf()
    plt.plot(testing_accuracy, color = '#ff7f0e')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim([0,100])
    plt.savefig(f'./result/{args.dataset}-{args.rate}-testing_accuracy.png', format='png')    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
    parser.add_argument("--rate", default= 0.1, type = float )

    args = parser.parse_args()
    main(args)