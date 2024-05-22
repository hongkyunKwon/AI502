import torch
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import argparse

def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        #imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        
    return train_loss.item()

def main(args):
    epochs = 100
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from MNIST import LeNet5
        model = LeNet5()
    elif args.dataset =="cifar10":
        traindataset = datasets.CIFAR10('../data', train=True, download=True,transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform) 
        from Cifar10 import LeNet5
        model = LeNet5()


    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    traindataset_data = traindataset.data[:1000]
    traindataset_target = traindataset.targets[:1000]


    model.train()
    for epoch in epochs:
        

        optimizer.step()
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10")
    parser.add_argument("--rate", default= 0.7, type = float )

    args = parser.parse_args()
    main(args)