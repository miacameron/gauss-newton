import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
import torch.linalg as linalg
import matplotlib.pyplot as plt
from models import *
from modules import *
from agfunctions import *
from plotting import *
import argparse
import torchvision
import torchvision.transforms as transforms
import os

parser = argparse.ArgumentParser()

parser.add_argument("--grad_type", help="type of weight transport algorithm (backprop, pseudo, random)")
parser.add_argument("--dataset", help="MNIST or CIFAR10")
parser.add_argument("--name", help='name of this experiment', default="unnamed")
parser.add_argument("--n_hidden", help="number of hidden layers", default=0)
parser.add_argument("--batch_size", help="training batch size", default=1)
parser.add_argument("--random_seed", help="random seed for pytorch", default=1234)
parser.add_argument("--epochs", help="number of epochs to run", default=3)
parser.add_argument("--connectivity", help="fc (fully-connected) or conv", default="fc")
parser.add_argument("--lr", help="learning rate", default=0.0001)


args = parser.parse_args()

torch.manual_seed(int(args.random_seed))
torch.set_float32_matmul_precision('high')

n_epochs = int(args.epochs)
batch_size_train = int(args.batch_size)
batch_size_test = 20
name = args.name
lr = float(args.lr)

net = None

assert (args.grad_type in ["backprop", "pseudo", "random"])
assert (args.dataset in ["MNIST", "CIFAR10"])
assert (args.connectivity in ["fc", "conv"])
dataset = args.dataset
grad_type = args.grad_type
connectivity = args.connectivity


if (dataset == "MNIST"):
    if (connectivity == "fc"):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x : torch.flatten(x))])
        trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)
        net = FCMNIST(grad_type=grad_type)
    else:
        raise Exception ("conv is not working")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)
        net = ConvMNIST(grad_type=grad_type)
elif (dataset == "CIFAR10"):
    if (connectivity == "fc"):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), transforms.Lambda(lambda x : torch.flatten(x))])
        trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)
        net = FCCIFAR(grad_type=grad_type)
    else:
        raise Exception ("conv is not working")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)
        net = ConvCIFAR(grad_type=grad_type)


def test_train(net, trainloader, testloader, name="", n_epochs=4, lr=0.001):

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optim = torch.optim.SGD(net.parameters(), lr=lr)

    train_losses = []
    train_counter = []
    test_losses = []
    test_accuracy = []

    def train(epoch):
        net.train()
        for batch_idx, (data, target_idx) in enumerate(trainloader):

            optim.zero_grad()
            if (connectivity == "fc"):
                data = torch.squeeze(data,dim=1).float().to(DEVICE)
            else:
                data = data.float().to(DEVICE)
            target_idx = target_idx.to(DEVICE)
            output = net(data)
            loss = loss_fn(output, target_idx)

            loss.backward()
            optim.step()

            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(trainloader.dataset)))
    

    def test():
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target_idx in testloader:
                if (connectivity == "fc"):
                    data = torch.squeeze(data).float().to(DEVICE)
                else:
                    data = data.float().to(DEVICE)
                target_idx = target_idx.to(DEVICE)
                target=nn.functional.one_hot(target_idx,num_classes=10).float()
                output = net(data)
                test_loss += loss_fn(output, target).item()
                pred_idx = torch.argmax(output.data, dim=-1)
                correct += pred_idx.eq(target_idx.data).sum().item()
            test_loss /= len(testloader.dataset)
            test_losses.append(test_loss)
            test_accuracy.append(100. * correct / len(testloader.dataset))
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(testloader.dataset),
                    100. * correct / len(testloader.dataset)))
            
    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
    
    torch.save(net.state_dict(), './results/{0}/model.pth'.format(name))
    torch.save(optim.state_dict(), './results/{0}/optimizer.pth'.format(name))
    
    return {"train_losses": train_losses, "train_counter": train_counter, "test_losses" : test_losses, "test_accuracy" : test_accuracy}


def main():
    try:
        os.mkdir("./results/{0}".format(name))
    except:
        print("Name already exists! Overwriting...")
    print(" dataset : {0} \n architecture : {1} \n algorithm : {2} \n device : {3}".format(dataset, args.connectivity, grad_type, DEVICE))
    losses = test_train(net, trainloader, testloader, name=name, n_epochs=n_epochs, lr=lr)
    torch.save(losses, './results/{0}/losses.pth'.format(name))

main()