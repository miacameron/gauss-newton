import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
import torch.linalg as linalg
import matplotlib.pyplot as plt
from models import *
from plotting import *
import argparse
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()

parser.add_argument("--grad_type", help="type of weight transport algorithm (backprop, pseudo, random)")
parser.add_argument("--dataset", help="MNIST or CIFAR10")
parser.add_argument("--name", help='name of this experiment', default="unnamed")
parser.add_argument("--n_hidden", help="number of hidden layers", default=0)
parser.add_argument("--batch_size", help="training batch size", default=50)
parser.add_argument("--random_seed", help="random seed for pytorch", default=1234)
parser.add_argument("--epochs", help="number of epochs to run", default=3)

args = parser.parse_args()

torch.manual_seed(args.random_seed)
n_epochs = args.epochs
batch_size_train = args.batch_size
batch_size_test = 1
name = args.name

assert (args.grad_type in ["backprop", "pseudo", "random"])
grad_type = args.grad_type

match args.dataset:
    case "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x : torch.flatten(x))])
        trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)

        if (args.n_hidden == 0):
            n_hidden = 5
        else:
            n_hidden = args.n_hidden

        input_size = 28*28
        hidden_size = 256
        output_size = 10

    case "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), transforms.Lambda(lambda x : torch.flatten(x))])
        trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)

        if (args.n_hidden == 0):
            n_hidden = 3
        else:
            n_hidden = args.n_hidden

        input_size = 32*32*3
        hidden_size = 1024
        output_size = 10

    case _:
        raise Exception("MNIST or CIFAR10 only") 


def test_train(net, trainloader, testloader, name="", n_epochs=4, lr=0.0001):

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    train_losses = []
    train_counter = []
    test_losses = []
    test_accuracy = []

    def train(epoch):
        net.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            optim.zero_grad()
            for i in range(batch_size_train):
                d=data[i,:].float()
                t=nn.functional.one_hot(target[i,:],num_classes=10).float()
                o = net(d)
                loss += loss_fn(o, t)
            loss.backward()
            optim.step()
            with torch.no_grad():
                    net.update_backwards()
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
                data=torch.squeeze(data).float()
                target_idx = torch.squeeze(target_idx)
                target=nn.functional.one_hot(torch.squeeze(target_idx),num_classes=10).float()
                output = net(data)
                test_loss += loss_fn(output, target).item()
                pred_idx = torch.argmax(output.data)
                correct += pred_idx.eq(target_idx.data).sum()
            test_loss /= len(testloader.dataset)
            test_losses.append(test_loss)
            test_accuracy.append(100. * correct / len(testloader.dataset))
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(testloader.dataset),
                    100. * correct / len(testloader.dataset)))
            
    torch.save(net.state_dict(), './results/{0}/model.pth'.format(name))
    torch.save(optim.state_dict(), './results/{0}/optimizer.pth'.format(name))
            
    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
    
    return {"train_losses": train_losses, "train_counter": train_counter, "test_losses" : test_losses, "test_accuracy" : test_accuracy}


def main():
    net = FullyConnected(n_hidden=n_hidden, input_size=input_size, hidden_size=hidden_size, output_size=output_size, grad_type=grad_type)
    losses = test_train(net, trainloader, testloader, name=name)
    torch.save(losses, './results/{0}/losses.pth'.format(name))