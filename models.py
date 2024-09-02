import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
import math

from globals import *
from agfunctions import *
from modules import *

class FullyConnected (nn.Module):

    def __init__(self, n_hidden=5, input_size=784,hidden_size=256, output_size=10, grad_type='backprop'):
        super().__init__()
        self.n_hidden=5
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size

        self.layers = nn.ModuleList()

        if (grad_type == 'backprop'):
            self.linearity = bplinear
            self.nonlinearity = nn.LeakyReLU(0.01)
        elif (grad_type == 'pseudo'):
            self.linearity = pseudolinear
            self.nonlinearity = nn.LeakyReLU(0.01)
        elif (grad_type == 'random'):
            self.linearity = randomlinear
            self.nonlinearity = nn.LeakyReLU(0.01)

        self.layers.append(self.linearity(self.hidden_size, self.input_size))

        for _ in range(n_hidden-1):
            self.layers.append(self.linearity(self.hidden_size, self.hidden_size))

        self.layers.append(self.linearity(self.output_size, self.hidden_size))
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
            x = x #(1 / torch.linalg.vector_norm(x,dim=-1))[...,None] * x # normalization
            #h = self.nonlinearity(x)
        #print(torch.linalg.vector_norm(x))
        return h

    def update_backwards(self):
        #for l in self.layers:
        #    l.update_backwards()
        return
    

class FCMNIST (nn.Module):

    def __init__(self, grad_type='backprop'):
        super().__init__()

        self.layers = nn.ModuleList()

        if (grad_type == 'backprop'):
            self.linearity = bplinear
            self.nonlinearity = nn.LeakyReLU(0.01)
        elif (grad_type == 'pseudo'):
            self.linearity = pseudolinear
            self.nonlinearity = nn.LeakyReLU(0.01) #InvertibleLeakyReLU(negative_slope=0.01)
        elif (grad_type == 'random'):
            self.linearity = randomlinear
            self.nonlinearity = nn.LeakyReLU(0.01)

        self.layers.append(self.linearity(400, 28*28))
        self.layers.append(self.linearity(200, 400))
        self.layers.append(self.linearity(100, 200))
        self.layers.append(self.linearity(50, 100))
        self.layers.append(self.linearity(10, 50))
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
            h = self.nonlinearity(x)
        return h

    def update_backwards(self):
        #for l in self.layers:
        #    l.update_backwards()
        return

class FCCIFAR (nn.Module):

    def __init__(self, grad_type='backprop'):
        super().__init__()

        self.layers = nn.ModuleList()

        if (grad_type == 'backprop'):
            self.linearity = bplinear
            self.nonlinearity = nn.LeakyReLU(0.01)
        elif (grad_type == 'pseudo'):
            self.linearity = pseudolinear
            self.nonlinearity = nn.LeakyReLU(0.01) #InvertibleLeakyReLU(negative_slope=0.01)
        elif (grad_type == 'random'):
            self.linearity = randomlinear
            self.nonlinearity = nn.LeakyReLU(0.01)

        self.layers.append(self.linearity(1000, 32*32*3))
        self.layers.append(self.linearity(500, 1000))
        self.layers.append(self.linearity(100, 500))
        self.layers.append(self.linearity(10, 100))
    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
            h = self.nonlinearity(x)
        return h

    def update_backwards(self):
        #for l in self.layers:
        #    l.update_backwards()
        return



class ConvMNIST (nn.Module):

    def __init__(self, grad_type="backprop"):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.lin_layers = nn.ModuleList()

        if (grad_type == 'backprop'):
            self.lin = bplinear
            self.conv = bpconv
        elif (grad_type == 'pseudo'):
            self.lin = pseudolinear
            self.conv = pseudoconv
        elif (grad_type == 'random'):
            self.lin = randomlinear
            self.conv = randomconv

        self.conv_layers.append(self.conv(32,1,4,4,28,28, 2))
        self.conv_layers.append(self.conv(64,32,3,3,13,13,2))
        self.lin_layers.append(self.lin(1024,2304))
        self.lin_layers.append(self.lin(10,1024))

        self.nonlinearity = nn.LeakyReLU(0.1)

        self.flatten = nn.Flatten(1,-1) # Flatten everything but the batch dimension

    def forward(self, x):
        for l in self.conv_layers:
            h = l(x)
            x = self.nonlinearity(h)
        
        x = self.flatten(x)
        for l in self.lin_layers:
            h = l(x)
            x = self.nonlinearity(h)
        
        z = x
        return z

    def update_backwards(self):
        for l in self.conv_layers:
            l.update_backwards()
        for l in self.lin_layers:
            l.update_backwards()
        return
    

class ConvCIFAR (nn.Module):

    def __init__(self, grad_type="backprop"):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.lin_layers = nn.ModuleList()

        if (grad_type == 'backprop'):
            self.lin = bplinear
            self.conv = bpconv
        elif (grad_type == 'pseudo'):
            self.lin = pseudolinear
            self.conv = pseudoconv
        elif (grad_type == 'random'):
            self.lin = randomlinear
            self.conv = randomconv

        self.conv_layers.append(self.conv(64,3,4,4,2))
        self.conv_layers.append(self.conv(128,64,3,3,2))
        self.conv_layers.append(self.conv(256,128,3,3,1))
        self.lin_layers.append(self.lin(1024,6400))
        self.lin_layers.append(self.lin(10,1024))

        self.nonlinearity = nn.LeakyReLU(0.1)

        self.flatten = nn.Flatten(1,-1) # Flatten everything but the batch dimension

    def forward(self, x):
        for l in self.conv_layers:
            h = l(x)
            x = self.nonlinearity(h)
        x = self.flatten(x)
        for l in self.lin_layers:
            h = l(x)
            x = self.nonlinearity(h)
        z = x
        return z

    def update_backwards(self):
        #for l in self.conv_layers:
        #    l.update_backwards()
        #for l in self.lin_layers:
        #    l.update_backwards()
        return