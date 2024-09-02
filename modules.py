import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
import math

from globals import *
from agfunctions import *

class bpconv(nn.Module):

    def __init__(self, out_ch, in_ch, kh, kw, ih, iw, stride):
        """
        kh : kernel height
        kw : kernel width
        c_in : input channels
        c_out : output channels
        s : stride (only 1 or 2 supported)
        """
        super().__init__()
        W = torch.empty((out_ch, in_ch, kh, kw), dtype=DTYPE, device=DEVICE).normal_(mean=0.0,std=math.sqrt(2/in_ch))
        oh = int((ih - kh)/stride + 1)
        ow = int((iw - kw)/stride + 1)
        b = torch.zeros((out_ch, oh, ow), dtype=DTYPE, device=DEVICE)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)
        
        if (stride == 1):
            self.grad = bpconvgrad_s1.apply
        elif (stride == 2):
            self.grad = bpconvgrad_s2.apply
        else:
            raise Exception("stride can only be 1 or 2")

    def forward(self, x):
        z = self.grad(x, self.W, self.b)
        return z

    def update_backwards(self):
        #new_W = self.W.clone()
        #self.W_t = new_W.transpose(2,3)
        return

class pseudoconv(nn.Module):

    def __init__(self, out_ch, in_ch, kh, kw, ih, iw, stride):
        """
        kh : kernel height
        kw : kernel width
        c_in : input channels
        c_out : output channels
        s : stride (only 1 or 2 supported)
        """
        super().__init__()
        W = torch.empty((out_ch, in_ch, kh, kw), dtype=DTYPE, device=DEVICE).normal_(mean=0.0,std=math.sqrt(2/in_ch))
        W_copy = W.detach().clone()

        oh = int((ih - kh)/stride + 1)
        ow = int((iw - kw)/stride + 1)
        b = torch.zeros((out_ch, oh, ow), dtype=DTYPE, device=DEVICE)

        self.W = nn.Parameter(W)
        self.W_inv = nn.Parameter(get_pinv(W_copy))
        self.b = nn.Parameter(b)
        
        if (stride == 1):
            self.grad = pseudoconvgrad_s1.apply
        elif (stride == 2):
            self.grad = pseudoconvgrad_s2.apply
        else:
            raise Exception("stride can only be 1 or 2")

    def forward(self, x):
        z = self.grad(x, self.W, self.W_inv, self.b)
        return z

    def update_backwards(self):
        new_W = self.W.clone()
        W_inv = get_pinv(new_W)
        print(torch.norm(self.W_inv - W_inv, p='fro'))
        return
    
class randomconv(nn.Module):

    def __init__(self, out_ch, in_ch, kh, kw, ih, iw, stride):
        """
        kh : kernel height
        kw : kernel width
        c_in : input channels
        c_out : output channels
        s : stride (only 1 or 2 supported)
        """
        super().__init__()
        W = torch.empty((out_ch, in_ch, kh, kw), dtype=DTYPE, device=DEVICE).normal_(mean=0.0,std=math.sqrt(2/in_ch))
        B = torch.empty((out_ch, in_ch, kh, kw), dtype=DTYPE, device=DEVICE).normal_(mean=0.0,std=math.sqrt(2/in_ch))
        oh = int((ih - kh)/stride + 1)
        ow = int((iw - kw)/stride + 1)
        b = torch.zeros((out_ch, oh, ow), dtype=DTYPE, device=DEVICE)

        self.W = nn.Parameter(W)
        self.B = nn.Parameter(B)
        self.b = nn.Parameter(b)
        
        if (stride == 1):
            self.grad = randomconvgrad_s1.apply
        elif (stride == 2):
            self.grad = randomconvgrad_s2.apply
        else:
            raise Exception("stride can only be 1 or 2")

    def forward(self, x):
        z = self.grad(x, self.W, self.B, self.b)
        return z

    def update_backwards(self):
        # Never update random backwards weights
        return


class pseudolinear(nn.Module):

    def __init__(self, out_dim, in_dim):
        super().__init__()
        W = torch.empty((out_dim, in_dim), dtype=DTYPE, device=DEVICE).normal_(mean=0.0,std=math.sqrt(2/in_dim))
        W_copy = W.detach().clone()
        b = torch.zeros((out_dim), dtype=DTYPE, device=DEVICE) 
        self.W = nn.Parameter(W)
        self.W_inv = torch.linalg.pinv(W_copy)
        self.b = nn.Parameter(b)
        self.grad = pseudograd.apply

    def forward(self, x):
        a = self.grad(x, self.W, self.W_inv, self.b)
        return a

    def update_backwards(self):
        new_W = self.W.clone()
        self.W_inv = torch.linalg.pinv(new_W)
        #print(torch.norm(self.W_inv - W_inv, p='fro'))
        return
    

class bplinear(nn.Module):

    def __init__(self, out_dim, in_dim):
        super().__init__()
        W = torch.empty((out_dim, in_dim), dtype=DTYPE, device=DEVICE).normal_(mean=0.0,std=math.sqrt(2/in_dim))
        b = torch.zeros((out_dim), dtype=DTYPE, device=DEVICE) 
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)
        self.W_t = W.transpose(0,1)
        self.grad = bpgrad.apply

    def forward(self, x):
        a = self.grad(x, self.W, self.b)
        return a
    
    def update_backwards(self):
        new_W = self.W.clone()
        self.W_t = get_transpose(new_W)
        return


class randomlinear(nn.Module):

    def __init__(self, out_dim, in_dim):
        super().__init__()
        W = torch.empty((out_dim, in_dim), dtype=DTYPE, device=DEVICE).normal_(mean=0.0,std=math.sqrt(2/in_dim))
        B = torch.empty((in_dim, out_dim), dtype=DTYPE, device=DEVICE).normal_(mean=0.0,std=math.sqrt(2/in_dim))
        b = torch.zeros((out_dim), dtype=DTYPE, device=DEVICE) 
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)
        self.B = nn.Parameter(B)
        self.grad = randomgrad.apply

    def forward(self, x):
        a = self.grad(x, self.W, self.B, self.b)
        return a
    
    def update_backwards(self):
        # Never update random backwards matrix
        return
    

class InvertibleLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.grad = invertibleLeakyReLUgrad.apply
        self.s = torch.tensor([negative_slope],dtype=DTYPE,device=DEVICE)

    def forward(self, x):
        a = self.grad(x, self.s)
        return a