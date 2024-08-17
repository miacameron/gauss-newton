import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
import math

from globals import *
from agfunctions import *

class bpconv(nn.Module):

    def __init__(self, out_ch, in_ch, kh, kw, stride):
        """
        kh : kernel height
        kw : kernel width
        c_in : input channels
        c_out : output channels
        s : stride (only 1 or 2 supported)
        """
        super().__init__()
        W = torch.randn((out_ch, in_ch, kh, kw), dtype=DTYPE, device=DEVICE) / 10
        self.W = nn.Parameter(W)
        self.W_t = W.transpose(2,3)
        
        if (stride == 1):
            self.grad = bpconvgrad_s1.apply
        elif (stride == 2):
            self.grad = bpconvgrad_s2.apply
        else:
            raise Exception("stride can only be 1 or 2")

    def forward(self, x):
        z = self.grad(x, self.W, self.W_t)
        return z

    def update_backwards(self):
        new_W = self.W.clone()
        self.W_t = new_W.transpose(2,3)
        return

class pseudoconv(nn.Module):

    def __init__(self, out_ch, in_ch, kh, kw, stride):
        """
        kh : kernel height
        kw : kernel width
        c_in : input channels
        c_out : output channels
        s : stride (only 1 or 2 supported)
        """
        super().__init__()
        W = torch.randn((out_ch, in_ch, kh, kw), dtype=DTYPE, device=DEVICE) / 20
        self.W = nn.Parameter(W)
        self.W_inv = get_pinv(W)
        
        if (stride == 1):
            self.grad = pseudoconvgrad_s1.apply
        elif (stride == 2):
            self.grad = pseudoconvgrad_s2.apply
        else:
            raise Exception("stride can only be 1 or 2")

    def forward(self, x):
        z = self.grad(x, self.W, self.W_inv)
        return z

    def update_backwards(self):
        new_W = self.W.clone()
        self.W_inv = get_pinv(new_W)
        #print(torch.norm(self.W_inv - W_inv, p='fro'))
        return
    
class randomconv(nn.Module):

    def __init__(self, out_ch, in_ch, kh, kw, stride):
        """
        kh : kernel height
        kw : kernel width
        c_in : input channels
        c_out : output channels
        s : stride (only 1 or 2 supported)
        """
        super().__init__()
        W = torch.randn((out_ch, in_ch, kh, kw), dtype=DTYPE, device=DEVICE) / 10
        B = torch.randn((out_ch, in_ch, kw, kh), dtype=DTYPE, device=DEVICE) / 10
        self.W = nn.Parameter(W)
        self.B = nn.Parameter(B)
        
        if (stride == 1):
            self.grad = randomconvgrad_s1.apply
        elif (stride == 2):
            self.grad = randomconvgrad_s2.apply
        else:
            raise Exception("stride can only be 1 or 2")

    def forward(self, x):
        z = self.grad(x, self.W, self.B)
        return z

    def update_backwards(self):
        # Never update random backwards weights
        return


class pseudolinear(nn.Module):

    def __init__(self, out_dim, in_dim):
        super().__init__()
        W = torch.randn((out_dim, in_dim), dtype=DTYPE, device=DEVICE) / 20
        b = torch.randn((out_dim), dtype=DTYPE, device=DEVICE) 
        self.W = nn.Parameter(W)
        self.W_inv = get_pinv(W)
        self.b = nn.Parameter(b)
        self.grad = pseudograd.apply

    def forward(self, x):
        a = self.grad(x, self.W, self.W_inv, self.b)
        return a

    def update_backwards(self):
        new_W = self.W.clone()
        self.W_inv = get_pinv(new_W)
        #print(torch.norm(self.W_inv - W_inv, p='fro'))
        return
    

class bplinear(nn.Module):

    def __init__(self, out_dim, in_dim):
        super().__init__()
        W = torch.randn((out_dim, in_dim), dtype=DTYPE, device=DEVICE) / 10
        b = torch.randn((out_dim), dtype=DTYPE, device=DEVICE) 
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)
        self.W_t = get_transpose(W)
        self.grad = bpgrad.apply

    def forward(self, x):
        a = self.grad(x, self.W, self.W_t, self.b)
        return a
    
    def update_backwards(self):
        new_W = self.W.clone()
        self.W_t = get_transpose(new_W)
        return


class randomlinear(nn.Module):

    def __init__(self, out_dim, in_dim):
        super().__init__()
        W = torch.randn((out_dim, in_dim), dtype=DTYPE, device=DEVICE) / 100
        b = torch.randn((out_dim), dtype=DTYPE, device=DEVICE) / 10
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)
        self.B = nn.Parameter(torch.randn(((W.transpose(0,1)).shape), dtype=DTYPE, device=DEVICE) / 100)
        self.grad = randomgrad.apply

    def forward(self, x):
        a = self.grad(x, self.W, self.B, self.b)
        return a
    
    def update_backwards(self):
        # Never update random backwards matrix
        return
    