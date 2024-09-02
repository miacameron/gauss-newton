import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import torch.linalg as linalg

from globals import *
from rankone_update import *

def get_transpose(A):
    A_ = torch.t(A)
    return A_

#@torch.compile
def get_pinv(A):
    A_ = torch.zeros((A.transpose(0,1).shape),dtype=DTYPE,device=DEVICE)
    # TODO THERE HAS GOT TO BE A BETTER WAY TO DO THIS 
    print(A.shape)
    for i in range(A.shape[2]):
        for j in range(A.shape[3]):
            A_[:,:,i,j] = torch.linalg.pinv(A[:,:,i,j],atol=0.001)
    return A_     


class invertibleLeakyReLUgrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, s):
        ctx.save_for_backward(s)
        return torch.max(s*x, x)

    @staticmethod
    def backward(ctx, d_out):
        s, = ctx.saved_tensors
        ds = None
        dx = torch.where(d_out < 0, (1/s)*d_out, d_out)
        return dx, ds
    
class pseudograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_inv, b):
        ctx.save_for_backward(x, W, W_inv)
        # Implementing batch support w transpose
        WT = W.transpose(0,1)
        a = x @ WT + b
        if (a != a).any():
            raise Exception("a : {}".format(a))
        return a
    
    @staticmethod
    def backward(ctx, d_out):
        x, W, W_inv = ctx.saved_tensors
        dx = dW = dW_inv = db = None
        #if (x.dim() == 2 and x.size(dim=0) != 1):
        #    print(x.shape)
        #    raise Exception('batch size > 1 not supported for pseudo during training')
        W_i = torch.linalg.pinv(W)
        W_invT = torch.transpose(W_i,0,1)
        dx = d_out @ W_invT
        dW = d_out.transpose(0,1) @ x #torch.outer(torch.squeeze(d_out), torch.squeeze(x))
        if (d_out != d_out).any():
            raise Exception("d_out : {}".format(d_out))
        if (x != x).any():
            raise Exception("d_out : {}".format(x))
        if (dW != dW).any():
            raise Exception("dW : {}".format(dW))
        if (W_inv != W_inv).any():
            raise Exception("W_inv : {}".format(W_inv))
        #dW_inv = -1 * (torch.linalg.pinv((W - dW)) - W_inv)
        #db = d_out
        return dx, dW, dW_inv, db
    

class bpgrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, b):
        # Implementing batch support w transpose
        ctx.save_for_backward(x, W)
        WT = torch.transpose(W,0,1)
        a = x @ WT + b
        if (a != a).any():
            raise Exception("a : {}".format(a))
        return a
    
    @staticmethod
    def backward(ctx, d_out):
        x, W = ctx.saved_tensors
        dx = dW = db = None
        #dx = torch.matmul(W_t, d_out)
        dx = d_out @ W
        #dW = torch.outer(d_out, x)
        dW = d_out.transpose(0,1) @ x
        if (d_out != d_out).any():
            raise Exception("d_out : {}".format(d_out))
        if (x != x).any():
            raise Exception("x : {}".format(x))
        if (dW != dW).any():
            raise Exception("dW : {}".format(dW))
        #db = d_out
        return dx, dW, db


class randomgrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, B, b):
        ctx.save_for_backward(x, B)
        # Implementing batch support w transpose
        WT = torch.transpose(W,0,1)
        a = x @ WT + b
        return a
    
    @staticmethod
    def backward(ctx, d_out):
        x, B = ctx.saved_tensors
        dx = dW = dB = db = None
        BT = torch.transpose(B,0,1)
        dx = d_out @ BT
        dW = (d_out[...,None] @ torch.transpose(x[...,None],1,2)).sum(dim=0)
        db = d_out
        return dx, dW, dB, db
    
# I dont know how to pass integer arguments to autograd functions, so I just made 2 for different strides
class bpconvgrad_s1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, b):
        ctx.save_for_backward(x,W)
        z = F.conv2d(x,W,stride=1) + b
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, W = ctx.saved_tensors
        dx = dW = db = None
        dW = F.conv2d(x.transpose(0,1), d_out.transpose(0,1)).transpose(0,1)
        dx = F.conv_transpose2d(d_out, W)
        db = d_out
        return dx, dW, db
    

class bpconvgrad_s2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, b):
        ctx.save_for_backward(x,W)
        z = F.conv2d(x,W,stride=2) + b
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, W = ctx.saved_tensors
        dx = dW = db = None
        dW = F.conv2d(x.transpose(0,1), d_out.transpose(0,1), dilation=2).transpose(0,1)
        dx = F.conv_transpose2d(d_out, W, stride=2)
        db = d_out
        return dx, dW, db
    

class randomconvgrad_s1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, B, b):
        ctx.save_for_backward(x,B)
        z = F.conv2d(x,W,stride=1) + b
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, B = ctx.saved_tensors
        dx = dW = dB = db = None
        dW = F.conv2d(x.transpose(0,1), d_out.transpose(0,1)).transpose(0,1)
        dx = F.conv_transpose2d(d_out, B)
        db = d_out
        return dx, dW, dB, db
    

class randomconvgrad_s2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, B, b):
        ctx.save_for_backward(x,B)
        z = F.conv2d(x,W,stride=2) + b
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, B = ctx.saved_tensors
        dx = dW = dB = db = None
        dW = F.conv2d(x.transpose(0,1), d_out.transpose(0,1), dilation=2).transpose(0,1)
        dx = F.conv_transpose2d(d_out, B, stride=2)
        return dx, dW, dB, db
    

class pseudoconvgrad_s1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_inv, b):
        ctx.save_for_backward(x,W_inv)
        z = F.conv2d(x,W,stride=1) + b
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, W_inv = ctx.saved_tensors
        dx = dW = dW_inv = db = None
        x_scaled = x #(1/(torch.linalg.vector_norm(x,dim=(2,3)).pow(4)))[...,None,None] * x
        dW = F.conv2d(x_scaled.transpose(0,1), d_out.transpose(0,1)).transpose(0,1)
        dx = F.conv_transpose2d(d_out, W_inv.transpose(0,1))
        dW_inv = rankone_convupdate(W_inv, d_out, dx, x_scaled, dW, 1)
        db = d_out
        return dx, dW, dW_inv, db
    

class pseudoconvgrad_s2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_inv, b):
        ctx.save_for_backward(x,W_inv)
        z = F.conv2d(x,W,stride=2) + b
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, W_inv = ctx.saved_tensors
        dx = dW = dW_inv = db = None
        x_scaled = x #(1/(torch.linalg.vector_norm(x,dim=(2,3)).pow(4)))[...,None,None] * x
        dW = F.conv2d(x_scaled.transpose(0,1), d_out.transpose(0,1), dilation=2).transpose(0,1)
        dx = F.conv_transpose2d(d_out, W_inv.transpose(0,1), stride=2)
        dW_inv = rankone_convupdate(W_inv, dx, x_scaled, dW, 2)
        db = d_out
        return dx, dW, dW_inv, db
    