import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.linalg as linalg
import math
'''
def LeakyReLU(x):
    return max(0.01*x, x)

def D_LeakyRelU(x):
    D = np.diag(0.01 if (x <= 0) else 1)
    return D
'''

def get_transpose(A):
    A_ = torch.t(A)
    return A_

def get_pinv(A):
    A_ = torch.linalg.pinv(A)
    return A_

class pseudograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_inv, b):
        ctx.save_for_backward(x, W_inv)
        a = torch.matmul(W,x) + b
        return a
    
    @staticmethod
    def backward(ctx, d_out):
        x, W_inv = ctx.saved_tensors
        dx = dW = dW_inv = None
        dx = torch.matmul(W_inv, d_out)
        dW = torch.outer(d_out, x)
        db = d_out
        return dx, dW, dW_inv, db
    

class bpgrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_t, b):
        ctx.save_for_backward(x, W_t)
        a = torch.matmul(W,x) + b
        return a
    
    @staticmethod
    def backward(ctx, d_out):
        x, W_t = ctx.saved_tensors
        dx = dW = dW_t = None
        dx = torch.matmul(W_t, d_out)
        dW = torch.outer(d_out, x)
        db = d_out
        return dx, dW, dW_t, db


class randomgrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, B, b):
        ctx.save_for_backward(x, B)
        a = torch.matmul(W,x) + b
        return a
    
    @staticmethod
    def backward(ctx, d_out):
        x, B = ctx.saved_tensors
        dx = dW = dB = None
        dx = torch.matmul(dB, d_out)
        dW = torch.outer(d_out, x)
        db = d_out
        return dx, dW, dB, db


class pseudo(nn.Module):

    def __init__(self, W, b):
        super().__init__()
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
        return
    

class backprop(nn.Module):

    def __init__(self, W, b):
        super().__init__()
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


class random(nn.Module):

    def __init__(self, W, b):
        super().__init__()
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)
        self.B = nn.Parameter(torch.randn((W.shape), dtype=torch.float))
        self.grad = randomgrad.apply

    def forward(self, x):
        a = self.grad(x, self.W, self.B, self.b)
        return a
    
    def update_backwards(self):
        # Never update random backwards matrix
        return
    

class FullyConnected (nn.Module):

    def __init__(self, n_hidden=5, input_size=784,hidden_size=256, output_size=10, grad_type='backprop'):
        super().__init__()

        self.n_hidden=5
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size

        self.nonlinearity = nn.LeakyReLU(0.1)

        if (grad_type == 'backprop'):
            self.linearity = backprop
        elif (grad_type == 'pseudo'):
            self.linearity = pseudo
        elif (grad_type == 'random'):
            self.linearity = random

        self.layers = [self.linearity(torch.randn((self.hidden_size, self.input_size),dtype=torch.float), 
                                      torch.randn((self.hidden_size), dtype=torch.float))]

        for _ in range(n_hidden-1):
            self.layers.append(self.linearity(torch.randn((self.hidden_size, self.hidden_size),dtype=torch.float),
                                              torch.randn((self.hidden_size), dtype=torch.float)))

        self.output = self.linearity(torch.randn((self.output_size, self.hidden_size),dtype=torch.float),
                                     torch.randn((self.hidden_size), dtype=torch.float))
    
    def forward(self, x):
        for l in self.layers:
            h = l(x)
            x = self.nonlinearity(h)
        y = self.output(x)
        #print("y shape: " + str(y.shape))
        return y

    def update_backwards(self):
        for l in self.layers:
            l.update_backwards()
        self.output.update_backwards()



"""
class test (nn.Module):

    def __init__(self):
        super().__init__()

        A = torch.from_numpy(np.random.normal(0,1,(10,10)))
        self.l1 = pseudo_module(A, mp_inverse(A))

    def forward(self, x):
        return self.l1(x)
"""