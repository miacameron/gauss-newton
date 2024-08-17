import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import torch.linalg as linalg

from globals import *


def get_transpose(A):
    A_ = torch.t(A)
    return A_

def get_pinv(A):
    A_ = torch.linalg.pinv(A)
    if (A.dim() == 2):
        A_ = torch.linalg.pinv(A)
    elif (A.dim() == 4):
        A_ = torch.zeros((A.transpose(2,3).shape),dtype=DTYPE,device=DEVICE)
        # TODO THERE HAS GOT TO BE A BETTER WAY TO DO THIS 
        for i in range(A.shape[0]):
            A_[i,:,:,:] = torch.linalg.pinv(A[i,:,:,:],atol=0.1)
    else:
        A_ = None
    return A_

@torch.compile
def rankone_update(A_inv, dx, x, dA):
    beta = (dx * x).sum(dim=1) + 1
    #print(beta.shape)
    #print("k shape : {}".format(k.shape))
    #print("x shape : {}".format(x.shape))
    #print("h shape : {}".format(h.shape))
    #print("A_inv shape : {}".format(A_inv.shape))
    if (not beta.all()): # if beta is approximately 0
        k = dx
        h = (x @ A_inv)
        #print("beta is 0")
        k_inv = (1/torch.linalg.vector_norm(k).pow(2)) * torch.transpose(k[...,None],1,2)
        h_inv = (1/torch.linalg.vector_norm(h).pow(2)) * torch.transpose(h[...,None],1,2)
        kk_inv = k[...,None] @ k_inv
        hh_inv = h[...,None] @ h_inv
        k_invA_inv_h_inv = (1/torch.linalg.vector_norm(h).pow(2)) * k_inv @ A_inv @ h
        G = -1 * ((kk_inv @ A_inv) - (A_inv @ hh_inv)) + (k_invA_inv_h_inv) @ dA
    else:
        G = -1 * torch.reciprocal(beta)[...,None,None] * (A_inv @ dA @ A_inv)
    return G


def rankone_convupdate(A_inv, dx, x, s):
    G = torch.zeros((A_inv.shape),dtype=DTYPE,device=DEVICE)
    for i in range(A_inv.shape[0]):
        for j in range(A_inv.shape[1]):
            G[i,j,:,:] = rankone_update(A_inv[i,j,:,:], )


#@torch.compile
def rankone_convupdate_bad(A_inv, dx, x, s):
        beta = (x * dx).sum(dim=(0,2,3)) + 1
        beta_zeros = (beta < 1e-5).nonzero()
        beta_nonzeros = (beta >= 1e-5).nonzero()
        k = dx
        h = F.conv2d(x, A_inv.transpose(2,3), stride=s)
        kh = F.conv2d(dx.transpose(0,1), h.transpose(0,1), dilation=s).transpose(0,1)

        if (x != x).any() or not x.isfinite().all():
            raise Exception("x has a nan")
        
        if (dx != dx).any() or not dx.isfinite().all():
            raise Exception("dx has a nan")

        if (kh != kh).any() or not kh.isfinite().all():
            print(dx)
            raise Exception("kh has a nan")

        #if (torch.linalg.vector_norm(k,dim=(2,3)) < 1e-3).any():
        #    raise Exception("k = 0")
        #if (torch.linalg.vector_norm(h,dim=(2,3)) < 1e-3).any():
        #    raise Exception("h = 0")

        G = torch.zeros((A_inv.shape), dtype=DTYPE, device=DEVICE)
        G_nonzero = -1 * (torch.reciprocal(beta)[None,...,None,None] * kh)

        if (G_nonzero != G_nonzero).any() or not G_nonzero.isfinite().all():
            raise Exception("G_nonzero has a nan")

        if (len(beta_zeros) != 0):
            #print("beta is 0 somewhere")
            #print(beta_zeros)
            #print(beta_nonzeros)
            k_scaling_factor = torch.where((torch.linalg.vector_norm(k.flatten(2,3),dim=-1) < 1e-5), 0.0, (1/torch.linalg.vector_norm(k.flatten(2,3),dim=-1)**2))
            if (k_scaling_factor != k_scaling_factor).any():
                raise Exception("k_scaling_factor has a nan")
            k_scaled = k_scaling_factor[...,None,None] * k
            kk_inv = F.conv2d(k_scaled.transpose(0,1), k.transpose(0,1)).transpose(0,1)
            if (kk_inv != kk_inv).any():
                raise Exception("kk_inv has a nan")
            #print("k shape : {0}".format(k.shape))
            #print("h shape : {0}".format(h.shape))
            #print("kh shape : {0}".format(kh.shape))
            #print("kk_inv shape : {0}".format(kk_inv.shape))
            h_scaling_factor = torch.where((torch.linalg.vector_norm(h.flatten(2,3),dim=-1) < 1e-5), 0.0, (1/torch.linalg.vector_norm(h.flatten(2,3),dim=-1)**2))
            if (h_scaling_factor != h_scaling_factor).any():
                raise Exception("h_scaling_factor has a nan")
            h_scaled = h_scaling_factor[...,None,None] * h
            hh_inv = F.conv2d(h_scaled.transpose(0,1), h.transpose(0,1)).transpose(0,1)
            if (hh_inv != hh_inv).any():
                raise Exception("hh_inv has a nan")
            #print("hh_inv shape : {0}".format(hh_inv.shape))
            A_invh = F.conv_transpose2d(h_scaled, A_inv.transpose(2,3), stride=s)
            if (A_invh != A_invh).any():
                raise Exception("A_invh has a nan")
            #print("A_invh shape : {0}".format(A_invh.shape))
            #print("k_scaled : {0}".format(k_scaled.shape))
            #print("A_invh : {0}".format(A_invh.shape))
            k_invA_inv_h_inv = (k_scaled * A_invh).sum(dim=(0,2,3))
            if (k_invA_inv_h_inv != k_invA_inv_h_inv).any():
                raise Exception("k_invA_inv_h_inv has a nan")
            #print("k_invA_invH_inv shape : {0}".format(k_invA_inv_h_inv.shape))
            kk_invA_inv = F.conv2d(A_inv, kk_inv)
            if (kk_invA_inv != kk_invA_inv).any():
                raise Exception("kk_invA_inv has a nan")
            #print("kk_invA_inv shape : {0}".format(kk_invA_inv.shape))
            A_invhh_inv = F.conv2d(A_inv.transpose(0,1), hh_inv).transpose(0,1)
            if (A_invhh_inv != A_invhh_inv).any():
                raise Exception("A_invhh_inv has a nan")
            #print("kAh shape : {0}".format(k_invA_inv_h_inv.shape))
            t = (k_invA_inv_h_inv[None,:,None,None] * kh)
            if (t != t).any():
                print(torch.linalg.vector_norm(k.flatten(2,3),dim=-1))
                #print(k_scaling_factor)
                #print(k_invA_inv_h_inv)
                #print(t)
                raise Exception("t has a nan")
            #print("A_invhh_inv shape : {0}".format(A_invhh_inv.shape))
            G_zero = -1 * kk_invA_inv - A_invhh_inv + t 
            G[:,beta_zeros,:,:] = G_zero[:,beta_zeros,:,:]
            G[:,beta_nonzeros,:,:] = G_nonzero[:,beta_nonzeros,:,:] 
            if (G_zero != G_zero).any() or not G_zero.isfinite().all():
                raise Exception("G_zero has a nan")
        else:
            G = G_nonzero
        return G
        
        

class pseudograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_inv, b):
        ctx.save_for_backward(x, W_inv)
        # Implementing batch support w transpose
        WT = W.transpose(0,1)
        a = x @ WT + b
        return a
    
    @staticmethod
    def backward(ctx, d_out):
        x, W_inv = ctx.saved_tensors
        dx = dW = dW_inv = db = None
        W_invT = torch.transpose(W_inv,0,1)
        dx = d_out @ W_invT
        x_scaled = x #(1/(torch.linalg.vector_norm(x,dim=-1).pow(4)))[...,None] * x
        dW = d_out[...,None] @  torch.transpose(x_scaled[...,None],1,2)
        #beta = (dx * x).sum(dim=1) + 1
        #dW_inv = rankone_update(W_inv,dx,x,dW)
        #print((1/(torch.linalg.vector_norm(x,dim=-1)**2)))
        #print(torch.trace(dW.sum(dim=0)))
        #print(torch.trace(dW_inv.sum(dim=0)))
        db = d_out
        return dx, dW, dW_inv, db
    

class bpgrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_t, b):
        # Implementing batch support w transpose
        WT = torch.transpose(W,0,1)
        ctx.save_for_backward(x, WT)
        a = x @ WT + b
        return a
    
    @staticmethod
    def backward(ctx, d_out):
        x, W_t = ctx.saved_tensors
        dx = dW = dW_t = None
        W_tT = torch.transpose(W_t,0,1)
        #dx = torch.matmul(W_t, d_out)
        dx = d_out @ W_tT
        #dW = torch.outer(d_out, x)
        dW = d_out[...,None] @ torch.transpose(x[...,None],1,2)
        db = d_out
        return dx, dW, dW_t, db


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
        dW = d_out[...,None] @ torch.transpose(x[...,None],1,2)
        db = d_out
        return dx, dW, dB, db
    
"""
class bpconvgrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_t):
        ctx.save_for_backward(x,W)
        z = torch.einsum('bjci,oci->boj', [x, W])
        return z
    
    @staticmethod
    def backward(ctx, d_out):
        x, W = ctx.saved_tensors
        dx = dW = dW_t = None
        dW = torch.einsum('bjci,boj->boci', [x, d_out])
        dx = torch.einsum('oj,oci->ci', W, d_out)
        return dx, dW, dW_t
"""

# I dont know how to pass integer arguments to autograd functions, so I just made 2 for different strides
class bpconvgrad_s1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_t):
        ctx.save_for_backward(x,W.transpose(2,3))
        z = F.conv2d(x,W,stride=1)
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, W_t = ctx.saved_tensors
        dx = dW = dW_t = None
        dW = F.conv2d(x.transpose(0,1), d_out.transpose(0,1)).transpose(0,1)
        dx = F.conv_transpose2d(d_out, W_t.transpose(2,3))
        return dx, dW, dW_t
    

class bpconvgrad_s2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_t):
        ctx.save_for_backward(x,W.transpose(2,3))
        z = F.conv2d(x,W,stride=2)
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, W_t = ctx.saved_tensors
        dx = dW = dW_t = None
        dW = F.conv2d(x.transpose(0,1), d_out.transpose(0,1), dilation=2).transpose(0,1)
        dx = F.conv_transpose2d(d_out, W_t.transpose(2,3), stride=2)
        return dx, dW, dW_t
    

class randomconvgrad_s1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, B):
        ctx.save_for_backward(x,B)
        z = F.conv2d(x,W,stride=1)
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, B = ctx.saved_tensors
        dx = dW = dB = None
        dW = F.conv2d(x.transpose(0,1), d_out.transpose(0,1)).transpose(0,1)
        dx = F.conv_transpose2d(d_out, B.transpose(2,3))
        return dx, dW, dB
    

class randomconvgrad_s2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, B):
        ctx.save_for_backward(x,B)
        z = F.conv2d(x,W,stride=2)
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, B = ctx.saved_tensors
        dx = dW = dB = None
        dW = F.conv2d(x.transpose(0,1), d_out.transpose(0,1), dilation=2).transpose(0,1)
        dx = F.conv_transpose2d(d_out, B.transpose(2,3), stride=2)
        return dx, dW, dB
    

class pseudoconvgrad_s1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_inv):
        ctx.save_for_backward(x,W_inv)
        z = F.conv2d(x,W,stride=1)
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, W_inv = ctx.saved_tensors
        dx = dW = dW_inv = None
        x_scaled = x #(1/(torch.linalg.vector_norm(x,dim=(2,3)).pow(4)))[...,None,None] * x
        #print("x dim (s1) : {0}".format(x.shape))
        #print("1/|x|^2 (s1) : {0}".format(1/(torch.linalg.vector_norm(x,dim=(2,3))**2)))
        dW = F.conv2d(x_scaled.transpose(0,1), d_out.transpose(0,1)).transpose(0,1)
        dx = F.conv_transpose2d(d_out, W_inv.transpose(2,3))
        if (d_out != d_out).any() or not d_out.isfinite().all():
            raise Exception("d_out has a nan")
        if (W_inv != W_inv).any() or not W_inv.isfinite().all():
            raise Exception("W_inv has a nan")
        #a = F.conv2d(x, W_inv.transpose(2,3))
        #print("1/beta (s1) : {0}".format(str(torch.reciprocal(beta))))
        #beta = (x * dx).sum(dim=(2,3)) + 1
        #dW_inv = rankone_convupdate(W_inv, dx, x_scaled, 1)
        #-1 * torch.reciprocal(beta)[...,None,None] * F.conv2d(dx.transpose(0,1), a.transpose(0,1)).transpose(0,1)
        return dx, dW, dW_inv
    

class pseudoconvgrad_s2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W, W_inv):
        ctx.save_for_backward(x,W_inv)
        z = F.conv2d(x,W,stride=2)
        return z
    
    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        x, W_inv = ctx.saved_tensors
        dx = dW = dW_inv = None
        x_scaled = x #(1/(torch.linalg.vector_norm(x,dim=(2,3)).pow(4)))[...,None,None] * x
        if (x_scaled != x_scaled).any() or not x_scaled.isfinite().all():
            raise Exception("x_scaled has a nan")
        #print("x dim (s2) : {0}".format(x.shape))
        #print("1/|x|^2 (s2) : {0}".format(1/(torch.linalg.vector_norm(x,dim=(2,3))**2)))
        dW = F.conv2d(x_scaled.transpose(0,1), d_out.transpose(0,1), dilation=2).transpose(0,1)
        dx = F.conv_transpose2d(d_out, W_inv.transpose(2,3), stride=2)
        if (d_out != d_out).any() or not d_out.isfinite().all():
            raise Exception("d_out has a nan")
        if (W_inv != W_inv).any() or not W_inv.isfinite().all():
            raise Exception("W_inv has a nan")
        #a = F.conv2d(x, W_inv.transpose(2,3),stride=2)
        #beta = (x * dx).sum(dim=(2,3)) + 1
        #dW_inv = rankone_convupdate(W_inv, dx, x_scaled, 2)
        #-1 * torch.reciprocal(beta)[...,None,None] * F.conv2d(dx.transpose(0,1), a.transpose(0,1), dilation=2).transpose(0,1)
        return dx, dW, dW_inv
    