import torch
import numpy as np
from globals import *

def rankone_update(A, A_inv, c, d):

    almost_zero = 1e-6


    if (c.dim() == 2):
        assert (c.size(dim=0) == 1)
        c = torch.squeeze(c)
    if (d.dim() == 2):
        assert (d.size(dim=0) == 1)
        d = torch.squeeze(d)
    
    v = A_inv @ c
    n = A_inv.transpose(0,1) @ d
    w = (torch.eye(A.shape[0]) - A @ A_inv) @ c
    m = (torch.eye(A.shape[1]) - A_inv @ A).transpose(0,1) @ d
    beta = 1 + torch.dot(d, v) #(v * d).sum(dim=0) + 1

    print("beta : {}".format(beta))

    is_m_zero = (torch.linalg.vector_norm(m) < almost_zero).any()
    is_w_zero = (torch.linalg.vector_norm(w) < almost_zero).any()
    is_beta_zero = (torch.abs(beta) < almost_zero)

    v_scaling = (1/(torch.linalg.vector_norm(v)**2))
    n_scaling = (1/(torch.linalg.vector_norm(n)**2))
    m_scaling = (1/(torch.linalg.vector_norm(m)**2))
    w_scaling = (1/(torch.linalg.vector_norm(w)**2))

    v_inv = v_scaling * v
    n_inv = n_scaling * n
    m_inv = m_scaling * m
    w_inv = w_scaling * w

    G = torch.linalg.pinv((A - LR*torch.outer(c,d))) - A_inv

    """
    # 6 Cases:
    if (is_w_zero and is_m_zero and is_beta_zero):
        print("Case 6")
        G = -1 * torch.outer(v, v_inv) @ A_inv - A_inv @ torch.outer(n, n_inv) + torch.dot(v_inv, (A_inv @ n)) * torch.outer(v,n_inv)
    elif (is_w_zero and not is_m_zero and is_beta_zero):
        print("Case 2")
        G = -1 * torch.outer(v, v_inv) @ A_inv - torch.outer(m_inv, n)
    elif (is_w_zero and is_m_zero and not is_beta_zero):
        print("Case 7")
        G = -1 * (1/beta) * torch.outer(v, n)
    elif (is_w_zero and not is_m_zero and not is_beta_zero):
        print("Case 3")
        p1 = ((torch.linalg.vector_norm(v)**2) / beta) * m + v
        p2 = ((torch.linalg.vector_norm(m)**2) / beta) * (A_inv.transpose(0,1) @ v) + n
        sig = torch.linalg.vector_norm(m)**2 * torch.linalg.vector_norm(v)**2 + torch.abs(beta)**2
        G = (1/beta) * torch.outer(m,v) @ A_inv - (beta/sig) * torch.outer(p1, p2)
    elif (not is_w_zero and not is_m_zero):
        print("Case 1")
        G = -1 * torch.outer(v,w_inv) - torch.outer(m_inv,n) + beta * torch.outer(m_inv,w_inv)
    elif (not is_w_zero and is_m_zero and is_beta_zero):
        print("Case 4")
        G = -1 * A_inv @ torch.outer(n,n_inv) - torch.outer(v,w_inv)
    elif (is_m_zero and not is_beta_zero):
        print("Case 5")
        p1 = ((torch.linalg.vector_norm(w)**2) / beta) * (A_inv @ n) + v
        p2 = ((torch.linalg.vector_norm(n)**2) / beta) * w + n
        sig = torch.linalg.vector_norm(n)**2 * torch.linalg.vector_norm(w)**2 + torch.abs(beta)**2
        G = (1/beta) * A_inv @ torch.outer(n,w) - (beta/sig) * torch.outer(p1,p2)
    else:
        raise Exception("Something's gone wrong in rankone_update...")
    """
    if (G != G).any():
        raise Exception("nan in G")

    return G


#@torch.compile
def rankone_convupdate(A_inv, dx, x, dA, s):
    if ((dA < 1e-5).all()):
        return None
    
    #print("dA : {}".format(dA.shape))
    #print("A_inv : {}".format(A_inv.shape))
    #print("dx : {}".format(dx.shape))
    #print("x : {}".format(x.shape))
    
    def conv_mult(A,B):
        assert(len(A.shape) == 4)
        assert(len(B.shape) == 4)
        C = torch.empty((A.shape[0], B.shape[1], A.shape[2], A.shape[3]), dtype=DTYPE, device=DEVICE)
        for i in range(B.shape[2]):
            for j in range(B.shape[3]):
                C[:,:,i,j] = A[:,:,i,j] @ B[:,:,i,j]
        return C
                
    beta = (dx * x).sum(dim=(2,3)).sum(dim=(0,1)) + 1
    kh = conv_mult(A_inv, conv_mult(dA, A_inv))

    #print("beta : {}".format(beta.shape) )
    #print("betkha : {}".format(kh.shape) )

    if (torch.abs(beta) < 1e-5).any(): # if beta is approximately 0
        k = dx
        h = torch.conv2d(x, A_inv.transpose(0,1), stride=s)
        k_scaling_factor = torch.where((torch.linalg.vector_norm(k,dim=(2,3)) < 1e-5), 0.0, (torch.reciprocal(torch.linalg.vector_norm(k,dim=(2,3))**2)))
        h_scaling_factor = torch.where((torch.linalg.vector_norm(h,dim=(2,3)) < 1e-5), 0.0, (torch.reciprocal(torch.linalg.vector_norm(h,dim=(2,3))**2)))
        k_inv = k_scaling_factor[...,None,None] * k 
        h_inv = h_scaling_factor[...,None,None] * h
        kk_inv = torch.conv2d(k.transpose(0,1), k_inv.transpose(0,1))
        #print("kk_inv : {}".format(kk_inv.shape))
        hh_inv = torch.conv2d(h.transpose(0,1), h_inv.transpose(0,1))
        #print("hh_inv : {}".format(hh_inv.shape))
        A_inv_h_inv = torch.conv_transpose2d(h_inv, A_inv.transpose(0,1), stride=s)
        k_invA_invh_inv = torch.conv2d(A_inv_h_inv, k_inv).sum(dim=(0,1))
        #print("k_invA_inv_h_inv : {}".format(k_invA_invh_inv.shape))
        #print("kh : {}".format(kh.shape))
        t = torch.squeeze(k_invA_invh_inv) * kh
        kk_invA_inv = torch.conv_transpose2d(kk_inv, A_inv)
        A_invhh_inv = torch.conv_transpose2d(hh_inv, A_inv.transpose(0,1)).transpose(0,1)
        G = -1 * kk_invA_inv - A_invhh_inv + t
    else:
        G = -1 * torch.reciprocal(beta) * kh

    if (G != G).any():
        if (h != h).any():
            raise Exception("h has nan")
        if (k != k).any():
            raise Exception("k has nan")
        if (h_inv != h_inv).any():
            raise Exception("h_inv has nan")
        if (k_inv != k_inv).any():
            raise Exception("k_inv has nan")
        raise Exception("nan in conv G")

    return G

#@torch.compile
def rankone_conv_update(A_inv, dx, x, dA, s):
    G = torch.zeros((A_inv.shape),dtype=DTYPE,device=DEVICE)
    in_ch, out_ch, kh, kw = A_inv.shape
    print("A_inv : {}".format(A_inv.shape))
    print("dx : {}".format(dx.shape))
    print("x : {}".format(x.shape))

    for i in range(A_inv.shape[2]):
        for j in range(A_inv.shape[3]):
            G[:,:,i,j] = rankone_update(A_inv[:,:,i,j], dx[:,:,s*(i+kh),s*(j+kw)], x[:,:,i,j], dA[:,:,i,j])
    return G