import torch


def iterative_pseudoinverse(A, X_0, iter, eps=1.0):

    X = X_0
    X_list = []

    y = torch.randn((iter, A.shape[0])) 

    for i in range(iter):
        y_0 = y[i,:]
        y_1 = X @ y_0
        y_2 = A @ y_1
        delta_X = torch.outer((y_2 - y_0), y_1)
        new_X = X + delta_X
        X_list.append(new_X)
        X = new_X

    return X_list