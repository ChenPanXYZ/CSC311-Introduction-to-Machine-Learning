# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
# from scipy.special import logsumexp
from sklearn.datasets import load_boston

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    N = x_train.shape[0]
    y_hat = None
    # NOTE: test_datum is one test sample where we want to make a prediction on.
    # Step 1: compute distance-based weights for each training example a(i)
    a = []

    

    # need to matrixify test_datum, because l2 needs matrix.
    test_datum = np.reshape(test_datum, (test_datum.shape[0], 1))

    dist = l2(test_datum.transpose(), x_train).flatten() # dist should be of shape: N_train x 1.

    numerators = np.exp(-dist / (2 * tau**2))

    denominator = np.sum(
              np.exp(-dist[j] / (2 * tau**2)) for j in range(0, N)
              )
#     denominator = np.sum(np.exp(-dist / (2 * tau**2)))

#     for i in range(0, N):
#        numerator = numerators[i]
#        a.append(numerator / denominator)
       
       
#     a = np.array(a)

    a = np.array(
       [numerators[i] / denominator for i in range(0, N)]
    )

    # Step 2: compute w*, numpy.linalg.solve: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html

    A = np.diag(a)
    I = np.diag([1] * test_datum.shape[0])

    w_star = np.linalg.solve((np.matmul(np.matmul(x_train.transpose(), A), x_train) + lam * I), np.matmul(np.matmul(x_train.transpose(), A), y_train))

    # Step 3: computer y_hat (easy part)
    y_hat = np.dot(test_datum.transpose(), w_star)

    return y_hat
    # return None
    ## TODO




def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    ## TODO

    # Reference: https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros


    training_count = int(np.round(0.7 * x.shape[0]))
    validation_count = x.shape[0] - training_count
    
#     indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = idx[:training_count], idx[training_count:]
    x_train, x_validation = x[training_idx,:], x[test_idx,:]
    y_train, y_validation = y[training_idx,], y[test_idx,]


    train_losses = []
    validation_losses = []
    for tau in taus:
       # for training
       loss = 0

       for i in range(0, len(x_validation)):
              y_hat = LRLS(x_validation[i], x_train, y_train, tau,lam=1e-5)
              loss += 1/2 * ((y_hat- y_validation[i]) ** 2)

       validation_losses.append(loss / validation_count)

    #    print(loss / validation_count)

    # print(train_losses)
    return validation_losses
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(train_losses)

    plt.savefig('q4.png')
    # plt.semilogx(test_losses)

