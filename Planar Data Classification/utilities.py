from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid


def relu(Z):

    A = np.maximum(0,Z)
    if type(Z) is np.ndarray:
        assert(A.shape == Z.shape)
    
    return A 


def relu_grad(Z):
    grad = np.zeros(Z.shape)
    grad[Z>0] =1 
    if type(Z) is np.ndarray:
        assert(grad.shape == Z.shape)
    
    return grad


def tanh_grad(Z):
    grad = (1 - np.tanh(Z)**2)
    if type(Z) is np.ndarray:
        assert(grad.shape == Z.shape)
        
    return grad


def softmax(Z):
    A = np.exp(Z)/(np.sum(np.exp(Z),axis = 0))
    assert(A.shape == Z.shape)
    
    return A


def dict_to_vector(params,grads):
    
    total = 0
    L = len(params) // 2
    theta = theta_grads = np.empty(0)
    for l in range(1,L+1):
        total += (params['W'+str(l)].size + params['b'+str(l)].size)
        theta = np.append(theta,params['W'+str(l)])
        theta = np.append(theta,params['b'+str(l)])
        
        theta_grads = np.append(theta_grads,grads['dW'+str(l)])
        theta_grads = np.append(theta_grads,grads['db'+str(l)])
    
    assert(total == theta.size)
    return theta.reshape(-1,1),theta_grads.reshape(-1,1)


def vector_to_dict(theta,p):
    
    L = len(p) // 2
    params = {}
    pos = 0 
    for l in range(1,L+1):
        w_size = p['W'+str(l)].size
        b_size = p['b'+str(l)].size
        params['W'+str(l)] = theta[pos:pos+w_size].reshape(p['W'+str(l)].shape)
        pos += w_size
        params['b'+str(l)] = theta[pos:pos+b_size].reshape(p['b'+str(l)].shape)
        pos +=b_size
        assert(params['W'+str(l)].shape == p['W'+str(l)].shape)
        assert(params['b'+str(l)].shape == p['b'+str(l)].shape)
    
    return params

    
def init_for_adam(params):
    
    L = len(params) //2
    v = {}
    s = {}
    
    for l in range(L):
        
        v["dW" + str(l+1)] = np.zeros(params['W'+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(params['b'+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(params['W'+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(params['b'+str(l+1)].shape)
        
    return v, s

def one_hot(y):
    C = np.unique(y).size
    y_hot = np.eye(C)[:,y]
    
    return y_hot
    
    
# Initializing the grid
def grid(X,t=1):
    xmin, xmax = X[0].min() -t, X[0].max() +t
    ymin, ymax = X[1].min() -t, X[1].max() +t
    u = np.linspace(xmin,xmax,500, dtype=np.float32)
    v = np.linspace(ymin,ymax,500, dtype=np.float32)
    xx,yy = np.meshgrid(u,v,indexing='xy')
    Z = np.c_[xx.ravel(),yy.ravel()].T

    return xx,yy,Z




    

