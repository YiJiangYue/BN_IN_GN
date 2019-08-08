# -*- coding: utf-8 -*-
import numpy as np

def batch_normal(x, gamma, beta, bn_params, momentum=0.9, eps=1e-5):
    """
    x: NxCxHxW
    gamma, beta: 论文中的两个可学习参数
    bn_params: 保留的前面的均值和方差，test用
    momentum: 给均值和方差加窗平均
    
    """
    mean = np.mean(x, axis=(0,2,3), keepdims=True)
    var = np.var(x, axis=(0,2,3), keepdims=True)
    x_normal = (x-mean)/np.sqrt(var+eps)
    result = gamma*x_normal + beta
    
    bn_params["mean"] = momentum*bn_params["mean"] + (1-momentum)*mean
    bn_params["var"] = momentum*bn_params["var"] + (1-momentum)*var
    
    return result, bn_params

def instance_normal(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2,3), keepdims=True)
    x_normal = (x-mean)/np.sqrt(var+eps)
    result = gamma*x_normal + beta
    return result
    
def group_normal(x, gamma, beta, G, eps=1e-5):
    N, C, H, W = x.shape
    x1 = x.reshape((N, G, C//G, H, W))
    mean = np.mean(x1, axis=(2, 3, 4), keepdims=True)
    var = np.var(x1, axis=(2, 3, 4), keepdims=True)
    
    x_normal = (x1-mean)/np.sqrt(var+eps)
    result = gamma*x_normal + beta
    result.resize((N, C, H, W))
    return result
    
def test():
    x = np.array([i for i in range(96)])
    x1 = x.reshape((2, 3, 4, 4))

    bn_params={"mean": 0, "var": 0}
    result1, bn_params = batch_normal(x1, 0.2, 0.1, bn_params)
    
    result2 = instance_normal(x1, 0.2, 0.1)
    
    result3 = group_normal(x1, 0.2, 0.1, 3)
    
    print(result3.shape)
    
test()

