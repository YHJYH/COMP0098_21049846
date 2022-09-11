# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 03:22:15 2022

@author: Yuheng
"""
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns 
from matplotlib import pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# https://discuss.pytorch.org/t/parameters-initialisation/20001
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        # https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/

def feature_loader(number: str) -> torch.Tensor:
    """
    Args:
        data_type: str, dataset used, located to corresponding file location.
    Interim: 
        feature_map: Dict[List[Tensor]]
        Dict size: number of total batches e.g. lenet5 = 10
        List size: number of layers e.g. lenet5=15 vgg16=16
        Tensor size: number of samples in one batch e.g. lenet5=1000
        E.g. lenet5:
            feature_map[2][9][400]: the feature map of 401-th sample in layer-10 from batch number 3. Indexed from 0.
    Returns:
        feature_map: Dict[List[numpy.array]]
    """
    feature_map = torch.load(r'output\cifar10\features%s.pt' % number)
    for key in feature_map.keys():
        for i in range(len(feature_map[key])):
            feature_map[key][i] = feature_map[key][i].cpu()
            feature_map[key][i] = feature_map[key][i].numpy()
    
    return feature_map

'''
def gram_linear(x):
    """
    Gram matrix for linear kernel.
    Args:
        x: Tensor, feature map of an input at a layer. shape: n * m
            n: number of samples
            m: number of features
    Returns:
        gram: Tensor, a gram matrix of shape n * n.
    """
    linear_gram = torch.matmul(x, x.T)
    return linear_gram
'''

def feature_reshape(x):
    
    batch_size = x.shape[0]
    res = np.reshape(x, (batch_size, -1))
    
    return res

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram

def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)

def cka_avg_new(feature_map_1, feature_map_2, gram_kernel):
    """
    Args:
        feature_map: feature map, Dict.
        gram_kernel: gram matrix used with chosen type of kernel. Function.
    Returns:
        average_cka: num_layers * num_layers matrix. Each entry is an averaged CKA value.
    """    
    res = []
    num_batches_1 = len(feature_map_1)
    num_batches_2 = len(feature_map_2)
    for key_1 in feature_map_1.keys():
        for key_2 in feature_map_2.keys():
            num_layers_1 = len(feature_map_1[key_1])
            num_layers_2 = len(feature_map_2[key_2])
            cka_vals = np.zeros((num_layers_1, num_layers_2))
            for i in range(num_layers_1):
                layer1 = feature_map_1[key_1][i]
                batch_size = len(layer1)
                layer1 = np.reshape(layer1, (batch_size, -1))
                gram1 = gram_kernel(layer1)
                for j in range(num_layers_2):
                    layer2 = feature_map_2[key_2][j]
                    layer2 = np.reshape(layer2, (batch_size, -1))
                    gram2 = gram_kernel(layer2)
                    
                    cka_vals[i][j] = cka(gram1, gram2)
                    
            res.append(cka_vals)
    
    average_cka = sum(res) / len(res)
  
    return average_cka

def cka_avg(feature_map, gram_kernel):
    """
    Args:
        feature_map: feature map, Dict.
        gram_kernel: gram matrix used with chosen type of kernel. Function.
    Returns:
        average_cka: num_layers * num_layers matrix. Each entry is an averaged CKA value.
    """    
    res = []
    num_batches = len(feature_map)
    for key in feature_map.keys():
        num_layers = len(feature_map[key])
        cka_vals = np.zeros((num_layers, num_layers))
        for i in range(num_layers):
            layer1 = feature_map[key][i]
            batch_size = len(layer1)
            layer1 = np.reshape(layer1, (batch_size, -1))
            gram1 = gram_kernel(layer1)
            for j in range(num_layers):
                layer2 = feature_map[key][j]
                layer2 = np.reshape(layer2, (batch_size, -1))
                gram2 = gram_kernel(layer2)
                
                cka_vals[i][j] = cka(gram1, gram2)
                
        res.append(cka_vals)
    
    average_cka = sum(res) / len(res)
  
    return average_cka


def sim_plot(sim_matrix, label='CKA (Linear)', title='LeNet-5 MNIST EPOCH=50'):
    
    NX, NY = sim_matrix.shape
    sns.heatmap(sim_matrix, linewidth=0.5, cmap='coolwarm', cbar_kws={'label': label})
    plt.xlabel('layers')
    plt.ylabel('layers')
    plt.title(title)
    plt.xlim(0, NY)
    plt.ylim(0, NX)
    plt.plot()
    
