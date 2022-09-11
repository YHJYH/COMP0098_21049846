# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:53:49 2022

@author: Yuheng
"""

from utils import *

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
'''
fm1 = feature_loader('5_1')
sim_matrix1 = cka_avg(fm1, gram_linear)
s1 = cka_avg(fm1, gram_rbf)
del fm1

fm2 = feature_loader('5_2')
sim_matrix2 = cka_avg(fm2, gram_linear)
s2 = cka_avg(fm2, gram_rbf)
del fm2

fm3 = feature_loader('5_3')
sim_matrix3 = cka_avg(fm3, gram_linear)
s3 = cka_avg(fm3, gram_rbf)
del fm3

fm4 = feature_loader('5_4')
sim_matrix4 = cka_avg(fm4, gram_linear)
s4 = cka_avg(fm4, gram_rbf)
del fm4

fm5 = feature_loader('5_5')
sim_matrix5 = cka_avg(fm5, gram_linear)
s5 = cka_avg(fm5, gram_rbf)
del fm5

sim_matrix = sim_matrix1 + sim_matrix2 + sim_matrix3 + sim_matrix4 + sim_matrix5
sim_matrix /= 5

s = s1 + s2 + s3 + s4 + s5
s /= 5
'''
fm1 = feature_loader('')
fm2 = feature_loader('')
sim_matrix = cka_avg_new(fm1, fm2, gram_linear)
s = cka_avg_new(fm1, fm2, gram_rbf)
del fm1, fm2

t = 'Similarity Matrix CIFAR10'
sim_plot(sim_matrix, label='CKA (Linear)', title=t)
#sim_plot(s, label='CKA (RBF)',title=t)
#t2 = 'VGG16-10% CIFAR10'
#sim_plot(s, label='CKA (RBF)', title=t)

'''
NX, NY = sim_matrix.shape
sns.heatmap(sim_matrix, linewidth=0.5, cmap='coolwarm', cbar_kws={'label': 'Linear CKA'})
plt.xlabel('layers')
plt.ylabel('layers')
plt.title('LeNet-5 MNIST EPOCH=50')
plt.xlim(0, NX)
plt.ylim(0, NY)
plt.plot()
'''