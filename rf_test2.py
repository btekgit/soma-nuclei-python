# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:46:35 2013

@author: btek
"""

import vigra
import numpy as np
from sklearn import datasets

X, Y = datasets.make_classification(n_samples=4000, n_features=100, 
        n_informative=10, n_redundant=50, 
        n_repeated=0, n_classes=20, 
        n_clusters_per_class=10, 
        weights=None, flip_y=0.01, 
        class_sep=1.0, 
        hypercube=True, 
        shift=0.0, scale=1.0, shuffle=True, random_state=42)
X = X.astype(np.float32)
Y = Y.astype(np.uint32)
y = Y.reshape(-1, 1)

rf = vigra.learning.RandomForest(10)

rf.learnRF(X, y, 5)