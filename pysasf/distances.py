#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy.linalg import solve
import seaborn as sns
from matplotlib.legend import Legend
import h5py
from scipy.spatial import distance

def mahalanobis_dist(P_arr, Pm):
    S = np.cov(P_arr.T)
    dist = []
    for P in P_arr:
        d = distance.mahalanobis(P, Pm, np.linalg.inv(S))       
        dist.append(d)
    return np.array(dist)

def euclidean_dist(P_arr, Pm):
    from scipy.spatial import distance
    #Pm = np.mean(P_arr, axis=1)
    dist = []
    for P in P_arr.T:
       d = distance.euclidean(P, Pm)
       dist.append(d)
    return np.array(dist)

def mahalanobis0_dist(P_arr, Pm):
    P_arr = P_arr[:,0:2].T
    vect = np.array([P_arr[0]-Pm[0], P_arr[1]-Pm[1]]) 
    S = np.cov(P_arr)
    #print(vect.shape)
    dist = []
    for v in vect.T:
        d = np.dot(np.dot(v.T, np.linalg.inv(S)), v)
        dist.append(d)
    
    # distâncias
    dist = np.array(dist)
    return dist














