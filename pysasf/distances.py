#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import numpy as np
from scipy.spatial import distance


def mahalanobis_dist(P_arr, P0):
    S_inv = np.linalg.inv(np.cov(P_arr.T))
    dist = []
    for P in P_arr:
        d = distance.mahalanobis(P, P0, S_inv)       
        dist.append(d)
    return np.array(dist)

def euclidean_dist(P_arr, P0):
    dist = []
    for P in P_arr.T:
       d = distance.euclidean(P, P0)
       dist.append(d)
    return np.array(dist)
'''
# Same result of mahalanobis from scipy
def mahalanobis0_dist(P_arr, Pm):
    #print(P_arr.shape)
    #Pm = np.mean(P_arr, axis=0)
    #print(Pm)
    vect = np.full((len(P_arr),P_arr.shape[1]),Pm)-P_arr
    S = np.cov(P_arr.T)
    dist = []
    for v in vect:
        d = np.dot(np.dot(v.T, np.linalg.inv(S)), v)
        dist.append(d)
    dist = np.array(dist)
    return dist
'''

def mahalanobis2d_dist(P_arr, P0):
    P_arr = P_arr[:,0:2].T
    vect = np.array([P_arr[0]-P0[0], P_arr[1]-P0[1]]) 
    S = np.cov(P_arr)
    dist = []
    for v in vect.T:
        d = np.dot(np.dot(v.T, np.linalg.inv(S)), v)
        dist.append(d)
    # distâncias
    dist = np.array(dist)
    return dist













