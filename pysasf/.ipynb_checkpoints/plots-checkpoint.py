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

def draw_hull(P, ss, n, idx, title = "Convex Hull", savefig = True,
              xlabel = "P1", ylabel="P2"): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        from scipy.spatial import ConvexHull#, convex_hull_plot_2d
        points = P.T   
        hull = ConvexHull(points)

        #fig = plt.figure(figsize=(6, 4))
    
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

        #plt.plot(Pm[0],Pm[1], "ro")
        plt.plot(P[0],P[1], "k," )
    
        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'k-')
        plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'k')
        plt.title(title)
        plt.xlabel('P1')
        plt.ylabel('P2')
        plt.xlim([-0.1, 0.9])
        plt.ylim([-0.1, 0.9])
        if savefig == True:
            plt.savefig(title+'.png')
        plt.show()
        return hull


def plot_cm_outputs(list_of_files, x_key, y_key):
    fig = plt.figure(figsize=(8, 4))  
    
    for file in list_of_files:
        df = pd.read_csv(file)
        x = df.get(x_key)
        y = df.get(y_key)
        plt.plot(x, y, "o-")
        
    plt.title('CV% plots: sediments')
    plt.xlabel('Sample size')
    plt.ylabel('CV%, 50 runs: 95% Confidence Region of P1,P2')
    plt.grid()
    plt.show()
    return None
    



