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

        
def plot_cm_outputs(list_of_files, x_key, y_key, 
                    savefig = True, path='output'):
    filename = 'plot_'+str(x_key)+'_'+str(y_key)
    fig = plt.figure(figsize=(8, 4))  

    def _plot():
        for file in list_of_files:
            df = pd.read_csv(file)
            x = df.get(x_key)
            y = df.get(y_key)
            plt.plot(x, y, "o-")
            plt.title('CV% plots: sediments')
            plt.xlabel('Sample size')
            plt.ylabel('CV%, 50 runs: 95% Confidence Region of P1,P2')
            plt.grid()
        
    if savefig:
        _plot()
        plt.grid()
        plt.savefig(path+'/'+filename+'.png')
        print('Plot figure saved in:',path+'/'+filename+'.png')
        plt.close()
    else:
        _plot()
        plt.grid()
        plt.show()
    return None
    
'''
Plot and save a convex hull figure from a list of points. 
If points is list or array mxn, m is de number of points and
n is de dimention of each point. Then, x_col and y_cols are 
the columns index corresponding of the x and y coordinates of 
points to be plotted.
'''
def draw_hull(P, x_col=0, y_col=1, savefig = False, 
              title='Convexhull', path='output', filename='convex_hull'):
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    
    
    fig = plt.figure(figsize=(4, 4))  

    points = np.vstack((P[:,x_col],P[:,y_col])).T
    #print(points.shape)
    hull = ConvexHull(points)
    
    #for simplex in hull.simplices:
        #plt.plot(points[simplex, 0], points[simplex, 1], 'k-',lw=1)
    def _plot():
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-',lw=1)
        Pm = np.mean(points, axis=0)
        plt.plot(Pm[0],Pm[1], "ro")
        plt.scatter(points[:,0],points[:,1],  marker=".", color='blue',s=0.5)
        plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'k-')
        plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'k')
        plt.title(title)
        plt.xlabel('P'+str(x_col+1))
        plt.ylabel('P'+str(y_col+1))
        plt.xlim((-0.1,1.0))
        plt.ylim((-0.1,1.0))
    if savefig:
        _plot()
        plt.savefig(path+'/'+filename+'.png')
        print('Plot figure saved in:',path+'/'+filename+'.png')
        plt.close()
    else:
        _plot()
        plt.show()
        
    return None #plt.gca()

def props_histo(props):
    n = props.shape[0]
    dim = props.shape[1]
    fig = plt.figure(figsize=(5*dim,3))
    
    for j in range(dim):
        ax = plt.subplot(1,dim,j+1)
        n, bins, patches = plt.hist(props[:,j], bins=50) 
        plt.title('P'+str(j))
    plt.show()

