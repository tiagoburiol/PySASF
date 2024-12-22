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

        
def plot_cm_outputs(list_of_files, x_key, y_key, 
                    savefig = False, path='../output', fname = None):
    """
    Plots the inputs from multiple CSV (preferably) files' x and y keys (column 1 and column 2).

    This function reads data from a list of CSV files and plots the specified
    x and y keys. It can also save the plot as a .png file to a specified path.

    Parameters
    ----------
    list_of_files : list of str
        A list containing the paths to the CSV files to be plotted.
    x_key : str
        The key (column name) to be used for the x-axis (must match the file's label for the column).
    y_key : str
        The key (column name) to be used for the y-axis (must match the file's label for the column).
    savefig : bool, optional
        If True, saves the plot as a PNG file. Default is False.
    path : str, optional
        The directory path where the plot will be saved if `savefig` is True.
        Default is '../output'.

    Returns
    -------
    None
        The function does not return any value. It either displays or saves the plot.
    """
    if fname==None:
        filename = 'plot_'+str(x_key)+'_'+str(y_key)
    else:
        filename = fname
    
    plt.figure(figsize=(8, 4))  
    plt.grid()
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
        plt.close()
        print('Plot figure saved in:',path+'/'+filename+'.png')
    else:
        _plot()
        #plt.grid()
        #plt.show()
    return None
    
'''
Plot and save a convex hull figure from a list of points. 
If points is list or array mxn, m is de number of points and
n is de dimention of each point. Then, x_col and y_cols are 
the columns index corresponding of the x and y coordinates of 
points to be plotted.
'''
def draw_hull(P, x_col=0, y_col=1, savefig = False, 
              title='Convexhull', path=None, filename='convex_hull',fileformat='png'):
    if path==None:
        print('Please, set a path to save the convex hull figure.')
    
    from scipy.spatial import ConvexHull
    plt.figure(figsize=(4, 4))  

    points = np.vstack((P[:,x_col],P[:,y_col])).T
    #print(points.shape)
    hull = ConvexHull(points)
    
    #for simplex in hull.simplices:
        #plt.plot(points[simplex, 0], points[simplex, 1], 'k-',lw=1)
    def _plot():
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-',lw=1)
        #_ = convex_hull_plot_2d(hull)
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
        plt.savefig(path+'/'+filename+'.'+fileformat)
        print('Plot figure saved in:',path+'/'+filename+'.'+fileformat)
        plt.close()
    else:
        _plot()
        plt.show()
        
    return None #plt.gca()

def props_histo(props):
    n = props.shape[0]
    dim = props.shape[1]
    plt.figure(figsize=(5*dim,3))
    
    for j in range(dim):
        plt.subplot(1,dim,j+1)
        n, bins, patches = plt.hist(props[:,j], bins=50) 
        plt.title('P'+str(j+1))
    plt.show()

def props_histo(bd):
    """
    Plots a histogram for each property in the input array.

    Parameters
    ----------
    props : numpy.ndarray
        A 2D numpy array containing the proportions to be plotted.
        The array should have shape (n, dim), where n is the number
        of samples and dim is the number of proportions.

    Returns
    -------
    None
        The function does not return any value. It displays the plots.

    Raises
    ------
    None
    """
    
    bd.df_dict
    n = props.shape[0]
    dim = props.shape[1]
    plt.figure(figsize=(5*dim,3))
    
    for j in range(dim):
        plt.subplot(1,dim,j+1)
        n, bins, patches = plt.hist(props[:,j], bins=50) 
        plt.title('P'+str(j+1))
    plt.show()


def data_histo(bd):
    """
    Plots a histogram for each DataFrame associated with the sources s in the basin dataframe.

    Parameters
    ----------
    bd : object
        A basin's dataframe that contains:
        - sources: Each of the sources in the basin.
        - df_dict: A dictionary where keys are the sources 's' and values are the sources' chemical tracers compositions.

    Returns
    -------
    None
        The function does not return any value. It displays the histograms for each DataFrame.

    Raises
    ------
    None
    """
    for s in bd.sources:
        df = bd.df_dict[s]
        df.hist()



