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
import time

from readers import read_datafile
import stats
import clarkeminella as cm
import os

class BasinData:

    def __init__(self, filename):
        from readers import read_datafile
        df = read_datafile(filename)
        # load data from file
        #df = pd.read_excel(filename)
        
        self.df_dict = {}
        self.cols = df.columns[1:]
        self.sources = []
        self.combs = None
        self.props = None
        #self.combs_filename = ''
        #self.props_filename = ''
        self.filename = ''
        self.output_format = 'bin'
        self.cm_df = None
        self.cm_all_Pfea = None
        self.cm_solver_option = 'ols'
        self.output_folder='../output'
        
        # split data on a dataframes dictionary
        names = df[df.columns[0]].unique()
        df_temp2 = pd.DataFrame()
        for name in names:
            data = df[df[df.columns[0]]==name].values[:,1:]
            if name[0:6]!='Target':
                df_temp = pd.DataFrame(data, columns=self.cols)
                self.df_dict[name]=df_temp
                self.sources.append(name)
            else:
                df_temp = pd.DataFrame(data,columns=self.cols)
                df_temp2 = pd.concat((df_temp2, df_temp))
        self.df_dict['Y']=df_temp2.reset_index(drop=True)

        Y = self.df_dict['Y'].values.astype(float)
        
        self.S_inv = np.linalg.inv(np.cov(Y.T))
        return None

    def infos(self):
        return stats.infos(self)
        
    def means(self):
        display(stats.means(self))
    
    def std(self):
        display(stats.std(self))

    def set_output_folder(self, path='../output'):
        # Setting the folder for output saves
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Folder '{self.output_folder}' criated succesfully.")
        else:
            print(f"Folder to save output files is: '{self.output_folder}'.")

    '''
    calculate_and_save_all_proportions
    '''
    def calculate_and_save_all_proportions(self, load=True):
        inicio = time.time()
        print('Calculating all proportions...')
        
        # Calculating...
        combs, Ps = cm.props_from_all_combinations(self, 
                                               solve_opt = self.cm_solver_option,
                                               save=True)
        
        fim = time.time()
        print ("Done! Time processing:",fim-inicio)
        print('Total combinations:',len(combs),', shape of proportions:', Ps.shape)
        inicio = time.time()

        # Saving
        self.set_output_folder()
        filename = ''
        for key in self.df_dict.keys():
            filename = filename+key+str(len(self.df_dict[key]))
        self.combs_filename = filename+'_combs.txt'
        self.props_filename = filename+'_props.txt'
        print('Saving combinations indexes in:',self.combs_filename)
        print('Saving proportions calculated in:',self.props_filename)
        np.savetxt(self.output_folder+'/'+self.combs_filename, combs,fmt='%s')
        np.savetxt(self.output_folder+'/'+self.props_filename, Ps, fmt='%1.4f')
        fim = time.time()
        print ("Time for save files:",fim-inicio)
        
        # Loading if load option is choosed
        if load:
            c_name = self.output_folder+'/'+self.combs_filename
            p_name = self.ooutput_folder+'/'+self.props_filename
            c,p = self.load_combs_and_props_from_files(c_name,p_name)
            self.combs = c
            self.props = p
        
        return None

  


    def load_combs_and_props_from_files(self,fc,fp):
        combs = np.loadtxt(fc).astype(int)
        Ps = np.loadtxt(fp)
        self.combs = combs
        self.props = Ps
        return combs, Ps

    def set_solver_option(self,solver_option):
        self.cm_solver_option = solver_option
        
    '''
    Calculate de confidence region
    '''    
    def confidence_region(self, P, p = 95, spacedist= 'mahalanobis'):
        if P.shape[0]>2:
            P=P[0:2,:]
        Pm = np.mean(P, axis=1)
        if spacedist=='mahalanobis':
            dist = self.mahalanobis_dist(P, Pm)
        if spacedist=='euclidean':
            dist = self.euclidean_dist(P)
        
        #dist = self.mahalanobis_dist(P, Pm)
        sorted_idx = np.argsort(dist)
        Psorted = P.T[sorted_idx].T
        end_idx = int((p/100)*len(Psorted.T))
        return (Psorted[:,:end_idx])



    def draw_hull(self, P, ss, n, idx, title = "Convex Hull", savefig = True,
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
    