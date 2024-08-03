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
import os


from pysasf.readers import read_datafile
from pysasf import stats
from pysasf import solvers
from pysasf import clarkeminella as cm

class BasinData:

    def __init__(self, filename):
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
        self.output_folder='output'
        self.solver_option = 'ols'
        
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
        return (stats.infos(self))
        
    def means(self):
        return (stats.means(self))
    
    def std(self):
        return (stats.std(self))

    def set_output_folder(self, path):
        self.output_folder = path
        print ('Setting output folder as:', path)
        # Setting the folder for output saves
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Folder '{self.output_folder}' criated succesfully.")
        else:
            print(f"Folder to save output files is: '{self.output_folder}'.")

    
   ################################################################################
    def calcule_all_props(self, solve_opt='ols'):
        from itertools import product
        df_dict = self.df_dict
        S_inv = self.S_inv
    
        filename = ''
        idx = []
        keys = list(df_dict.keys())
        for key in keys:
            size = len(df_dict[key])
            filename=filename+key+str(size)
            idx.append(list(np.arange(size)))
        self.filename = filename
    
        # Gera todas as combinações possíveis
        combs = list(product(*idx))
        total = len(combs)

        #cria um array vazio para todos os Ps
        Ps = np.empty((len(combs),len(keys)-1)).astype(float)

        for k, comb in enumerate(combs):
            X = []
            for i in range(len(comb)-1):
                key = keys[i]
                pos = comb[i]
                data = df_dict[key].values[pos]
                X.append(data.astype(float)) 
            y = df_dict['Y'].values[comb[-1]].astype(float)
            X = np.array(X).T
            #SOLVE
            if solve_opt == 'gls':
                P = solvers.solve_gls_4x4(y,X,S_inv)
            if solve_opt == 'ols':
                P = solvers.solve_ols_4x4(y,X)
            if solve_opt == 'opt':
                P = solvers.solve_minimize(y,X)
            Ps[k] = P
        return combs, Ps
    ################################################################################
    
    '''
    calculate_and_save_all_proportions
    '''
    def calculate_and_save_all_proportions(self, load=True):
        inicio = time.time()
        print('Calculating all proportions...')
        
        # Calculating...
        #combs, Ps = cm.props_from_all_combinations(self, 
        #                                       solve_opt = self.cm_solver_option,
        #                                       save=True)

        # Calculating...
        combs, Ps = self.calcule_all_props(solve_opt = self.cm_solver_option)
        
        fim = time.time()
        print ("Done! Time processing:",fim-inicio)
        print('Total combinations:',len(combs),', shape of proportions:', Ps.shape)
        inicio = time.time()

        # Saving
        self.set_output_folder(self.output_folder)
        filename = ''
        for key in self.df_dict.keys():
            filename = filename+key+str(len(self.df_dict[key]))
        self.combs_filename = filename+'_combs.txt'
        self.props_filename = filename+'_props.txt'
        print('Saving combinations indexes in:',
              self.output_folder+'/'+self.combs_filename)
        print('Saving proportions calculated in:',
              self.output_folder+'/'+self.props_filename)
        np.savetxt(self.output_folder+'/'+self.combs_filename, combs,fmt='%s')
        np.savetxt(self.output_folder+'/'+self.props_filename, Ps, fmt='%1.4f')
        fim = time.time()
        print ("Time for save files:",fim-inicio)
        
        # Loading if load option is choosed
        if load:
            c_name = self.output_folder+'/'+self.combs_filename
            p_name = self.output_folder+'/'+self.props_filename
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


    