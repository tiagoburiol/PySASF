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
#import logging
#logging.basicConfig(level=logging.INFO)
import os

from pysasf.readers import read_datafile
from pysasf import stats
from pysasf import solvers
from pysasf import clarkeminella as cm
from IPython.display import clear_output



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
        #self.output_format='txt'
        self.solver_option = 'ols'
        self.feas = None
        
        
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
        
    def set_solver_option(self,solver_option):
        self.cm_solver_option = solver_option
        

    # SET OUTPUT SOLVER ##############################################################
    def set_output_folder(self, path):
        if path!=self.output_folder:
            self.output_folder = path
            print ('Setting output folder as:', path)
            # Setting the folder for output saves
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                print(f"Folder '{self.output_folder}' criated succesfully.")
            else:
                print(f"Folder to save output files is: '{self.output_folder}'.")

    # SAVE FILE ########################################################################
    def _save_array_in_file(self,array,output_folder,filename,fileformat,numberformat):
        self.set_output_folder(output_folder)
        if fileformat == 'txt':          
            if numberformat =='int':
                np.savetxt(self.output_folder+'/'+filename+'.'+fileformat,array, fmt='%s')
            if numberformat =='float32':
                np.savetxt(self.output_folder+'/'+filename+'.'+fileformat,array, fmt='%1.4f')

        
        if fileformat == 'gzip':
            import pyarrow as pa
            import pyarrow.parquet as pq
            if numberformat =='int':
                array = np.array(array).astype(np.uint16)
            if numberformat =='float32':
                array = np.array(array).astype(np.float32)
            t = pa.Table.from_arrays([array.ravel()],['col0'])
            pq.write_table(t, self.output_folder+'/'+filename+'.'+fileformat,
                           compression='gzip')
            
    # LOAD FILE ########################################################################
    def load_combs_and_props_from_files(self,fc,fp):
        #import pyarrow as pa
        import pyarrow.parquet as pq
        if fc[-3:]=='txt' and fp[-3:]=='txt':
            combs = np.loadtxt(fc).astype(int)
            Ps = np.loadtxt(fp)
            self.combs = combs
            self.props = Ps
        if fc[-4:]=='gzip' and fp[-4:]=='gzip':
            combs = np.array(pq.read_table(fc))
            array_shape = (int(len(combs)/4),4)
            combs = combs.reshape(array_shape)
            self.combs = combs
            
            Ps = np.array(pq.read_table(fp))
            array_shape = (int(len(Ps)/3),3)
            Ps = Ps.reshape(array_shape)
            self.props = Ps
        return combs, Ps

    def save_feasebles(self, Ps, output_folder,filename,fileformat,):
        booleans = [np.all(P>0) for P in Ps]
        self._save_array_in_file(booleans,output_folder,filename,fileformat,'int')
        print ('Feasebles boolean array is sabed in:',output_folder+'/'+filename)
        self.feas=booleans
        return None

    
   ################################################################################
    def calcule_all_props(self, solve_opt='ols'):
        from concurrent.futures import ThreadPoolExecutor, as_completed
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

        t1 = time.time()
        for k, comb in enumerate(combs):
            t2 = time.time()
            if t2-t1>1:
                clear_output(wait=True)
                percent = np.round(100*k/len(combs),2)
                print('Calculating proportion:', k, 'of', len(combs),
                      '(', percent ,'%)', end='\r', flush=True)
                t1=t2
                
            #clear_output(wait=False)
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
            fim = time.time()
            #clear_output(wait=True)
        return combs, Ps
    ###############################################################################

    
    
    '''
    calculate_and_save_all_proportions
    '''
    def calculate_and_save_all_proportions(self, format='txt', load=True):
        inicio = time.time()
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
        self.combs_filename = filename+'_combs'
        self.props_filename = filename+'_props'
        print('Saving combinations indexes in:',
              self.output_folder+'/'+self.combs_filename)
        print('Saving proportions calculated in:',
              self.output_folder+'/'+self.props_filename)
        #np.savetxt(self.output_folder+'/'+self.combs_filename, combs,fmt='%s')
        #np.savetxt(self.output_folder+'/'+self.props_filename, Ps, fmt='%1.4f')
        self._save_array_in_file(combs, self.output_folder, self.combs_filename, format,'int')
        self._save_array_in_file(Ps, self.output_folder, self.props_filename, format,'float32')
        self.save_feasebles(Ps,self.output_folder,filename+'_feas', format)
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




    