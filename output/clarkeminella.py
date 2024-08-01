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
from itertools import product
from IPython.display import clear_output
from itertools import product
#from random import randrange
#from scipy.linalg import solve
#import seaborn as sns
#from matplotlib.legend import Legend
#import h5py
from distances import mahalanobis_dist, mahalanobis0_dist
import solvers
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def random_subsamples(bd,nlist):
    df_dict = bd.df_dict
    ss_dict = {}
    for i, key in enumerate(df_dict.keys()):
        data = df_dict[key].values.astype(float)
        ss_data = data[np.random.choice(len(data), nlist[i], replace=False)]
        ss_dict[key]=data[np.random.choice(len(data), nlist[i], replace=False)]
    return ss_dict

def random_subsamples2(bd,nlist):
    df_dict = bd.df_dict

    ss_df_dict = {}
    for i, key in enumerate(df_dict.keys()):
        data = df_dict[key].values.astype(float)
        ss_data = data[np.random.choice(len(data), nlist[i], replace=False)]
        ss_df_dict[key]=pd.DataFrame(ss_data, columns = bd.cols)
    return ss_df_dict




'''
Esta função processa o cálculo das porcentagens usando as médias
Faz as contas usando reps repetições tomando aletoriamente um conjunto
de subamostras com tamanhos definidos por n_list. Tem opção de plotar. 
'''
def get_props_from_subsamples_means(bd,reps,n_list, plot=True):
    df_dict = bd.df_dict
    Y = bd.df_dict['Y'].values.astype(float)
    S_inv = np.linalg.inv(np.cov(Y.T))
    
    if len(df_dict.keys())!=len(n_list):
        print("The n_list needs to be of the same size of lands data.")
    else:
        Ps = []
        for i in range(reps):
            ss_dict = random_subsamples(bd,n_list)
            X = []
            for key in bd.sources:
                X.append(ss_dict[key].mean(axis=0))
            X = np.array(X).T
            y = ss_dict['Y'].mean(axis=0)
            P = solvers.solve_gls_4x4(y,X,S_inv)
            #print(P)
            if np.all(np.array(P[0:3])>=0):
                #print (P[0:3], 'soma:', np.sum(P[0:3]))    
                Ps.append(list(P))
        Ps = np.array(Ps)
        if plot:
            plt.plot(Ps[:,0],Ps[:,1],'.')
            plt.xlabel("P1")
            plt.ylabel("P2")
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.plot(Ps[:,0],Ps[:,1],'.')
            plt.show()
        return(Ps)

################################################################################
def props_from_all_combinations(bd, solve_opt='ols',save=True):
    from itertools import product
    df_dict = bd.df_dict
    S_inv = bd.S_inv
    
    filename = ''
    idx = []
    keys = list(df_dict.keys())
    for key in keys:
        size = len(df_dict[key])
        filename=filename+key+str(size)
        idx.append(list(np.arange(size)))
    bd.filename = filename
    
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


def props_from_all_combinations_BACKUP(df_dict, S_inv):
    #counter1 = 0; counter2 = 0; percent1 = 0; percent2 = 0
    
    idx = []
    keys = list(df_dict.keys())
    for key in keys:
        size = len(df_dict[key])
        idx.append(list(np.arange(size)))
    # Gera todas as combinações possíveis
    combs = list(product(*idx))
    total = len(combs)
    print (combs.shape)

    P1=[];P2=[];P3=[]

    for comb in combs:
        X = []
        for i in range(len(comb)-1):
            key = keys[i]
            pos = comb[i]
            data = df_dict[key].values[pos]
            X.append(data.astype(float)) 
        y = df_dict['Y'].values[comb[-1]]
        X = np.array(X).T
        #counter1+=1


        #SOLVE
        P = solve_gls_4x4(y,X,S_inv)

        
        if np.all(P[0:3]>=0):
            #counter2+=1
            #percent1 =  counter1/total
            #percent2 =  counter2/counter1
            P1.append(P[0])
            P2.append(P[1])
            P3.append(P[2])
  

    #clear_output(wait=True)
    #print(f'Found {counter1} solutions with {counter2}({percent2:.2%}) feasibles.')
    return np.array([P1, P2, P3])




#def props_from_all_combinations_and_reduction(bd, n_list):
#    ss_dict = random_subsamples2(bd, n_list)
#    Ps = props_from_all_combinations(ss_dict, bd.S_inv)
#    return Ps

def randon_props_subsamples(bd, key, n):
    Ps = bd.props
    combs = bd.combs
    key_idx = list(bd.df_dict.keys()).index(key)
    size = len(bd.df_dict[key])
    rand = np.random.choice(np.arange(size), n, replace=False)
    selected_combs = combs[np.where(np.isin(combs[:,key_idx],rand))]
    selected_Ps = Ps[np.where(np.isin(combs[:,key_idx],rand))]
    return selected_Ps


def cm_feasebles(Ps):
    Ps_feas = []
    for P in Ps:
        #if np.all(P[0:2]>=0) and np.sum(P[0:2]<=1):
        if np.all(P>0) and np.sum(P<1):
           Ps_feas.append(P)
    return np.array(Ps_feas)


def run_repetitions_and_reduction (bd, key, 
                                        reductions,
                                        repetitions = 50):
    cv = lambda x: np.std(x) / np.mean(x) *100
    CVs = []
    areas_medias = []
    filename = bd.filename+'_'+str(key)
    
    # create the columns names for return dataframe
    df_out_cols = ['nSamp','CV','Mean','Std', 'Total', 'Feas']
    nSources = len(bd.sources)
    for i in range(nSources):
        df_out_cols.append('MeanP'+str(i+1))
    df_out_data = []

    # get data from df for reductions
    data = bd.df_dict[key].values.astype(float)
    
    for n in reductions:
        areas = []
        filename = filename+'-'+str(n)
        for i in range(repetitions):
            clear_output(wait=True)
            print ('Processing for', n, 'subsamples of',key, ', repetition number', i)

            Ptot = randon_props_subsamples(bd, key, n)
            Pfea =  cm_feasebles(Ptot) 
            if Pfea.shape[0]>=4:
                Pcr = confidence_region(Pfea[:,0:2], p = 95)
                hull = ConvexHull(Pcr)
                areas.append(hull.volume)
        
        # insert data for df_out
        nSamp = n
        CV = np.round(cv(areas),4)
        Mean = np.round(np.mean(areas),4)
        Std = np.round(np.std(areas),4)
        Total = len(Ptot)
        Feas = len(Pfea)
        df_out_data_n = [nSamp,CV,Mean,Std,Total,Feas]
        for i in range(nSources):
            df_out_data_n.append(np.mean(Pfea, axis=0)[i])
        df_out_data.append(df_out_data_n)
        bd.cm_df = df_out_data

        CVs.append(cv(areas))
        areas_medias.append(np.mean(areas))

   
    print('Done!')
    clear_output(wait=True)
    
    df_out = pd.DataFrame(df_out_data, columns=df_out_cols)
    display(df_out)
    bd.cm_df = df_out

    # Saving file in cvs
    print ('Saving in', filename+'.csv')
    df_out.to_csv(filename+'.csv')
    
    return (df_out)


def props_whith_repetitions_and_reduction (df_dict, S_inv, key,
                                           reductions,
                                           repetitions = 50):
    from scipy.spatial import ConvexHull
    from IPython.display import clear_output

    # create the columns names for return dataframe
    df_out_cols = ['nSamp','CV','Mean','Std', 'Total', 'Feas']
    nSources = len(df_dict.keys())
    for i in range(nSources):
        df_cols.append('MeanP'+str(i))
    df_out_data = []
    
    # coeficiet of variability
    cv = lambda x: np.std(x) / np.mean(x) *100
    
    CVs = []
    areas_medias = []

    # get data from df for reductions
    data = df_dict[key].values.astype(float)

    # copy of full df for a reducted df creation 
    cols = df_dict[key].columns
    df_dict_reducted = df_dict.copy()

    # calculate props and all stufs
    for n in reductions:
        print("Reduction:", n)
        areas = []
        if n==len(df_dict[key]):
            reps = 1   
        else:
            reps = repetitions
        
        for i in range(reps):
             data = df_dict[key].values.astype(float)
             data = data[np.random.choice(len(data), n, replace=False)]
             df_dict_reducted[key] = pd.DataFrame(data, columns=cols)
            
            
             #clear_output(wait=True)
             P = props_from_all_combinations(df_dict_reducted, S_inv)
             P =  cm_feasebles(P) 
             #print(P)
             #name = props_from_all_combinations(df_dict_reducted, S_inv)
             #output_file_list.append(name)
    
             Pcr = confidence_region(P.T, p = 95)
             #points = Pcr.T
             #print(points)
             hull = ConvexHull(Pcr[:,0:2])
             areas.append(hull.volume)

        

        CVs.append(cv(areas))
        areas_medias.append(np.mean(areas))
        
        df_out_data.append([n,cv(areas),np.mean(areas),np.std(areas),len(df_dict_reducted), len(P),np.mean(P[0]),np.mean(P[1]),np.mean(P[2])])
        display(df_out_data)
        
    print('--------------------------------------------------------------')
    print ("Áreas médias:", np.round(areas_medias,3))
    print ("Desvios padrão:", np.round(CVs,3))
    return (reductions,CVs)





def get_props_from_all_subsamples_combinations(bd, n_list, filename=None):
    P1=[]; P2=[]; P3=[]
    Y = bd.df_dict['Y'].values.astype(float)
    S_inv = np.linalg.inv(np.cov(Y.T))

    ss_dict = random_subsamples(bd, n_list)
    X = []
    ns = []

    keys = list(ss_dict.keys())
    for i in range (len(ss_dict['Y'])):
        y = Y[i]
        for j in range(len(ss_dict[keys[0]])): 
            S0 = ss_dict[keys[0]][j]
            for k in range(len(ss_dict[keys[1]])):
                S1 = ss_dict[keys[1]][k]
                for w in range(len(ss_dict[keys[2]])):
                    S2 = ss_dict[keys[2]][w]
                    X = np.array([S0,S1,S2]).T
                    P = solve_gls_4x4(y,X,S_inv)
                    
                    # Inclui apenas valores em que P1,P2>0 e P1+P2<1
                    #print(P[0:3])
                    if np.all(P[0:3]>=0):
                         P1.append(P[0])
                         P2.append(P[1])
                         P3.append(P[2])
                         #print (np.sum(P[0:3]))
    #print("Quantidade de soluções viáveis:", len(P1))
    return np.array([P1, P2, P3])


def confidence_region(P, p = 95):
    #from distances import mahalanobis_dist
    #P = P[0:2]
    Pm = np.mean(P, axis=0)
    dist = mahalanobis0_dist(P, Pm)
    #dist = mahalanobis_dist(P, Pm)
    
    sorted_idx = np.argsort(dist)
    Psorted = P[sorted_idx]

    # em ordem crescente
    end_idx = int((p/100)*len(Psorted))
    #print ("Os 95% mais próximos:", Psorted[:,:end_idx])
    return (Psorted[:end_idx,:])


def draw_hull(P):
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    fig = plt.figure(figsize=(4, 4))  
    
    Pm = np.mean(P, axis=0)
    points = P[:,0:2]   
    hull = ConvexHull(points)

    #fig = plt.figure(figsize=(6, 4))
    
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-',lw=1)

    plt.plot(Pm[0],Pm[1], "ro")
    plt.scatter(P[:,0],P[:,1],  marker=".", color='blue',s=1)
    
    plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'k-')
    plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'k')
    plt.title('95% confidence region'+' for '+str(P.shape[0])+' feasible solutions')
    plt.xlabel('P1')
    plt.ylabel('P2')
    #plt.xlim((-0.1,1.0))
    #plt.ylim((-0.1,1.0))
    plt.savefig('cloud_'+'GLS-Clarke'+str(P.shape[1])+'.eps')
    plt.show()
    return hull

































