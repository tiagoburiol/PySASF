#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:09:48 2021

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from scipy.linalg import solve
import seaborn as sns


"""
A Python implementation of the “fingerprinting approach” based on the Clarke and Minella (2016) method.  
"""
class PropSource_Target_GP:
    
    '''Inicialize'''
    def __init__(self, filename):
        self.df = pd.read_excel(filename, sheet_name=None)
        self.CB = np.array(self.df['g_Source(CB)'].values)
        self.UR = np.array(self.df['g_Source(UR)'].values)
        self.CF = np.array(self.df['g_Source(CF)'].values)
        self.Y = np.array(self.df['Sediment(Y)'].values)
        self.Si = np.linalg.inv(np.cov(self.Y.T))
        return None
        
    
    def infos(self):
        for key in self.df:
            print ("Sheet name:", key)
            print ("Columns:", self.df[key].columns.tolist())
            print ("Number of samples:", self.df[key].shape[0])
            print("--")
            
    def nsample(self,ns):
        ns=[]
        for key in self.df:
            nsa =self.df[key].shape[0]
            ns.append(nsa)
        return np.array(ns)            
    
    def means(self):
        for key in list(self.df):
            print (key,':')
            print ( self.df[key].mean(axis=0))
            print ('--')
            
    def std(self):
        for key in list(self.df):
            print (key,':')
            print ( self.df[key].std(axis=0))
            print ('--')
            
    def hist(self):
        for key in list(self.df):
            self.df[key].hist(bins=10)
            
    def _buildSourceContribMatrix(self, a):
        g = len(self.df.keys())-1 # número de fontes
        A = np.ones((g+1,g+1))
        A[:g,:g] = np.dot(a,a.T)
        A[g, g]=0
        return A     
        
        
    '''
    ----------------------------------------------------------------
    SOLVERS OPTIONS: To solve the overdetermined linear system of equations by least squares method
    1) OLS - ordinary least squares = solve_ols_4x4 (solve_ols_2x2 or solve_minimize2 or solve_ols_4x4_mod)
    2) GLS - generalized least squares = solve_gls_4x4
                    
    ----------------------------------------------------------------
    '''

    def solve_ols_4x4(self,y,d,e,l, normalize=True):
        if normalize == True:
             d=d/y; e=e/y; l=l/y; y=y/y
        X = np.array([d/y,e/y,l/y])
        A = np.vstack([X.T, np.ones(3)])
    
        AtA = np.dot(A.T,A)
        AtA = np.vstack([AtA, np.ones(3)])
        AtA = np.vstack([AtA.T, np.ones(4)]).T
        AtA[-1,-1] = 0
    
        y = np.append(y,[1])
        y = y[:, np.newaxis]
        
        Aty = np.dot(A.T,y)
        Aty = np.append(Aty,[1])

        P = np.dot(np.linalg.inv(AtA),Aty)
        return (P)
        
    def solve_gls_4x4(self,y,d,e,l,S_inv):
        #Xy = np.array([d/y,e/y,l/y]).T
        X = np.array([d,e,l]).T
        #S_inv = np.linalg.inv(np.cov(X.T))
  
        B = S_inv.dot(X)
        C = S_inv.dot(y)
        AtA = np.dot(X.T,B)
        yty = np.dot(X.T,C)
        AtA = np.vstack([AtA, np.ones(3)])
        AtA = np.vstack([AtA.T, np.ones(4)]).T
        AtA[-1,-1] = 0
        Aty = np.append(yty,[1])
        Ps = np.dot(np.linalg.inv(AtA),Aty)
        return (Ps, X, y, AtA, Aty)
    '''
    ----------------------------------------------------------------
    SOLVERS OLS Extras: 
    '''
    
    def solve_ols_2x2(self,y,d,e,l, normalize = True):
        if normalize == True:
            d=d/y; e=e/y; l=l/y; y=y/y
        A =  np.array([(d-l).T , (e-l).T]).T
        y=y-l
        #P = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y-l)
        P = np.linalg.lstsq(A, y, rcond=None)[0]
        P = np.append(P,1-P[0]-P[1])
        return P 
        
    def solve_minimize2(self,y,d,e,l):
        from scipy.optimize import minimize, Bounds, LinearConstraint
        A =  np.array([d.T ,e.T, l.T]).T
        P0 = np.array([0.3, 0.3, 0.3])
        #X0 = -np.log((1./P0)-1.)
        def f(P):   
            return sum(((y-np.dot(A,P))/y)**2)
        lc = LinearConstraint([1,1,1], [1], [1])
        bnds = Bounds([0,0,0],[1,1,1])
        S = minimize(f, P0, 
                     method='trust-constr', 
                     #jac=rosen_der, 
                     #hess='cs',
                     constraints=lc,
                     options={'verbose': 0},
                     bounds=bnds)
        #P = 1.0/(1.0+np.exp(-S.x))
        P = S.x
        return(P)
    
    def solve_ols_4x4_mod(self,y,d,e,l, normalize=True):
        if normalize == True:
             d=d/y; e=e/y; l=l/y; y=y/y
        a = np.array([d/y,e/y,l/y])
        A = self._buildSourceContribMatrix(a)
        Z = np.append(np.dot(a, y/y), 1)
        P = np.dot(np.linalg.inv(A),Z)
        #P = solve(A, Z)
        return (P)
        
    '''
    ----------------------------------------------------------------------
    DISTANCE OPTIONS: Use scipy.spatial.distance implemantation for 
    compute distance between two numeric vectors. 
    - braycurtis: Compute the Bray-Curtis distance between two 1-D arrays.
    - euclidian: Computes the Euclidean distance between two 1-D arrays.
    - mahalanobis: Compute the Mahalanobis distance between two 1-D arrays. 
    -----------------------------------------------------------------------
    '''
    def mahalanobis_dist(self, P_arr, Pm):
        from scipy.spatial import distance
        S = np.cov(P_arr)
        dist = []
        for P in P_arr.T:
            d = distance.mahalanobis(P, Pm, np.linalg.inv(S))       
            dist.append(d)
        return np.array(dist)
        
    def braycurtis_dist(self,P_arr):
        from scipy.spatial import distance
        Pm = np.mean(P_arr, axis=1)
        #S = np.cov(P_arr)
        dist = []
        for P in P_arr.T:
            #S = np.cov(P,Pm)
            d = distance.braycurtis(P, Pm)
            dist.append(d)
        return np.array(dist)
        
    def euclidean_dist(self,P_arr):
        from scipy.spatial import distance
        Pm = np.mean(P_arr, axis=1)
        
        dist = []
        for P in P_arr.T:
            d = distance.euclidean(P, Pm)
            dist.append(d)
        return np.array(dist)
        
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
        from scipy.spatial import ConvexHull, convex_hull_plot_2d
        points = P.T   
        hull = ConvexHull(points)

        fig = plt.figure(figsize=(6, 4))
    
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
    
    
    def plot2D(self, P, ss, n, idx, marker=',', mean = False, convex_hull = False, 
                     title = "Scatter plot: P1 and P2", savefig= True,
                     xlabel = "P1", ylabel= "P2"): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        
        #fig = plt.figure(figsize=(6, 6))
        if mean==True: 
            Pm = np.mean(P, axis=1)
            plt.plot(Pm[0],Pm[1], "ko")
            
        if convex_hull ==True:
            from scipy.spatial import ConvexHull
            points = P.T   
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
                #plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r-', lw=1)
                #plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
                #plt.xlim([-0.1, 0.9])
                #plt.ylim([-0.1, 0.9])
                #plt.axis('equal')

            print ("Area:", hull.volume) # para convex hull 2d é a área
        ax = plt.gca() #you first need to get the axis handle
        ax.set_aspect(1.0) #sets the height to width ratio to 1.5. 
        plt.xlim(-0.1, 1)
        plt.ylim(-0.1, 1)
        plt.plot(P[0],P[1], "k,")
        plt.grid()                
        #plt.title(title+ '('+str(P.shape[1])+')')
        plt.title(title+ '('+str(ss)+')'+ '('+str(n)+')'+ '('+str(P.shape[1])+')')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.axis('equal')

        

        if savefig == True:
            #plt.savefig(title+ '.png')
            plt.savefig(title+' Subset='+str(idx)+' Samples='+str(ss)+'_and_runs='+str(n)+'.png')
        plt.show()

        
    '''
  Subset random: It randomly chooses, without repetition, the samples from each subset.
    '''
    def randon_choice(self, nCB, nUR, nCF):
        CBs = self.CB[np.random.choice(len(self.CB), nCB, replace=False)] 
        URs = self.UR[np.random.choice(len(self.UR), nUR, replace=False)] 
        CFs = self.CF[np.random.choice(len(self.CF), nCF, replace=False)] 
        return CBs,URs,CFs

    '''
    Run
    '''
    def run(self, Ys,Ds,Es,Ls, solve=0):
        P1 = []; P2 = []; P3 = []
        nY = len(Ys); nD = len(Ds); nE = len(Es); nL = len(Ls)
        S_inv = self.Si
        #for i in range (nY):
        #print("Ys do run",Ys)
        y = Ys
        for j in range (nD): 
            d = Ds[j]
            for k in range (nE):
                e = Es[k]
                for w in range (nL):
                    l = Ls[w]
                    if solve==0:
                        P = self.solve_ols_2x2(y,d,e,l, normalize=True)
                    elif solve==1:
                            #P = self.solve_minimize2(y,d,e,l)
                        P = self.solve_ols_4x4(y,d,e,l, normalize=True)
                    elif solve==2:
                        P = self.solve_gls_4x4(y,d,e,l,S_inv)[0]
                    elif solve==3:
                        P = self.solve_minimize2(y,d,e,l)
                        # Inclui apenas valores em que P1,P2>0 e P1+P2<1
                    if P[0]>0 and P[1]>0 and P[2]>0:
                        if P[0]<1 and P[1]<1 and P[2]<1:
                            P1.append(P[0])
                            P2.append(P[1])
                            P3.append(P[2])
                                        #print (P[0], P[1], P[2])
        #print("Quantidade de soluções viáveis:", len(P1))
        return np.array([P1, P2, P3])
            
 
           
    def report(self, infos):
        #print("nSamp \tMeanA \tStd \tSolution \tCR_MD_95 \tMeanP1 \tMeanP2 \tMeanP3")
        #print('____________________________________________________________________________')
        #for row in infos:
           # print("%i \t%.3f \t%.3f \t%7d \t%7d \t%.3f \t%.3f \t%.3f" \
             # % (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7]))
        #print('_____________________________________________________________________________')
        
        print("Target \tnSamp \tMeanA \tStd \tCV_A \tNComb \tCR_Mean\tStd_CR\tMeanP1 \tStdP1 \tMeanP2 \tStdP2 \tMeanP3 \tStdP3")
        print('_________________________________________________________________________________________________________________________')
        for row in infos:
            print("%i \t%i \t%.3f \t%.3f \t%.3f \t%5d \t%5d \t%5d \t%.3f \t%.3f \t%.3f \t%.3f \t%.3f \t%.3f" \
              % (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13]))
        print('__________________________________________________________________________________________________________________________')
        
        #print ("Mean of the areas:", np.round(areas_mean,3))
        #print ("Coefficient of variation:", np.round(coefs_var,3))
        

        
    def multi_runs(self, n, nY, nCB, nUR, nCF,plots2D=[]):
        from scipy.spatial import ConvexHull
        cv = lambda x: np.std(x) / np.mean(x) *100
        #propss=['10%','20%']
        propss=['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
    ##repeat the simulation n times varying nY, nCB, nUR and nCF
        CVs_2= []
        areas_mean_2 = [] 
        infos_2 = []
        ynumber=[];samples_sizes_aux=[];propss_aux=[]
        #raw_data = []
        raw_CV_data = []
        raw_Summ_data = []
        raw_area_data = []
        #if isinstance(nY, list):
            #samples_sizes = nY.copy()
            #self.Si = np.linalg.inv(np.cov(self.Y.T))
           # idx = 0
        Ys=np.array(self.df['Sediment(Y)'].values)
            
        for ii in range (nY):
         
            print('____')
            ysamp=ii+1
            y = Ys[ii]
            #print('y:', y)
            #ynumber.append(ii+1)
            if isinstance(nCB, list):
                samples_sizes = nCB.copy()
            #self.Si = np.linalg.inv(np.cov(self.D.T))
                idx = 1
            if isinstance(nUR, list):
                samples_sizes = nUR.copy()
            #self.Si = np.linalg.inv(np.cov(self.E.T))
                idx = 2
            if isinstance(nCF, list):
                samples_sizes = nCF.copy()
            #self.Si = np.linalg.inv(np.cov(self.L.T))
                idx = 3

            print('Set Samples sizes:',samples_sizes)
            
            #print(samples_sizes_aux)
            #print(len(samples_sizes_aux))
            raw_data = []
            #raw_area_data = []
            ssp=0
            for ss in samples_sizes:
                samples_sizes_aux.append(ss)
                propss_aux.append(propss[ssp])
                #print('samples_sizes_aux',samples_sizes_aux)
                #print(len(samples_sizes_aux))
                areas_0 = []; areas_1 = []; areas_2 = []  
                leng_2= [];P_1r = []; P_2r = []; P_3r = []
                
            # Choosing the source or sediment to reduce the number of samples
            #if idx == 0:
                #nY = ss
                if idx == 1:
                    nCB = ss    
                if idx == 2:
                    nUR = ss 
                if idx == 3:
                    nCF = ss
                    
                if ss==samples_sizes[-1]:
                    times=1
                else:
                    times=n
    
                #print('N Sample size:', ss)
                #print('times',times)
                print('Target=', ysamp, 'Sample size=',ss)
                
                for i in range(times):
                    CBs,URs,CFs = self.randon_choice(nCB, nUR, nCF) 
                    nSamples = nCB*nUR*nCF
                   
                    #print('times=',i+1)
                
                    # Solve = 2 
                    P_2 = self.run(y,CBs,URs,CFs, solve=2)
                    P_2 = self.confidence_region(P_2, p = 95)
                    points_2 = P_2.T 
                    lenPs=len(points_2)
                    #print("lenPs",lenPs)
                    leng_2.append(lenPs)
                    
                    hull_2 = ConvexHull(points_2) #areas region confiance
                    areaHull=hull_2.volume
                    #print("areahull",areaHull)
                    areas_2.append(hull_2.volume)
                    
                   
                    ## salve in a list all Ps for each i
                    P_1r.extend(P_2[0])
                    P_2r.extend(P_2[1])
                    P_3r.extend(1-P_2[0]-P_2[1])
                    
                    ## salve in a dataframe all Ps for each i P1=CB, P2=UR and P3=CF
                    p_df = pd.DataFrame({
                        'CB': P_2[0],
                        'UR': P_2[1],
                        'CF': 1-P_2[0]-P_2[1],
                    }).stack().reset_index().drop("level_0", axis=1)
                    p_df.columns = ["source", "value"]
                    p_df["Target"] = ysamp 
                    p_df["Samples"] = ss
                    p_df["redu"]=propss[ssp]
                    p_df["times"] = i+1  
                    raw_data.append(p_df)
                    #print(raw_data) 
                    
                     ## salve in a dataframe all area for each i
                    area_df = pd.DataFrame({
                        "Area_2": areaHull,
                        "Feasible": lenPs,
                    },index=[0])
                    area_df.columns = ["Area_2", "Feasible"]
                    area_df["Target"] = ysamp 
                    area_df["Samples"] = ss
                    area_df["redu"]=propss[ssp]
                    area_df["times"] = i+1
                    raw_area_data.append(area_df)
                    #print(raw_data)
                    
                    ## salve in a dataframe each Area for each i (pasta _ old)
                ssp=ssp+1  
                ynumber.append(ii+1) #salve each target number
               
                CVs_2.append(cv(areas_2))
                areas_mean_2.append(np.mean(areas_2))
                #print(np.mean(areas_2),np.std(areas_2), cv(areas_2))
                
                infos_2.append([ysamp, ss, np.mean(areas_2),np.std(areas_2), cv(areas_2),nSamples, \
                      int(np.mean(leng_2)) ,int(np.std(leng_2)), np.mean(P_1r), np.std(P_1r), \
                      np.mean(P_2r), np.std(P_2r), np.mean(P_3r), np.std(P_3r)])
   
        #choose which reduction to print figure ### in this script plot2D=nsample
                if ss in plots2D:
            # Save the data to a csv file
            #if idx == 0:
                    #raw_df.to_csv("Simulation_Run10_Y.csv", index=False)
                    #raw_CV_df.to_csv("CV2_Y.csv", index=False)
                    if idx == 1: 
                        #pd.concat(raw_data).to_csv("Simulation_Ps_Run10_CB.csv", index=False) 
                        pd.concat(raw_data).to_csv(f"Simulation_Ps_Run{n}_CB_Target{ysamp}.csv",index=False)
                        pd.concat(raw_area_data).to_csv(f"Simulation_Area_Run{n}_CB_AllTarget.csv", index=False) 
                    if idx == 2:
                        pd.concat(raw_data).to_csv(f"Simulation_Ps_Run{n}_UR_Target{ysamp}.csv", index=False)
                        pd.concat(raw_area_data).to_csv(f"Simulation_Area_Run{n}_UR_AllTarget.csv", index=False) 
                    if idx == 3:
                        pd.concat(raw_data).to_csv(f"Simulation_Ps_Run{n}_CF_Target{ysamp}.csv", index=False)
                        pd.concat(raw_area_data).to_csv(f"Simulation_Area_Run{n}_CF_AllTarget.csv", index=False) 
                    #del raw_data, raw_area_data
               
                
        CV_df = pd.DataFrame({
                   "nSample": samples_sizes_aux,
                   "redu":propss_aux,
                   "areas_mean_2": areas_mean_2,
                   "CV_2": CVs_2,
                   "Target": ynumber
                  })
        CV_df.columns = ["nSample", "propss","areas_mean_2","CV_2","Target"]
        raw_CV_data.append(CV_df)
        raw_CV_df = pd.concat(raw_CV_data)
        
       
        Summ_df = pd.DataFrame(infos_2)
        Summ_df.columns = ["Target","nSamp","MeanA","Std","CV_A","NComb","CR_Mean","Std_CR",\
                           "MeanP1","StdP1","MeanP2","StdP2","MeanP3","StdP3"]
        raw_Summ_data.append(Summ_df)
        raw_Summ_df = pd.concat(raw_Summ_data)
        
        #if idx == 0: 
                #raw_CV_df.to_csv("Resume_CV2_Run{n}_Y.csv", index=False) sep=' ', decimal='.', float_format='%.5f'
                #raw_Summ_df.to_csv("Resume_Solve2_Run10_Y.csv", index=False) 
        if idx == 1:
                raw_CV_df.to_csv(f"Resume_CV2_Run{n}_CB.csv", index=False) 
                raw_Summ_df.to_csv(f"Resume_Solve2_Run{n}_CB.csv", index=False) 
        if idx == 2:
                raw_CV_df.to_csv(f"Resume_CV2_Run{n}_UR.csv", index=False)
                raw_Summ_df.to_csv(f"Resume_Solve2_Run{n}_UR.csv", index=False) 
        if idx == 3:
                raw_CV_df.to_csv(f"Resume_CV2_Run{n}_CF.csv", index=False)
                raw_Summ_df.to_csv(f"Resume_Solve2_Run{n}_CF.csv", index=False) 
        
     
        print('------------------------------------------------------------------------------------------------------------------------')
        #print('Solve_0')
        #self.report(infos_0, CVs_0, areas_mean_0)
        #print('--------------------------------------------------------------')
        #print('Solve_1')
        #self.report(infos_1, CVs_1, areas_mean_1)
        #print('--------------------------------------------------------------')
        print('Solve_2')
        #self.report(infos_2, CVs_2, areas_mean_2)
        self.report(infos_2)
        print('---------------------------------------------------------------------------------------------------------')
  
        print("Next step: Figure")
        #return (CVs_0,CVs_1, CVs_2)
        return (CVs_2)
        
       
            
            
            
