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
from scipy.spatial import ConvexHull, convex_hull_plot_2d

"""
A Python implementation of the “fingerprinting approach” based on the Clarke and Minella (2016) method.  
"""
class PropSourceR3D_Target_CO:
    
    '''Inicialize'''
    def __init__(self, filename):
        self.df = pd.read_excel(filename, sheet_name=None)
        self.CB = np.array(self.df['g_Source(CB)'].values)
        self.UR = np.array(self.df['g_Source(UR)'].values)
        self.CF = np.array(self.df['g_Source(CF)'].values)
        self.NG = np.array(self.df['g_Source(NG)'].values) #native grassland
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
        
    def solve_gls_5x5(self,y,d,e,l,g,S_inv):
        #Xy = np.array([d/y,e/y,l/y]).T
        X = np.array([d,e,l,g]).T
        #S_inv = np.linalg.inv(np.cov(X.T))
  
        B = S_inv.dot(X)
        C = S_inv.dot(y)
        AtA = np.dot(X.T,B)
        yty = np.dot(X.T,C)
        AtA = np.vstack([AtA, np.ones(4)])
        AtA = np.vstack([AtA.T, np.ones(5)]).T
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
        if P.shape[0]>3:
            P=P[0:3,:]
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
                        xlabel = "CB", ylabel="UR", zlabel="CF"): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        from scipy.spatial import ConvexHull, convex_hull_plot_2d
        points = P.T   
        hull = ConvexHull(points)
        
        fig = plt.figure(figsize = (20,10),facecolor="w") 
        ax = plt.axes(projection="3d") 
        #scatter_plot = ax.scatter3D(x,y,z,c =cluster_colors,marker ='o')
        #for simplex in hull.simplices:
                #ax.plot3D(X[simplex, 0], X[simplex, 1],X[simplex, 2], 's-')
                #print ("volume:", hull.volume) 

        #fig = plt.figure(figsize=(6, 4))
    
        for simplex in hull.simplices:
            ax.plot3D(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')

        #plt.plot(Pm[0],Pm[1], "ro")
        ax.plot3D(P[0],P[1],P[2], "," )
    
        ax.plot3D(points[hull.vertices,0], points[hull.vertices,1], points[hull.vertices,2], 'k-')
        ax.plot3D(points[hull.vertices[0],0], points[hull.vertices[0],1], points[hull.vertices[0],2], 'k')
        plt.title(title)
        plt.xlabel('CB')
        plt.ylabel('UR')
        plt.zlabel('CF')
        plt.xlim([-0.1, 0.9])
        plt.ylim([-0.1, 0.9])
        plt.zlim([-0.1, 0.9])
        if savefig == True:
            plt.savefig(title+'.png')
        plt.show()
        return hull
    
    
    def plot3D(self, P, ss, j, n, nun, marker=',', mean = False, convex_hull = False, 
                     title = "Scatter plot: CB and UR and CF", savefig= True,
                     xlabel = "CB", ylabel= "UR", zlabel= "CF"): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        from scipy.spatial import ConvexHull, convex_hull_plot_2d
        
        ssp=int(ss*100)
       
        #fig = plt.figure(figsize = (8,6),facecolor="w") 
        #ax = plt.axes(projection="3d")
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')
        #ax = plt.axes(projection="3d") 
     
        #ax.plot3D(points[hull.vertices,0], points[hull.vertices,1], points[hull.vertices,2], 'k-')
        #ax.plot3D(points[hull.vertices[0],0], points[hull.vertices[0],1], points[hull.vertices[0],2], 'k')
       # plt.annotate(f'Area={hull.volume:.2f}',hull.volume, xytext=(0.5*offset, 1.2*offset), textcoords ='offset points')
        plt.grid() 
        
        #ax.plot3D(P[0],P[1],P[2], ".",color="grey",markersize=5) #"k,", ms = 5
        ax.plot3D(P[0],P[1],P[2], ".",color="grey",markersize=5)
        if convex_hull ==True:
            points = P.T   
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot3D(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-', lw=1)
                #plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r-', lw=1)
                #plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
                #plt.xlim([-0.1, 0.9])
                #plt.ylim([-0.1, 0.9])
                #plt.axis('equal')

            #print ("Area:", hull.volume) # para convex hull 2d é a área
            areaH=round(hull.volume,2)
          
        #ax = plt.gca() #you first need to get the axis handle
        #ax.set_aspect(1.0) #sets the height to width ratio to 1.5. 
        #ax.scatter3D(P[0],P[1],P[2],marker ='o')
            
            ax.set_xlim(-0.1, 1)
            ax.set_ylim(-0.1, 1)
            ax.set_zlim(-0.1, 1)
       
                       
        #plt.title('Target '+str(nun)+' '+' '+str(ssp)+'% samples', fontsize = "medium")
        #plt.title('Target '+str(nun)+'', fontsize = "medium")
        #plt.suptitle(title+'_times'+str(j+1)+'_red'+str(ss)+ '_run'+str(n)+ '_feasible'+str(P.shape[1])+'')

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
         
            ax.set_box_aspect(aspect=None, zoom=0.8)
            
        if mean==True: 
            Pm = np.mean(P, axis=1)
            #plt.plot(Pm[0],Pm[1], "rX")
            xm = Pm[0]
            ym = Pm[1]
            zm = Pm[2]
            ax.text(0.5, 0.5, 0.8, "Valores Médios", color='b')
            ax.text(0.5, 0.5, 0.7, "$\overline{P1}$=%0.2f"%(xm) , color='b')
            ax.text(0.5, 0.5, 0.6, "$\overline{P2}$=%0.2f"%(ym) , color='b')
            ax.text(0.5, 0.5, 0.5, "$\overline{P3}$=%0.2f"%(zm) , color='b')
          
        if savefig ==True:
            plt.savefig(title+'_CO_Target='+str(nun)+'_run'+str(n)+'_times'+str(j+1)+'_red'+str(ss)+'_feasible'+str(P.shape[1])+'.png', \
                        dpi=300,bbox_inches='tight')
            plt.savefig(title+'_CO_Target='+str(nun)+'_run'+str(n)+'_times'+str(j+1)+'_red'+str(ss)+'_feasible'+str(P.shape[1])+'.pdf'\
                             ,bbox_inches='tight')
            
        plt.tight_layout()
        plt.show()
        plt.close()
  ###plot boxplot. In this case is plotted
    
    def plotBoxRun(self, dataT, n, nun,savefig= True): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        #print(data)
        sns.set_style("whitegrid")
        seq=[1,int(n/2),n]  #can change  
        #dataT['redu']=pd.Categorical(dataT['redu'],categories=['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%'])#'30%',
        dataT['redu']=pd.Categorical(dataT['redu'],categories=['100%','75%','50%','25%','15%'])
        #for nun in range(1, 27): #nun=1...27, where the target number is equal to 26
        #dataT = data    #[data["Target"]==nun]
        print('Target ', nun )
   # boxplot print('Target ', nun, 'runn=',n )choice a specific runn=1...n
        for runn in seq: 
            if runn==1:
                datar = dataT[dataT["times"]==runn]
                dataTr=datar
            else:
                datar = dataT[dataT["times"]==runn]
                data1 = dataT[dataT["redu"]=="100%"] #30% 100
                dataTr=pd.concat([datar, data1])
            print('runn=',runn )
    # create figure
                
            fig, ax = plt.subplots(1, 1, figsize=(6, 4),sharey=True, sharex=True)  # 4 rows of 3 plots each to make 12 plots
                #plt.xlim(-0.1, 1)
                #plt.ylim(-0.1, 1)
               #fig.suptitle("Target"+ ' '+str(nun)+' '"to"+' ' "Run"+ ' '+str(runn)+' ') 
            sns.boxplot(y=dataTr["value"],x=dataTr["source"],hue=dataTr["redu"],palette='Paired',notch=True)
                #ax.set_title("Reductions"+ ' '+str(ss)+' ')
            #axs.get_legend().remove()
                #plt.grid()
                #plt.title('Target '+''+str(nun)+'', fontsize = "medium")
            ax.set_xlabel('')
            ax.set_ylabel('Relative source contribution') 
            ax.legend(bbox_to_anchor =[1.0, 1.0], prop = {"size": 8},loc='best', title = "% samples")
               # ax.yticks(np.arange(0, 1, step=0.2))
               # plt.set_xlim(-0.05, 1.05)
                #plt.set_ylim(-0.05, 1.05)
            plt.tight_layout()
                
            if savefig == True:
                    plt.savefig('Ps_Boxplot3D_CO_run_times='+str(runn)+'_Target='+str(nun)+'.png', dpi = 300,bbox_inches='tight')
                    plt.savefig('Ps_Boxplot3D_CO_run_times='+str(runn)+'_Target='+str(nun)+'.pdf',bbox_inches='tight')

            plt.show()
        plt.close()
                
    def plotBoxAll(self, dataT, n,nun,savefig= True): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        sns.set_style("whitegrid")
        #dataT['redu']=pd.Categorical(dataT['redu'],categories=['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%'])#
        dataT['redu']=pd.Categorical(dataT['redu'],categories=['100%','75%','50%','25%','15%'])
        ###for nun in range(1, 27): #nun=1...27, where the target number is equal to 26
        print('Target ', nun )
        #dataT = data   ###[data["Target"]==nun]
            #dataT['redu']=pd.Categorical(dataT['redu'],categories=['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%'])
    # create figure
            
        fig, ax = plt.subplots(1, 1, figsize=(6, 4),sharey=True, sharex=True)  # 4 rows of 3 plots each to make 12 plots
            #fig.suptitle("Target"+ ' '+str(nun)+' '"to"+' '+str(n)+' runs') 
            #plt.title('Target '+''+str(nun)+'', fontsize = "medium")
        sns.boxplot(y=dataT["value"],x=dataT["source"],hue=dataT["redu"],palette='Paired',notch=True)
            #ax.set_title("Reductions"+ ' '+str(ss)+' ')
            #axs.get_legend().remove()
        ax.set_ylabel( 'Relative source contribution')  
        ax.set_xlabel('')
        ax.legend(bbox_to_anchor =[1.0, 1.0], prop = {"size": 8},loc='best', title = "% samples")
        plt.tight_layout()
            
        if savefig == True:
                plt.savefig('Ps_Boxplot3D_CO_runAll='+str(n)+'_Target='+str(nun)+'.png', dpi = 300,bbox_inches='tight')
                plt.savefig('Ps_Boxplot3D_CO_runAll='+str(n)+'_Target='+str(nun)+'.pdf',bbox_inches='tight')

        plt.show()
        plt.close()         
    '''
  Subset random: It randomly chooses, without repetition, the samples from each subset.
    '''
    def randon_choice(self, nCB, nUR, nCF,nNG):
        CBs = self.CB[np.random.choice(len(self.CB), nCB, replace=False)] 
        URs = self.UR[np.random.choice(len(self.UR), nUR, replace=False)] 
        CFs = self.CF[np.random.choice(len(self.CF), nCF, replace=False)] 
        NGs = self.NG[np.random.choice(len(self.NG), nNG, replace=False)]
        return CBs,URs,CFs,NGs

    '''
    Run
    '''
    def run(self, Ys,Ds,Es,Ls,Gs, solve=0):
        
        P1 = []; P2 = []; P3 = []; P4 = []
        nY = len(Ys); nD = len(Ds); nE = len(Es); nL = len(Ls); nG = len(Gs)
        S_inv = self.Si
        #for i in range (nY):
        #print("Ys do run",Ys)
        #Psol=0
        y = Ys
        for j in range (nD): 
            d = Ds[j]
            for k in range (nE):
                e = Es[k]
                for w in range (nL):
                    l = Ls[w]
                    for r in range (nG):
                        g = Gs[r]
                        if solve==0:
                            P = self.solve_ols_2x2(y,d,e,l, normalize=True)
                        elif solve==1:
                            #P = self.solve_minimize2(y,d,e,l)
                            P = self.solve_ols_4x4(y,d,e,l, normalize=True)
                        elif solve==2:
                            P = self.solve_gls_5x5(y,d,e,l,g,S_inv)[0]
                        #print(P[0], P[1], P[2])
                        elif solve==3:
                            P = self.solve_minimize2(y,d,e,l)           
                        # Inclui apenas valores em que P1,P2>0 e P1+P2<1
                        if P[0]>0 and P[1]>0 and P[2]>0  and P[3]>0:
                            if P[0]<1 and P[1]<1 and P[2]<1 and P[3]<1:
                                P1.append(P[0])
                                P2.append(P[1])
                                P3.append(P[2])
                                P4.append(P[3])
                            #Psol=1
                    
        #print("Psol=",Psol)
                    
        return np.array([P1, P2, P3, P4])   #,Psol
            
 
           
    def reportCV(self, infos):
        
        print("Target \ts% nCB nUR nCF nNG \tMeanA \tStd \tCV_A \tNC \tF_Mean F_Std")
        print('_________________________________________________________________________________________________________________________')
        for row in infos:
            print("%i \t%s %i %i %i \t%.3f \t%.3f \t%.3f \t%5d \t%5d %5d"  \
              % (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10]))
        print('__________________________________________________________________________________________________________________________')
        
        #print ("Mean of the areas:", np.round(areas_mean,3))
        #print ("Coefficient of variation:", np.round(coefs_var,3))
        
    def reportP(self, infos):
        
        print("Target \ts% nCB nUR nCF nNG\tMeanP1 \tStdP1 \tMeanP2 \tStdP2 \tMeanP3 \
               \tStdP3 \tMeanP4 \tStdP4")
        print('_________________________________________________________________________________________________________________________')
        for row in infos:
            print("%i \t%s %i %i %i %i \t%.3f \t%.3f \t%.3f \t%.3f \t%.3f \t%.3f \t%.3f \t%.3f"  \
              % (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],\
                 row[12],row[13]))
        print('__________________________________________________________________________________________________________________________')
        
        #print ("Mean of the areas:", np.round(areas_mean,3))
        #print ("Coefficient of variation:", np.round(coefs_var,3))
        
    def multi_runs(self, n, nY, nCB, nUR, nCF, nNG, redus):
        from scipy.spatial import ConvexHull
        cv = lambda x: np.std(x) / np.mean(x) *100
        
        print('nCB=',nCB,'nUR=',nUR,'nCF=',nCF,'nNG=',nNG)
        #propss=['10%','20%'] #test
        #propss=['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'] #,'40%','50%','60%','70%','80%','90%','100%']
        propss=['15%','25%','50%','75%','100%']
        nSo=[nCB,nUR,nCF,nNG]
        plotA=[redus[0],redus[1],redus[2]] # #choice some reduction to plot confidence region. Ex: redus[0]=10%
        plott=[0,int(n/2)-1,n-1]#   #choice some run to plot confidence region. EX: 0=first run and n-1=last run 
        tar=[14,15,20,21] #choice some target
        #print(propss[0])
        
        
    ##repeat the simulation n times varying nCB, nUR and nCF for each target
        CVs_2= []
        areas_mean_2 = [] 
        infos_2CV = []
        infos_2P = []
        ynumber=[]
        raw_CV_data = []
        raw_Summ_data = []
        #raw_area_data = []
        
        ss_aux1=[];ss_aux2=[];ss_aux3=[];ss_aux4=[];propss_aux=[]
        
        Ys=np.array(self.df['Sediment(Y)'].values)
        
        
        for ii in range (nY):
            
            raw_data = []
            
            print('____')
            ysamp=ii+1
            y = Ys[ii]
            #print('y:', y)
            #ynumber.append(ii+1)
            
            if isinstance(redus, list):
                samples_sizes = redus.copy()
                idx = 1
            ssp=0
           
            
            for ss in samples_sizes:
                
                propss_aux.append(propss[ssp])
                #print("propos=",propss[ssp])
                
                areas_2 = []
                leng_2= [];P_1r = []; P_2r = []; P_3r = []; P_4r = []
                
                if ss==samples_sizes[-1]:
                    times=1
                else:
                    times=n
    
                #print('N Sample size:', ss)
                #print('times',times)
                print('Target=', ysamp, 'Sample size=',ss)
                m1=nSo[0]*ss
                m2=nSo[1]*ss
                m3=nSo[2]*ss
                m4=nSo[3]*ss
                
                nCBs=int(m1)
                nURs=int(m2)
                nCFs=int(m3)
                nNGs=int(m4)
                
                print('nCBs=',nCBs)
                print('nURs=',nURs)
                print('nCFs=',nCFs)
                print('nNGs=',nNGs)
                
                ss_aux1.append(nCBs) #samples_sizes_aux
                ss_aux2.append(nURs) #samples_sizes_aux
                ss_aux3.append(nCFs) #samples_sizes_aux
                ss_aux4.append(nNGs) #samples_sizes_aux
                
                for i in range(times):
                    CBs,URs,CFs,NGs = self.randon_choice(nCBs, nURs, nCFs,nNGs) 
                    nSamples = nCBs*nURs*nCFs*nNGs
                    #print('nSamples=',nSamples)
                   
                    #print('times=',i+1)
                    
                    # Solve = 2 
                    P_2 = self.run(y,CBs,URs,CFs,NGs, solve=2)
                    
                    lenPs2=len(P_2.T)
                    #print("lenPs2",lenPs2)
                    
                    if lenPs2>4:
                        P_2 = self.confidence_region(P_2, p = 95)
                        points_2 = P_2.T 
                        
                        lenPs=len(points_2)
                        #print("lenPs_CR",lenPs)
                        leng_2.append(lenPs)
                    
                        hull_2 = ConvexHull(points_2) #areas region confiance
                        areaHull=hull_2.volume
                        #print("areahull",areaHull)
                        areas_2.append(hull_2.volume)
                        
                             ## plots area
                        if ss in plotA: #choice reduction
                            if i in plott: #choice run some times 
                                if ysamp in tar:
                                    self.plot3D(P_2, ss, i,n , ysamp, mean=True, convex_hull = True, title = "Ps_Confidence_region3D_95%")
                   
                    ## salve in a list all Ps for each i
                        P_1r.extend(P_2[0])
                        P_2r.extend(P_2[1])
                        P_3r.extend(P_2[2])
                        P_4r.extend(1-P_2[0]-P_2[1]-P_2[2])
                    
                    ## salve in a dataframe all Ps for each i P1=CB, P2=UR and P3=CF
                        p_df = pd.DataFrame({
                            'P1': P_2[0],
                            'P2': P_2[1],
                            'P3': P_2[2],
                            'P4': 1-P_2[0]-P_2[1]-P_2[2],
                        }).stack().reset_index().drop("level_0", axis=1)
                        p_df.columns = ["source", "value"]
                        p_df["Target"] = ysamp 
                        p_df["nCB"] = nCBs
                        p_df["nUR"] = nURs
                        p_df["nCF"] = nCFs
                        p_df["nNG"] = nNGs
                        p_df["redu"]=propss[ssp]
                        p_df["times"] = i+1  
                        raw_data.append(p_df)
                    #print(raw_data) 
                    
                     ## salve in a dataframe all area for each i
                        #area_df = pd.DataFrame({
                           # "Area_2": areaHull,
                           # "Feasible": lenPs,
                        #},index=[0])
                        #area_df.columns = ["Area_2", "Feasible"]
                        #area_df["Target"] = ysamp 
                        #area_df["nCB"] = nCBs
                        #area_df["nUR"] = nURs
                        #area_df["nCF"] = nCFs
                       # area_df["redu"]=propss[ssp]
                        #area_df["times"] = i+1
                        #raw_area_data.append(area_df)   ## salve in a dataframe each Area for each i (pasta _ old)
                    else:
                        print("no solutions") 
                       
                     
                CVs_2.append(cv(areas_2))
                areas_mean_2.append(np.mean(areas_2))
                #print(np.mean(areas_2),np.std(areas_2), cv(areas_2))
                
                infos_2CV.append([ysamp,propss[ssp],nCBs,nURs,nCFs,np.mean(areas_2),np.std(areas_2), cv(areas_2),nSamples, \
                          (np.mean(leng_2)) ,(np.std(leng_2))])
                
                infos_2P.append([ysamp,propss[ssp],nCBs,nURs,nCFs, np.mean(P_1r), np.std(P_1r), \
                          np.mean(P_2r), np.std(P_2r), np.mean(P_3r), np.std(P_3r),np.mean(P_4r), np.std(P_4r)])
              
                       #if ss in redus: 
                ##Save the data to dataframe for boxplot
                raw_def=pd.concat(raw_data)
               
                ##only small number samples
                ## Save the data to file for boxplot 
                #if idx == 1: 
                    #pd.concat(raw_data).to_csv(f"Simulation_RedAll_CO_Ps_Sources_Run{n}_Target{ysamp}.csv",index=False)
                    #pd.concat(raw_data).to_csv(f"Simulation_CO_RC_Sources_Run{n}_CB_Target{ysamp}.csv",index=False)
                 # update of reduction        
                ssp=ssp+1           
                  # update of target      
                ynumber.append(ii+1) #salve each target number
                    
                    
            print("Next step: BoxPlot EACH run")
               
            self.plotBoxRun(raw_def, n, ysamp,savefig= True)
            
                
            print("Next step: BoxPlot with ALL run")
                
            self.plotBoxAll(raw_def, n, ysamp,savefig= True)
            
            raw_def= []
       
        CV_df = pd.DataFrame({
                   "nCB": ss_aux1,
                   "nUR": ss_aux2,
                   "nCF": ss_aux3,
                   "nNG": ss_aux4,
                   "redu":propss_aux,
                   "areas_mean_2": areas_mean_2,
                   "CV_2": CVs_2,
                   "Target": ynumber
                  })
        CV_df.columns = ["nCB","nUR","nCF","nNG", "propss","areas_mean_2","CV_2","Target"]
        raw_CV_data.append(CV_df)
        raw_CV_df = pd.concat(raw_CV_data)
        
        Summ_df = pd.DataFrame(infos_2P)
        Summ_df.columns = ["Target","s%","nCB","nUR","nCF","MeanP1","StdP1","MeanP2","StdP2","MeanP3","StdP3","MeanP4","StdP4"]
        raw_Summ_data.append(Summ_df)
        raw_Summ_df = pd.concat(raw_Summ_data)
        
        if idx == 1:
            raw_CV_df.to_csv(f"Resume3D_CV2_Sources_Run{n}_CO.csv", index=False) 
            raw_Summ_df.to_csv(f"Resume3D_Solve2_P_Sources_Run{n}_CO.csv", index=False) 
     
     
        print('------------------------------------------------------------------------------------------------------------------------')
        #print('Solve_0')
        #self.report(infos_0, CVs_0, areas_mean_0)
        #print('--------------------------------------------------------------')
        #print('Solve_1')
        #self.report(infos_1, CVs_1, areas_mean_1)
        #print('--------------------------------------------------------------')
        print('Solve_2')
        #self.report(infos_2, CVs_2, areas_mean_2)
        print('NC=combination number,   F_Mean=feasible mean,  F_Std=feasible standard deviation')
        self.report_2CV(infos)
        print('---------------------------------------------------------------------------------------------------------')
  
        self.report_2P(infos)
        print('---------------------------------------------------------------------------------------------------------')
  
        print("Next step: Figure")
        #return (CVs_0,CVs_1, CVs_2)
        return (CVs_2)
        
       
            
            
            
