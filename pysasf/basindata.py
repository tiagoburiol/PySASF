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

    def calculate_and_save_all_proportions(self, load=True):
        inicio = time.time()
        # name of output file
        name = ''
        for key in self.df_dict.keys():
            name = name+key+str(len(self.df_dict[key]))

        print('Calculating all proportions...')
        combs, Ps = cm.props_from_all_combinations(self, 
                                               solve_opt = self.cm_solver_option,
                                               save=True)
        #print(len(combs), Ps.shape)
        self.combs_filename = name+'_combs.txt'
        self.props_filename = name+'_props.txt'

        print('Total combinations:',len(combs),', shape of proportions:', Ps.shape)
        print('Saving combinations indexes in:',self.combs_filename)
        print('Saving proportions calculated in:',self.props_filename)
        np.savetxt(self.combs_filename, combs,fmt='%s')
        np.savetxt(self.props_filename, Ps, fmt='%1.4f')
        #np.save(name+'_combs.npy', combs)
        #np.save(name+'props.npy', Ps)
        
        fim = time.time()
        print ("Done! Time for processing and save:",fim-inicio)
        
        if load:
            c,p = self.load_combs_and_props_from_files(self.combs_filename,
                                                       self.props_filename)
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
    
    #plot convexhull. In this case is not plotted
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

            print ("Area:", hull.volume) #convex hull 2d is the area
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
        
        
    #plot boxplot. In this case is plotted
    
    def plotBoxRun(self, data, n, idx, nun, savefig= True): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        #print(data)
        sns.set_style("whitegrid")
        seq=[1,n]  #can change  ,int(n/2)
        
        for nun in range(1, 27): #nun=1...27, where the target number is equal to 26
            dataT = data[data["Target"]==nun]
            if idx==1:
                j='CB'
            #DataOrd = data.sort_values(by = 'redu', ascending = False) #Ordena a lista
                                                            #['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%'])
                dataT['redu']=pd.Categorical(dataT['redu'],categories=['20%','10%'])
            if idx==2:
                j='UR'                                      #['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%'])
                dataT['redu']=pd.Categorical(dataT['redu'],categories=['20%','10%'])
            if idx==3:
                j='CF'                                   ##['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%','5%'])
                dataT['redu']=pd.Categorical(dataT['redu'],categories=['20%','10%'])
                
           
    # boxplot choice a specific runn=1...n
            
            for runn in seq: 
                if runn==1:
                    datar = dataT[dataT["times"]==runn]
                    dataTr=datar
                else:
                    datar = dataT[dataT["times"]==runn]
                    data1 = dataT[dataT["redu"]=="20%"] #"100%"
                    dataTr=pd.concat([datar, data1])
        
    # create figure
                fig, ax = plt.subplots(1, 1, figsize=(5, 3),sharey=True, sharex=True)  # 4 rows of 3 plots each to make 12 plots
                fig.suptitle("Target"+ ' '+str(nun)+' '"to"+' ' "Run"+ ' '+str(runn)+' ') 
                sns.boxplot(y=dataTr["value"],x=dataTr["source"],hue=dataTr["redu"],palette='Paired',notch=True)
                ax.set_title("Reductions at"+ ' '+str(j)+' ')
            #axs.get_legend().remove()
                ax.set_xlabel('')
                ax.set_ylabel( 'Relative source contribution') 
                ax.legend(bbox_to_anchor =[1.0, 1.1], prop = {"size": 9},loc='best', title = "% samples")
                plt.tight_layout()
                
                if savefig == True:
                    plt.savefig('Boxplot_runn='+str(runn)+'_Target='+str(nun)+'_'+str(j)+'.png', dpi = 300,bbox_inches='tight')
                    plt.savefig('Boxplot_runn='+str(runn)+'_Target='+str(nun)+'_'+str(j)+'.pdf',bbox_inches='tight')

                plt.show()
                
                
    def plotBoxAll(self, data, n, idx, nun,savefig= True): # xlabel = "P1 (CB)", ylabel= "P2 (UR)"
        sns.set_style("whitegrid")
       
        for nun in range(1, 27): #nun=1...27, where the target number is equal to 26
            dataT = data[data["Target"]==nun]
            if idx==1:
                j='CB'                                                #['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%'])
                dataT['redu']=pd.Categorical(dataT['redu'],categories=['20%','10%'])
            if idx==2:
                j='UR'                                                #['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%'])
                dataT['redu']=pd.Categorical(dataT['redu'],categories=['20%','10%'])
            if idx==3:
                j='CF'                                                #['100%','90%','80%','70%','60%','50%','40%','30%','20%','10%','5%'])
                dataT['redu']=pd.Categorical(dataT['redu'],categories=['20%','10%'])
                
    # create figure
            fig, ax = plt.subplots(1, 1, figsize=(5, 3),sharey=True, sharex=True)  # 4 rows of 3 plots each to make 12 plots
            fig.suptitle("Target"+ ' '+str(nun)+' '"to"+' '+str(n)+' runs') 
            sns.boxplot(y=dataT["value"],x=dataT["source"],hue=dataT["redu"],palette='Paired',notch=True)
            ax.set_title("Reductions at"+ ' '+str(j)+' ')
            #axs.get_legend().remove()
            ax.set_ylabel( 'Relative source contribution')  
            ax.set_xlabel('')
            ax.legend(bbox_to_anchor =[1.0, 1.1], prop = {"size": 9},loc='best', title = "% samples")
            plt.tight_layout()
            
            if savefig == True:
                plt.savefig('Boxplot_runAll='+str(n)+'_Target='+str(nun)+'_'+str(j)+'.png', dpi = 300,bbox_inches='tight')
                plt.savefig('Boxplot_runAll='+str(n)+'_Target='+str(nun)+'_'+str(j)+'.pdf',bbox_inches='tight')

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
            
 

    '''
    Run
    Ys is a subsample of Y data
    '''    
    def run_and_save(self, Ys, Ds, Es, Ls):
        from IPython.display import clear_output
        counter1 = 0; counter2 = 0; percent1 = 0; percent2 = 0
        total = len(Ys)*len(Ds)*len(Es)*len(Ls)
        for i in range(len(Ys)):
            y = Ys[i]
            for j in range(len(Ds)): 
                d = Ds[j]
                for k in range(len(Es)):
                    e = Es[k]
                    for w in range(len(Ls)):
                        l = Ls[w]
                        counter1+=1
                        P = self.solve_gls_4x4(y,d,e,l,self.Si)[0][0:3]
                        #P.tofile(file_all, format='str')
                        if not (np.any(P<0) or np.any(P>1)):
                            # write in file
                            P.tofile(self.file, format='str')
                            #print some stats
                            counter2+=1
                            clear_output(wait=True)
                            percent1 =  counter1/total
                            percent2 =  counter2/counter1
                            print(f'Found {counter1}({percent1:.2%}) solutions with {counter2}({percent2:.2%}) feasibles.')
        return None


           
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
        

        
    def multi_runs(self, n, nY, nCB, nUR, nCF, plots2D=[]):
        from scipy.spatial import ConvexHull
        cv = lambda x: np.std(x) / np.mean(x) *100
        ##repeat the simulation n times varying nY, nCB, nUR and nCF
        CVs_2= []
        areas_mean_2 = [] 
        infos_2 = []
        ynumber=[];samples_sizes_aux=[];propss_aux=[]
        raw_CV_data = []
        raw_Summ_data = []
        raw_def= []
        raw_data = []
        raw_area_data = []
            
        Ys=np.array(self.df['Sediment(Y)'].values)
            
        for ii in range (nY):
            print('____')
            ysamp=ii+1
            y = Ys[ii]
            #print('y:', y)
            #ynumber.append(ii+1)
            if isinstance(nCB, list):
                samples_sizes = nCB.copy()
                #propss=['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
                propss=['10%','20%']
                idx = 1
            if isinstance(nUR, list):
                samples_sizes = nUR.copy()
                #propss=['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
                propss=['10%','20%']
                idx = 2
            if isinstance(nCF, list):
                samples_sizes = nCF.copy()
                #propss=['5%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
                propss=['10%','20%']
                idx = 3

            print('Set Samples sizes:',samples_sizes)
            
            #print(samples_sizes_aux)
            #print(len(samples_sizes_aux))
            
           # raw_data = []
           # raw_area_data = []
            
            ssp=0
            for ss in samples_sizes:
                samples_sizes_aux.append(ss)
                propss_aux.append(propss[ssp])
                #print('samples_sizes_aux',samples_sizes_aux)
                #print(len(samples_sizes_aux))
                areas_0 = []; areas_1 = []; areas_2 = []  
                leng_2= [];P_1r = []; P_2r = []; P_3r = []
                
            # Choosing the source or sediment to reduce the number of samples
                if idx == 1:
                    nCB = ss            
                if idx == 2:
                    nUR = ss 
                if idx == 3:
                    nCF = ss
                    
                if ss==samples_sizes[-1]: #valid for the total number of samples,ie, 100% for CB, UR and CF
                    times=1
                else:
                    times=n
    
                #print('N Sample size:', ss)
                #print('times',times)
                print('Target=', ysamp, 'Sample size=',ss)
                
                
                '''
                Here the solutions are calculed 
                '''
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
                    
                    ## salve in a dataframe each Area for each i (pasta _ old)
                ssp=ssp+1  
                ynumber.append(ii+1) #salve each target number 
               
                CVs_2.append(cv(areas_2))
                areas_mean_2.append(np.mean(areas_2))
                #print(np.mean(areas_2),np.std(areas_2), cv(areas_2))
                
                infos_2.append([ysamp, ss, np.mean(areas_2),np.std(areas_2), cv(areas_2),nSamples, \
                      int(np.mean(leng_2)) ,int(np.std(leng_2)), np.mean(P_1r), np.std(P_1r), \
                      np.mean(P_2r), np.std(P_2r), np.mean(P_3r), np.std(P_3r)])
   
 #       #choose which reduction to print figure ### in this script plot2D=nsample
 #               if ss in plots2D:
 #           # Save the data to file for boxplot
 #                   raw_def=pd.concat(raw_data)
 #               
 #                   #print(raw_def)
 #             
 #       print("Next step: BoxPlot EACH run")
 #              
 #       self.plotBoxRun(raw_def, n, idx,ysamp, savefig= True)
 #           
 #               
 #       print("Next step: BoxPlot with ALL run")
 #               
 #       self.plotBoxAll(raw_def, n, idx,ysamp, savefig= True)
                

        CV_df = pd.DataFrame({
                   "Target": ynumber,
                   "nSample": samples_sizes_aux,
                   "redu":propss_aux,
                   "areas_mean_2": areas_mean_2,
                   "CV_2": CVs_2,
                  })
        CV_df.columns = ["Target","nSample", "propss","areas_mean_2","CV_2"]
        raw_CV_data.append(CV_df)
        raw_CV_df = pd.concat(raw_CV_data)
        
       
        Summ_df = pd.DataFrame(infos_2)
        Summ_df.columns = ["Target","nSamp","MeanA","Std","CV_A","NComb","CR_Mean","Std_CR",\
                           "MeanP1","StdP1","MeanP2","StdP2","MeanP3","StdP3"]
        raw_Summ_data.append(Summ_df)
        raw_Summ_df = pd.concat(raw_Summ_data)
       
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
  
        #print("Next step: BoxPlot with ALL run")
               
        #self.plotBoxAll(raw_def_1, n, idx, savefig= True)
        
        
        
        #return (CVs_0,CVs_1, CVs_2)
        return (CVs_2)
        
       
            
            
            
