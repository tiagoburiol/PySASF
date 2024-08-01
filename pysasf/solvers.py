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

'''
To solve the overdetermined linear system of equations 
by least squares method
'''  
def solve_gls_4x4(y,X,S_inv):
    B = S_inv.dot(X)
    C = S_inv.dot(y)
    AtA = np.dot(X.T,B)
    yty = np.dot(X.T,C)
    AtA = np.vstack([AtA, np.ones(3)])
    AtA = np.vstack([AtA.T, np.ones(4)]).T
    AtA[-1,-1] = 0
    Aty = np.append(yty,[1])
    P = np.dot(np.linalg.inv(AtA),Aty)[0:3]
    return (P)

def solve_ols_4x4(y,X):
    #print(X.shape)
    #print(y.shape)
    X = np.array(X.T/y)
    y = y/y
    
    A = np.vstack([X.T, np.ones(len(X))])
    
    AtA = np.dot(A.T,A)
    AtA = np.vstack([AtA, np.ones(len(X))])
    AtA = np.vstack([AtA.T, np.ones(len(X)+1)]).T
    AtA[-1,-1] = 0
    
    y = np.append(y,[1])
    y = y[:, np.newaxis]
        
    Aty = np.dot(A.T,y)
    Aty = np.append(Aty,[1])

    P = np.dot(np.linalg.inv(AtA),Aty)
    return (P[0:len(X)])

def solve_minimize(y,A):
    from scipy.optimize import minimize
        
    #A = np.array([d.T ,e.T, l.T]).T
    
    P0 = np.array([0.3, 0.3, 0.3])
    X0 = -np.log((1./P0)-1.)
        
    def f(P):   
        return sum(((y-np.dot(A,P))/y)**2)
    
    def con(P):
        return P.sum()-1.0
       
    cons = ({'type': 'eq', 'fun': con})
    
    S = minimize(f, X0, 
                 method = "SLSQP", 
                 #options={'fatol': 0.0001},
                 constraints=cons)
    
    P = 1.0/(1.0+np.exp(-S.x))
    return(P)

def solve_minimize2(y,A):
    from scipy.optimize import minimize, Bounds, LinearConstraint
    #https://docs.scipy.org/doc/scipy/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
        
    #A =  np.array([d.T ,e.T, l.T]).T
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

def _buildSourceContribMatrix(a):
        g = len(a) # número de fontes
        A = np.ones((g+1,g+1)) 
        A[:g,:g] = np.dot(a,a.T)
        A[g, g]=0
        return A     

def solve_ols_4x4_mod(X, y):
        if normalize == True:
             d=d/y; e=e/y; l=l/y; y=y/y
        a = np.array([d/y,e/y,l/y])
        A = self._buildSourceContribMatrix(a)
        Z = np.append(np.dot(a, y/y), 1)
        P = np.dot(np.linalg.inv(A),Z)
        #P = solve(A, Z)
        return (P)
