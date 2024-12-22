#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on jul 2024

@author: tiagoburiol
@coworker: buligonl; josue@sehnem.com
"""

import numpy as np
import statsmodels.api as sm

'''
To solve the overdetermined linear system of equations 
by least squares method
'''  
def solve_gls_4x4(y,X,S_inv):
    '''
    Solves the generalized least squares (GLS) problem for a 4x4 system of equations.
    Purpose: to solve the overdetermined linear system of equations by generalized least squares method

    Parameters
    --------------------
    y : numpy.ndarray
        A 1D numpy array containing the selected Y (target) values.
    X : numpy.ndarray
        A 2D numpy array containing the sources' selected proportions values.
    S_inv : numpy.ndarray
        A 2D numpy array containing the inverse of the covariance matrix.

    Returns
    --------------------
    numpy.ndarray
        A 1D numpy array containing the estimated parameters of the GLS model.

    Raises
    --------------------
    None
    '''
    #print (X.shape) #7x3
    #print(S_inv.shape) #7x7
    B = S_inv.dot(X)#7x3
    C = S_inv.dot(y)#7x1
    
    AtA = np.dot(X.T,B)#3x3
    yty = np.dot(X.T,C)#3x1
    AtA = np.hstack((AtA, np.ones((X.shape[1],1))))
    AtA = np.vstack((AtA, np.ones((1,X.shape[1]+1))))
    AtA[-1,-1] = 0
    Aty = np.append(yty,[1])
    P = np.dot(np.linalg.inv(AtA),Aty)[0:X.shape[1]]
    return (P)

def solve_ols_cm(y,X):
    '''
    Solves the ordinary least squares (OLS) problem with constant mean (CM) constraints.

    Parameters
    --------------------
    y : numpy.ndarray
        A 1D numpy array containing the selected Y (target) values.
    X : numpy.ndarray
        A 2D numpy array containing the sources' selected proportions values.

    Returns
    --------------------
    numpy.ndarray
        A 1D numpy array containing the estimated parameters of the OLS-CM model, excluding the constant term.

    Raises
    --------------------
    None
    '''
    X = np.divide(X,y.reshape(len(X),1))
    A = X.T@X
    A = np.hstack((A, np.ones((len(A),1))))
    A = np.vstack((A, np.ones((1,len(A)+1))))
    A[-1,-1]=0
    Z = np.dot(X.T,np.ones(len(X)))
    Z = np.append(Z,[1])
    #print(A)
    #print(Z)
    P = np.linalg.inv(A)@Z
    return (P[0:-1])

## needs statmodel module
## same results of solve_ols
#def solve_ols_sm(y,X):
#    X = np.divide(X,y.reshape(len(X),1))
#    X = np.vstack([X, np.ones((1,X.shape[1]))])
#    y = np.ones(len(X)) 
#    ols_model = sm.OLS(y, X).fit()
#    return (ols_model.params)


def solve_ols(y,X):
    '''
    Solves the ordinary least squares (OLS) problem.

    Parameters
    --------------------
    y : numpy.ndarray
        A 1D numpy array containing the selected Y (target) values.
    X : numpy.ndarray
        A 2D numpy array containing the sources' selected proportions values.

    Returns
    --------------------
    numpy.ndarray
        A 1D numpy array containing the estimated parameters of the OLS model.

    Raises
    --------------------
    None
    '''
    X = np.divide(X,y.reshape(len(X),1))
    y = np.ones(len(X)) 
    P = np.linalg.inv(X.T@X)@(X.T@y)
    return (P)


def solve_minimize(y,A):
    '''
    Solves an optimization problem using the scipy.optimize.minimize function.

    Parameters
    --------------------
    y : numpy.ndarray
        A 1D numpy array containing the selected Y (target) values.
    X : numpy.ndarray
        A 2D numpy array containing the sources' selected proportions values.

    Returns
    --------------------
    numpy.ndarray
        A 1D numpy array containing the estimated parameters of the optimization problem.

    Raises
    --------------------
    None
    '''
    from scipy.optimize import minimize
        
    #A = np.array([d.T ,e.T, l.T]).T
    
    P0 = np.array([0.3, 0.3, 0.3])
    X0 = -np.log((1./P0)-1.)
        
    def f(P):
        '''
        Objective function to be minimized.

        Parameters
        --------------------
        P: numpy.ndarray
            A 1D numpy array containing the parameters to be optimized.

        Returns
        --------------------
        float
            The value of the objective function.

        Raises
        --------------------
        None
        '''
        return sum(((y-np.dot(A,P))/y)**2)
    
    def con(P):
        '''
        Constraint function for the optimization problem.

        Parameters
        --------------------
        P: numpy.ndarray
            A 1D numpy array containing the parameters to be optimized.

        Returns
        --------------------
        float
            The value of the constraint function.

        Raises
        --------------------
        None
        '''
        return P.sum()-1.0
       
    cons = ({'type': 'eq', 'fun': con})
    
    S = minimize(f, X0, 
                 method = "SLSQP", 
                 #options={'fatol': 0.0001},
                 constraints=cons)
    
    P = 1.0/(1.0+np.exp(-S.x))
    return(P)

def solve_minimize2(y,A):
    """
    Solves a constrained error minimization problem using the SciPy `minimize` function.

    Parameters
    ----------
    y : numpy.ndarray
        A 1D numpy array containing the selected Y (target) values.
    A : numpy.ndarray
        A 2D numpy array containing the sources' selected proportions values.

    Returns
    -------
    numpy.ndarray
        The optimal possible proportions solution vector.

    Raises
    ------
    None
    """
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




