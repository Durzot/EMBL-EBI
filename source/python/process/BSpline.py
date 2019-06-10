# -*- coding: utf-8 -*-
"""
This class is inspired by BSplineBasis.java class of bigsnakeutils plugin
from Biomedical Imaging Group. 

Designed to be run in Python 3 virtual environment 3.7_vtk

Implements several basis B-spline functions and interpolation algorithm

@version: May 14, 2019
@author: Yoann Pradat
"""

import numpy as np

# ----------------------------------------------------------------------------
# BASIS FUNCTIONS' SUPPORT

# Length of the support of the linear B-spline. 
LINEARBSPLINESUPPORT = 2
# Length of the support of the quadratic B-spline
QUADRATICBSPLINESUPPORT = 3
# Length of the support of the cubic B-spline
CUBICBSPLINESUPPORT = 4

# Length of the support of the exponential B-spline basis function with
# three roots.
 
ESPLINE3SUPPORT = 3

# Length of the support of the exponential B-spline basis function with
# four roots.
ESPLINE4SUPPORT = 4

# ============================================================================
# FUNCTIONS

# ----------------------------------------------------------------------------
# B-SPLINE FUNCTIONS

# Causal constant B-spline.
def ConstantBspline(t):
    SplineValue = 0.0
    t -= 0.5
    if t > -0.5 and t < 0.5:
        SplineValue = 1
    elif t == 0.5 or t == -0.5:
        SplineValue = 0.5
    return SplineValue

# Causal linear B-spline
def LinearBSpline(t):
    SplineValue = 0.0
    t -= 1.0
    if t >= -1.0 and t <= 0.0:
        SplineValue = t+1
    elif t > 0.0 and t <= 1.0:
        SplineValue = -t+1
    return SplineValue

# ============================================================================
# Algorithms for interpolation

def DivDiff(tau, gtau):
    tau = np.array(tau)
    n = tau.shape[0] 
    
    # Dictionary of divided difference of increasing order
    div_diff = {k: np.zeros(n-k) for k in range(n)}
    div_diff[0] = gtau

    for k in range(1, n):
        for i in range(n-k):
            if tau[i] == tau[i+k]:
                raise ValueError("Please choose distinct sites")
            else:
                div_diff[k][i] = (div_diff[k-1][i+1]-div_diff[k-1][i])/(tau[i+k]-tau[i])
    return div_diff[n-1][0]

def SlopeCSNotAKnot(tau, gtau):
    tau = np.array(tau) # Numpy array 
    gtau = np.array(gtau)

    # Number of points
    n = tau.shape[0]

    # Slopes
    S = np.zeros(n)
    
    # Matrices for the system AS = B
    A = np.zeros((n,n))
    B = np.zeros(n)
    
    # Equation for C^3 continuous at first interior break
    A[0,0] = tau[2]-tau[1]
    A[0,1] = tau[2]-tau[0]
    B[0] = ((tau[1]-tau[0] + 2*(tau[2]-tau[0]))*(tau[2]-tau[1])*DivDiff(tau[0:2], gtau[0:2]) +
            (tau[1]-tau[0])**2*DivDiff(tau[1:3], gtau[1:3]))/(tau[2]-tau[0])
    
    # Compute slopes that make interpolant C^2 continuous at n-2 interior breaks
    for i in range(1, n-1):
        Dtau_i = tau[i+1]-tau[i]
        Dtau_im = tau[i]-tau[i-1]
        
        A[i,i-1] = Dtau_i
        A[i,i] = 2*(Dtau_i+Dtau_im)
        A[i,i+1] = Dtau_im
        B[i] = 3*((Dtau_i*DivDiff(tau[i-1:i+1], gtau[i-1:i+1]) + Dtau_im*DivDiff(tau[i:i+2], gtau[i:i+2])))
        
    # Equation for C^3 continuous at last interior break
    A[-1,-2] = tau[-1]-tau[-3]
    A[-1,-1] = tau[-2]-tau[-3]
    B[-1] = ((tau[-1]-tau[-2])**2*DivDiff(tau[-3:-1], gtau[-3:-1]) + 
             (2*(tau[-1]-tau[-3]) + tau[-1]-tau[-2])*(tau[-2]-tau[-3])*DivDiff(tau[-2:], gtau[-2:]))/(tau[-1]-tau[-3])
            
    # Solve the system for S    
    S = np.linalg.inv(A).dot(B)

    return S

    
