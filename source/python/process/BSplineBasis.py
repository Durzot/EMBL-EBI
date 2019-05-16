# -*- coding: utf-8 -*-
"""
This class is inspired by BSplineBasis.java class of bigsnakeutils plugin
from Biomedical Imaging Group. 

Designed to be run in Python 3 virtual environment 3.7_vtk

Implements several basis B-spline functions.

@version: May 14, 2019
@author: Yoann Pradat
"""

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
def ConstantBSpline(t):
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
    if t >= -1.0 and t <= -0.0:
		SplineValue = t + 1
    elif t > -0.0 and t <= 1.0:
		SplineValue = -t + 1
	return SplineValue


# Causal quadratic B-spline
def QuadraticSpline(t):
	SplineValue = 0.0
	t -= 1.5
    if t >= -1.5 and t <= -0.5:
		SplineValue = 0.5 * t * t + 1.5 * t + 1.125
    elif t > -0.5 and t <= 0.5:
		SplineValue = -t * t + 0.75
    elif t > 0.5 and t <= 1.5:
		SplineValue = 0.5 * t * t - 1.5 * t + 1.125
	return SplineValue


