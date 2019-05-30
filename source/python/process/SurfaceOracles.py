# -*- coding: utf-8 -*-
"""
This class implements Oracles for evaluating point values and first order derivatives
of some surfaces.

Designed to be run in Python 3 virtual environment 3.7_vtk

@version: May 28, 2019
@author: Yoann Pradat
"""

import numpy as np
from snake3D.Snake3DNode import Snake3DNode

class OracleSphere(object):
    def __init__(self, R=1):
        self.R = R

    def value(self, u, v):
        x = self.R*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        y = self.R*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        z = self.R*np.cos(np.pi*v)
        return Snake3DNode(x, y, z)

    def v_deriv(self, u, v):
        x = self.R*np.pi*np.cos(2*np.pi*u)*np.cos(np.pi*v)
        y = self.R*np.pi*np.sin(2*np.pi*u)*np.cos(np.pi*v)
        z = -self.R*np.pi*np.sin(np.pi*v)
        return Snake3DNode(x, y, z)

    def u_deriv(self, u, v):
        x = -2*self.R*np.pi*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        y = 2*self.R*np.pi*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        z = 0
        return Snake3DNode(x, y, z)

    def uv_deriv(self, u, v):
        x = -2*self.R*np.pi**2*np.sin(2*np.pi*u)*np.cos(np.pi*v)
        y = 2*self.R*np.pi**2*np.cos(2*np.pi*u)*np.cos(np.pi*v)
        z = 0
        return Snake3DNode(x, y, z)


class OracleHalfSphere(object):
    def __init__(self, R=1):
        self.R = R

    def value(self, u, v):
        x = self.R*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        y = self.R*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        if v < 0.5:
            z = self.R*np.cos(np.pi*v)
        else:
            z = 0
        return Snake3DNode(x, y, z)

    def v_deriv(self, u, v):
        x = self.R*np.pi*np.cos(2*np.pi*u)*np.cos(np.pi*v)
        y = self.R*np.pi*np.sin(2*np.pi*u)*np.cos(np.pi*v)
        if v < 0.5:
            z = -self.R*np.pi*np.sin(np.pi*v)
        else:
            z = 0
        return Snake3DNode(x, y, z)

    def u_deriv(self, u, v):
        x = -2*self.R*np.pi*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        y = 2*self.R*np.pi*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        z = 0
        return Snake3DNode(x, y, z)

    def uv_deriv(self, u, v):
        x = -2*self.R*np.pi**2*np.sin(2*np.pi*u)*np.cos(np.pi*v)
        y = 2*self.R*np.pi**2*np.cos(2*np.pi*u)*np.cos(np.pi*v)
        z = 0
        return Snake3DNode(x, y, z)


class OracleCone(object):
    def __init__(self):
        pass
        
    def value(self, u, v):
        x = u        
        y = u*np.cos(v)
        z = u*np.sin(v)        
        return Snake3DNode(x, y, z)

    def v_deriv(self, u, v):
        x = 0        
        y = -u*np.sin(v)
        z = u*np.cos(v)        
        return Snake3DNode(x, y, z)

    def u_deriv(self, u, v):
        x = 1        
        y = np.cos(v)
        z = np.sin(v)        
        return Snake3DNode(x, y, z)

    def uv_deriv(self, u, v):
        x = 0        
        y = -np.sin(v)
        z = np.cos(v)        
        return Snake3DNode(x, y, z)

class OracleTorus(object):
    def __init__(self, R=2, r=0.5):
        self.R = R # Radius of big circle
        self.r = r # Radius of small circle
        
    def value(self, u, v):
        x = np.cos(2*np.pi*u)*(self.R + self.r*np.cos(2*np.pi*v))    
        y = np.sin(2*np.pi*u)*(self.R + self.r*np.cos(2*np.pi*v))    
        z = self.r*np.sin(2*np.pi*v)        
        return Snake3DNode(x, y, z)

    def v_deriv(self, u, v):
        x = np.cos(2*np.pi*u)*(-2*np.pi*self.r*np.sin(2*np.pi*v))    
        y = np.sin(2*np.pi*u)*(-2*np.pi*self.r*np.sin(2*np.pi*v))    
        z = 2*np.pi*self.r*np.cos(2*np.pi*v)        
        return Snake3DNode(x, y, z)

    def u_deriv(self, u, v):
        x = -2*np.pi*np.sin(2*np.pi*u)*(self.R + self.r*np.cos(2*np.pi*v))    
        y = 2*np.pi*np.cos(2*np.pi*u)*(self.R + self.r*np.cos(2*np.pi*v))    
        z = 0      
        return Snake3DNode(x, y, z)

    def uv_deriv(self, u, v):
        x = -2*np.pi*np.sin(2*np.pi*u)*(-2*np.pi*self.r*np.sin(2*np.pi*v))    
        y = 2*np.pi*np.cos(2*np.pi*u)*(-2*np.pi*self.r*np.sin(2*np.pi*v))    
        z = 0      
        return Snake3DNode(x, y, z)


