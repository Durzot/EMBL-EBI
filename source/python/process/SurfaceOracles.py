# -*- coding: utf-8 -*-
"""
This class implements Oracles for evaluating point values and first order derivatives
of some surfaces.

Designed to be run in Python 3 virtual environment 3.7_vtk

@version: June 10, 2019
@author: Yoann Pradat
"""

import numpy as np
from auxiliary.aux import Point3D

class OracleSphere(object):
    def __init__(self, R=1):
        self.R = R

    def value(self, u, v):
        x = self.R*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        y = self.R*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        z = self.R*np.cos(np.pi*v)
        return Point3D(x, y, z)

    def v_deriv(self, u, v):
        x = self.R*np.pi*np.cos(2*np.pi*u)*np.cos(np.pi*v)
        y = self.R*np.pi*np.sin(2*np.pi*u)*np.cos(np.pi*v)
        z = -self.R*np.pi*np.sin(np.pi*v)
        return Point3D(x, y, z)

    def u_deriv(self, u, v):
        x = -2*self.R*np.pi*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        y = 2*self.R*np.pi*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        z = 0
        return Point3D(x, y, z)

    def uv_deriv(self, u, v):
        x = -2*self.R*np.pi**2*np.sin(2*np.pi*u)*np.cos(np.pi*v)
        y = 2*self.R*np.pi**2*np.cos(2*np.pi*u)*np.cos(np.pi*v)
        z = 0
        return Point3D(x, y, z)


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
        return Point3D(x, y, z)

    def v_deriv(self, u, v):
        x = self.R*np.pi*np.cos(2*np.pi*u)*np.cos(np.pi*v)
        y = self.R*np.pi*np.sin(2*np.pi*u)*np.cos(np.pi*v)
        if v < 0.5:
            z = -self.R*np.pi*np.sin(np.pi*v)
        else:
            z = 0
        return Point3D(x, y, z)

    def u_deriv(self, u, v):
        x = -2*self.R*np.pi*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        y = 2*self.R*np.pi*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        z = 0
        return Point3D(x, y, z)

    def uv_deriv(self, u, v):
        x = -2*self.R*np.pi**2*np.sin(2*np.pi*u)*np.cos(np.pi*v)
        y = 2*self.R*np.pi**2*np.cos(2*np.pi*u)*np.cos(np.pi*v)
        z = 0
        return Point3D(x, y, z)


class OracleCone(object):
    def __init__(self):
        pass
        
    def value(self, u, v):
        x = u        
        y = u*np.cos(v)
        z = u*np.sin(v)        
        return Point3D(x, y, z)

    def v_deriv(self, u, v):
        x = 0        
        y = -u*np.sin(v)
        z = u*np.cos(v)        
        return Point3D(x, y, z)

    def u_deriv(self, u, v):
        x = 1        
        y = np.cos(v)
        z = np.sin(v)        
        return Point3D(x, y, z)

    def uv_deriv(self, u, v):
        x = 0        
        y = -np.sin(v)
        z = np.cos(v)        
        return Point3D(x, y, z)

class OracleTorus(object):
    def __init__(self, R=2, r=0.5):
        self.R = R # Radius of big circle
        self.r = r # Radius of small circle
        
    def value(self, u, v):
        x = np.cos(2*np.pi*u)*(self.R + self.r*np.cos(2*np.pi*v))    
        y = np.sin(2*np.pi*u)*(self.R + self.r*np.cos(2*np.pi*v))    
        z = self.r*np.sin(2*np.pi*v)        
        return Point3D(x, y, z)

    def v_deriv(self, u, v):
        x = np.cos(2*np.pi*u)*(-2*np.pi*self.r*np.sin(2*np.pi*v))    
        y = np.sin(2*np.pi*u)*(-2*np.pi*self.r*np.sin(2*np.pi*v))    
        z = 2*np.pi*self.r*np.cos(2*np.pi*v)        
        return Point3D(x, y, z)

    def u_deriv(self, u, v):
        x = -2*np.pi*np.sin(2*np.pi*u)*(self.R + self.r*np.cos(2*np.pi*v))    
        y = 2*np.pi*np.cos(2*np.pi*u)*(self.R + self.r*np.cos(2*np.pi*v))    
        z = 0      
        return Point3D(x, y, z)

    def uv_deriv(self, u, v):
        x = -2*np.pi*np.sin(2*np.pi*u)*(-2*np.pi*self.r*np.sin(2*np.pi*v))    
        y = 2*np.pi*np.cos(2*np.pi*u)*(-2*np.pi*self.r*np.sin(2*np.pi*v))    
        z = 0      
        return Point3D(x, y, z)


class OracleKleinBottle(object):
    def __init__(self):
        pass
    
    def value(self, u, v):
        cosu = np.cos(2*np.pi*u)
        sinu = np.sin(2*np.pi*u)
        cosv = np.cos(2*np.pi*v)
        sinv = np.sin(2*np.pi*v)
        coshalfu = np.cos(np.pi*u)
        sinhalfu = np.sin(np.pi*u)
 
        x = cosu*(coshalfu*(np.sqrt(2)+cosv) + sinhalfu*sinv*cosv)
        y = sinu*(coshalfu*(np.sqrt(2)+cosv) + sinhalfu*sinv*cosv)
        z = -sinhalfu*(np.sqrt(2)+cosv) + coshalfu*sinv*cosv
        return Point3D(x, y, z)

    def v_deriv(self, u, v):
        cosu = np.cos(2*np.pi*u)
        sinu = np.sin(2*np.pi*u)
        cosv = np.cos(2*np.pi*v)
        sinv = np.sin(2*np.pi*v)
        coshalfu = np.cos(np.pi*u)
        sinhalfu = np.sin(np.pi*u)
 
        x = cosu*(-2*np.pi*coshalfu*sinv + 2*np.pi*sinhalfu*(cosv**2-sinv**2))
        y = sinu*(-2*np.pi*coshalfu*sinv + 2*np.pi*sinhalfu*(cosv**2-sinv**2))
        z = 2*np.pi*sinhalfu*sinv + 2*np.pi*coshalfu*(cosv**2-sinv**2)
        return Point3D(x, y, z)

    def u_deriv(self, u, v):
        cosu = np.cos(2*np.pi*u)
        sinu = np.sin(2*np.pi*u)
        cosv = np.cos(2*np.pi*v)
        sinv = np.sin(2*np.pi*v)
        coshalfu = np.cos(np.pi*u)
        sinhalfu = np.sin(np.pi*u)
 
        x = -2*np.pi*sinu*(coshalfu*(np.sqrt(2)+cosv) + sinhalfu*sinv*cosv) \
                + cosu*(-np.pi*sinhalfu*(np.sqrt(2)+cosv) + np.pi*coshalfu*sinv*cosv)
        y = 2*np.pi*cosu*(coshalfu*(np.sqrt(2)+cosv) + sinhalfu*sinv*cosv) \
                + sinu*(-np.pi*sinhalfu*(np.sqrt(2)+cosv) + np.pi*coshalfu*sinv*cosv)
        z = -np.pi*coshalfu*(np.sqrt(2)+cosv) - np.pi*sinhalfu*sinv*cosv
        return Point3D(x, y, z)

    def uv_deriv(self, u, v):
        cosu = np.cos(2*np.pi*u)
        sinu = np.sin(2*np.pi*u)
        cosv = np.cos(2*np.pi*v)
        sinv = np.sin(2*np.pi*v)
        coshalfu = np.cos(np.pi*u)
        sinhalfu = np.sin(np.pi*u)
 
        x = -2*np.pi*sinu*(-2*np.pi*coshalfu*sinv + 2*np.pi*sinhalfu*(cosv**2-sinv**2)) \
                + cosu*(2*np.pi**2*sinhalfu*sinv + 2*np.pi**2*coshalfu*(cosv**2-sinv**2)) 
        y = 2*np.pi*cosu*(-2*np.pi*coshalfu*sinv + 2*np.pi*sinhalfu*(cosv**2-sinv**2)) \
                + sinu*(2*np.pi**2*sinhalfu*sinv + 2*np.pi**2*coshalfu*(cosv**2-sinv**2))
        z = 2*np.pi**2*coshalfu*sinv - 2*np.pi**2*sinhalfu*(cosv**2-sinv**2)
        return Point3D(x, y, z)


class OracleKlein8(object):
    def __init__(self, a=4):
        self.a = a

    def value(self, u,v):
        cosu = np.cos(2*np.pi*u)
        sinu = np.sin(2*np.pi*u)
        cosv = np.cos(2*np.pi*v)
        sinv = np.sin(2*np.pi*v)
        coshalfu = np.cos(np.pi*u)
        sinhalfu = np.sin(np.pi*u)
        cosdbv = np.cos(4*np.pi*v)
        sindbv = np.sin(4*np.pi*v)

        x = (self.a+coshalfu*sinv - sinhalfu*sindbv)*cosu
        y = (self.a+coshalfu*sinv - sinhalfu*sindbv)*sinu
        z = sinhalfu*sinv + coshalfu*sindbv
        return Point3D(x, y, z)

    def v_deriv(self, u, v):
        cosu = np.cos(2*np.pi*u)
        sinu = np.sin(2*np.pi*u)
        cosv = np.cos(2*np.pi*v)
        sinv = np.sin(2*np.pi*v)
        coshalfu = np.cos(np.pi*u)
        sinhalfu = np.sin(np.pi*u)
        cosdbv = np.cos(4*np.pi*v)
        sindbv = np.sin(4*np.pi*v)

        x = (2*np.pi*coshalfu*cosv - 4*np.pi*sinhalfu*cosdbv)*cosu
        y = (2*np.pi*coshalfu*cosv - 4*np.pi*sinhalfu*cosdbv)*sinu
        z = 2*np.pi*sinhalfu*cosv + 4*np.pi*coshalfu*cosdbv
        return Point3D(x, y, z)

    def u_deriv(self, u, v):
        cosu = np.cos(2*np.pi*u)
        sinu = np.sin(2*np.pi*u)
        cosv = np.cos(2*np.pi*v)
        sinv = np.sin(2*np.pi*v)
        coshalfu = np.cos(np.pi*u)
        sinhalfu = np.sin(np.pi*u)
        cosdbv = np.cos(4*np.pi*v)
        sindbv = np.sin(4*np.pi*v)

        x = -2*np.pi*(self.a+coshalfu*sinv - sinhalfu*sindbv)*sinu \
                + (-np.pi*sinhalfu*sinv - np.pi*coshalfu*sindbv)*cosu
        y = 2*np.pi*(self.a+coshalfu*sinv - sinhalfu*sindbv)*cosu \
            + (-np.pi*sinhalfu*sinv - np.pi*coshalfu*sindbv)*sinu
        z = np.pi*coshalfu*sinv - np.pi*sinhalfu*sindbv
        return Point3D(x, y, z)

    def uv_deriv(self, u, v):
        cosu = np.cos(2*np.pi*u)
        sinu = np.sin(2*np.pi*u)
        cosv = np.cos(2*np.pi*v)
        sinv = np.sin(2*np.pi*v)
        coshalfu = np.cos(np.pi*u)
        sinhalfu = np.sin(np.pi*u)
        cosdbv = np.cos(4*np.pi*v)
        sindbv = np.sin(4*np.pi*v)

        x = -2*np.pi*(2*np.pi*coshalfu*cosv - 4*np.pi*sinhalfu*cosdbv)*sinu \
                + (-2*np.pi**2*sinhalfu*cosv - 4*np.pi**2*coshalfu*cosdbv)*cosu
        y = 2*np.pi*(2*np.pi*coshalfu*cosv - 4*np.pi*sinhalfu*cosdbv)*cosu \
                + (-2*np.pi**2*sinhalfu*cosv - 4*np.pi**2*coshalfu*cosdbv)*sinu
        z = 2*np.pi**2*coshalfu*cosv - 4*np.pi**2*sinhalfu*cosdbv
        return Point3D(x, y, z)
