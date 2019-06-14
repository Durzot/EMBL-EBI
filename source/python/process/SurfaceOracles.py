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

    def uu_deriv(self, u, v):
        x = -4*self.R*np.pi**2*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        y = -4*self.R*np.pi**2*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        z = 0
        return Point3D(x, y, z)

    def vv_deriv(self, u, v):
        x = -self.R*np.pi**2*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        y = -self.R*np.pi**2*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        z = -self.R*np.pi**2*np.cos(np.pi*v)
        return Point3D(x, y, z)

    def gauss_curv(self, u,v):
        return 1/self.R**2


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
    
    def gauss_curv(self, u,v):
        return np.cos(2*np.pi*v)/(self.r*(self.R+self.r*np.cos(2*np.pi*v)))


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

# Surface from Hermite exponential splines representation
class OraclePolySphere(object):
    def __init__(self, R, M_1, M_2):
        self.R = R
        self.M_1 = M_1
        self.w_1 = 2*np.pi/M_1
        self.M_2 = M_2
        self.w_2 = np.pi/M_2

        self.oracle = OracleSphere(R)

    def _Dg_1(self, w, x, order=1):
        assert 0 <= x
        if 1 < x:
            return 0
        else:
            qw = np.exp(w*1j)*(w*1j-2) + w*1j+2
            aw = (w*1j + 1 + np.exp(w*1j)*(w*1j-1))/qw
            bw = -w*1j*(np.exp(w*1j)+1)/qw
            cw = 1/qw
            dw = -np.exp(w*1j)/qw

            if order==0:
                return (aw + bw*x + cw*np.exp(w*x*1j) + dw*np.exp(-w*x*1j)).real
            elif order==1:
                return (bw + cw*w*1j*np.exp(w*x*1j) - dw*w*1j*np.exp(-w*x*1j)).real
            else:
                return (cw*(w*1j)**order*np.exp(w*x*1j) + dw*(-w*1j)**order*np.exp(-w*x*1j)).real
    
    def _phi_1(self, w, x):
        if 1 <= np.abs(x):
            return 0
        else:
            return self._Dg_1(w, np.abs(x), order=0)

    def _Dg_2(self, w, x, order=1):
        assert 0 <= x
        if 1 < x:
            return 0
        else:
            qw = np.exp(w*1j)*(w*1j-2) + w*1j+2
            pw = np.exp(2*w*1j)*(w*1j-1) + w*1j+1
            aw = pw/(w*1j*(np.exp(w*1j)-1)*qw)
            bw = -(np.exp(w*1j) -1)/qw
            cw = (np.exp(w*1j)-w*1j-1)/(w*1j*(np.exp(w*1j)-1)*qw)
            dw = -np.exp(w*1j)*(np.exp(w*1j)*(w*1j-1)+1)/(w*1j*(np.exp(w*1j)-1)*qw)

            if order==0:
                return (aw + bw*x + cw*np.exp(w*x*1j) + dw*np.exp(-w*x*1j)).real
            elif order==1:
                return (bw + cw*w*1j*np.exp(w*x*1j) - dw*w*1j*np.exp(-w*x*1j)).real
            else:
                return (cw*(w*1j)**order*np.exp(w*x*1j) + dw*(-w*1j)**order*np.exp(-w*x*1j)).real

    def _phi_2(self, w, x):
        if 1 <= np.abs(x):
            return 0
        else:
            if 0 <= x:
                return self._Dg_2(w, x, order=0)
            else:
                return -self._Dg_2(w, -x, order=0)


    def value(self, u, v):
        value = Point3D(0, 0, 0)
        for k in range(self.M_1):
            for l in range(self.M_2+1):
                phi_1_w1_per = self._phi_1(self.w_1, self.M_1*u-k) + self._phi_1(self.w_1, self.M_1*(u-1)-k)
                phi_2_w1_per = self._phi_2(self.w_1, self.M_1*u-k) + self._phi_2(self.w_1, self.M_1*(u-1)-k)
                phi_1_w2 = self._phi_1(self.w_2, self.M_2*v-l)
                phi_2_w2 = self._phi_2(self.w_2, self.M_2*v-l)

                c_1 = self.oracle.value(k/self.M_1, l/self.M_2)
                #c_2 = self.oracle.v_deriv(k/self.M_1, l/self.M_2)
                #c_3 = self.oracle.u_deriv(k/self.M_1, l/self.M_2)
                #c_4 = self.oracle.uv_deriv(k/self.M_1, l/self.M_2)
                value += c_1*phi_1_w1_per*phi_1_w2
                #value += c_2*phi_1_w1_per*phi_2_w2
                #value += c_3*phi_2_w1_per*phi_1_w2 
                #value += c_4*phi_2_w1_per*phi_2_w2

        return value

    def u_deriv(self, u, v):
        return Point3D(0,0,0)

    def v_deriv(self, u, v):
        return Point3D(0,0,0)

    def uv_deriv(self, u, v):
        return Point3D(0,0,0)


class OracleBone(object):
    def __init__(self, R=1):
        self.R = R
        self.M = 5
        self.c_1 = [0, 0.5, 0.25, 0.5, 0] # Values to interpolate
        self.c_2 = [1, 0, 0, 0, -1] # Derivatives to interpolate
    
    def value(self, u, v):
        x = self.R*np.cos(2*np.pi*u)*np.sin(np.pi*v)
        y = self.R*np.sin(2*np.pi*u)*np.sin(np.pi*v)
        z = self.R*np.cos(np.pi*v)
        return Point3D(x, y, z)

    def hphi_1(self, x):
        if 1 <= np.abs(x):
            return 0
        else:
            return 1 - 3*x**2 + 2*np.abs(x)**3
        
    def hphi_2(self, x):
        if 1 <= np.abs(x):
            return 0
        else:
            if 0 <= x:
                return x - 2*x**2 + x**3
            else:
                return x + 2*x**2 + x**3
            
    def dhphi_1(self, x):
        if 1 <= np.abs(x):
            return 0
        else:
            if 0 <= x:
                return - 6*x + 6*x**2
            else:
                return - 6*x - 6*x**2
        
    def dhphi_2(self, x):
        if 1 <= np.abs(x):
            return 0
        else:
            if 0 <= x:
                return 1 - 4*x + 3*x**2
            else:
                return 1 + 4*x + 3*x**2
            
    def func(self, v):
        assert 0 <= v and v <=1 
        M = self.M
        return sum([self.c_1[k]*self.hphi_1((M-1)*v-k) + self.c_2[k]*self.hphi_2((M-1)*v-k) for k in range(M)])
    
    def dfunc(self, v):
        assert 0 <= v and v <=1 
        M = self.M
        return sum([self.c_1[k]*(M-1)*self.dhphi_1((M-1)*v-k) + self.c_2[k]*(M-1)*self.dhphi_2((M-1)*v-k) for k in range(M)])
    
    
    def value(self, u, v):
        x = self.R*np.cos(2*np.pi*u)*self.func(v)
        y = self.R*np.sin(2*np.pi*u)*self.func(v)
        z = self.R*np.cos(np.pi*v)
        return Point3D(x, y, z)

    def v_deriv(self, u, v):
        x = self.R*np.cos(2*np.pi*u)*self.dfunc(v)
        y = self.R*np.sin(2*np.pi*u)*self.dfunc(v)
        z = -self.R*np.pi*np.sin(np.pi*v)
        return Point3D(x, y, z)

    def u_deriv(self, u, v):
        x = -2*self.R*np.pi*np.sin(2*np.pi*u)*self.func(v)
        y = 2*self.R*np.pi*np.cos(2*np.pi*u)*self.func(v)
        z = 0
        return Point3D(x, y, z)

    def uv_deriv(self, u, v):
        x = -2*self.R*np.pi*np.sin(2*np.pi*u)*self.dfunc(v)
        y = 2*self.R*np.pi*np.cos(2*np.pi*u)*self.dfunc(v)
        z = 0
        return Point3D(x, y, z)


#class OracleX(object):
#    def __init__(self, R):
#        self.R = R
#
#    def value(self, u, v):
#        x = self.R*np.cos(2*np.pi*u)*np.sin(np.pi*v)
#        y = self.R*np.sin(2*np.pi*u)*np.sin(np.pi*v)
#        z = self.R*np.cos(2*np.pi*u)*np.cos(np.pi*v)
#        return Point3D(x, y, z)
#
#    def v_deriv(self, u, v):
#        x = self.R*np.pi*np.cos(2*np.pi*u)*np.cos(np.pi*v)
#        y = self.R*np.pi*np.sin(2*np.pi*u)*np.cos(np.pi*v)
#        z = -self.R*np.pi*np.cos(2*np.pi*u)*np.sin(np.pi*v)
#        return Point3D(x, y, z)
#
#    def u_deriv(self, u, v):
#        x = -2*self.R*np.pi*np.sin(2*np.pi*u)*np.sin(np.pi*v)
#        y = 2*self.R*np.pi*np.cos(2*np.pi*u)*np.sin(np.pi*v)
#        z = -2*np.pi*self.R*np.sin(2*np.pi*u)*np.cos(np.pi*v)
#        return Point3D(x, y, z)
#
#    def uv_deriv(self, u, v):
#        x = -2*self.R*np.pi**2*np.sin(2*np.pi*u)*np.cos(np.pi*v)
#        y = 2*self.R*np.pi**2*np.cos(2*np.pi*u)*np.cos(np.pi*v)
#        z = 2*np.pi**2*self.R*np.sin(2*np.pi*u)*np.sin(np.pi*v)
#        return Point3D(x, y, z)
