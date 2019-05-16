# -*- coding: utf-8 -*-
"""
This class is inspired by SphereSnake.java class of bigsnake3d plugin
from Biomedical Imaging Group. 

Designed to be run in Python 3 virtual environment 3.7_vtk

Three-dimensional exponential spline snake

@version: May 16, 2019
@author: Yoann Pradat
"""

import numpy as np
from auxiliary.aux import Color, Point3D
from snake3D.Snake3DNode import Snake3DNode
from snake3D.Snake3DScale import Snake3DScale

class SphereSnake(object):
    def __init__(self, M_1, M_2, nSamplesPerSeg=11):
        # Number of control points on latitudes
        self.M_1 = M_1

        # Number of control points on longitudes
        self.M_2 = M_2

        # Number of samples between consecutive control points 
        # for drawing sking of the snake
        self.nSamplesPerSeg = nSamplesPerSeg

        # Number of points on latitudes for scale (closed curve)
        self.MR_1 = nSamplesPerSeg*M_1

        # Number of points on longitudes for scale (open curve)
        self.MR_2 = nSamplesPerSeg*M_2

        # Number of coefficients of spline representation
        self.nCoefs = M_1*(M_2+3)
        self.coefs = [Snake3DNode(0, 0, 0) for _ in range(self.nCoefs)]

        # Number of free parameters
        # For continuous representation coefs values c[k,-1], c[k,0], c[k, M_2] 
        # c[k, M_2+1] for k=0,..,M_1-1 are constrained by other values of the coefs
        # and poles related vectors: CN, T_1N, T_2N and same for south
        self.nFreeParams = M_1*(M_2-1) + 6
        self.freeParams = [Snake3DNode(0, 0, 0) for _ in range(self.nFreeParams)]

        # Samples of the coordinates of the snake contour 
        self.snakeContour = [[Point3D(0, 0, 0) for _ in range(self.MR_2)] for _ in range(self.MR_1)]

    """ Update values of coefficients for representation of the spline
        from values of the free parameters
    """
    def _updateCoefs(self):
       coefs = [Snake3DNode(0, 0, 0) for _ in range(self.nCoefs)]
       M_1 = self.M_1
       M_2 = self.M_2

       # c[k,l] k=0,.., M_1-1, l=1,.., M_2-1
       for l in range(1, M_2):
           for k in range(M_1):
               coefs[k + (l+1)*M_1] = self.freeParams[k + (l-1)*M_1]

       for k in range(M_1):
           CN = self.freeParams[M_1 * (M_2-1)]
           CS = self.freeParams[M_1 * (M_2-1)+1]
           T_1N = self.freeParams[M_1 * (M_2-1)+2]
           T_2N = self.freeParams[M_1 * (M_2-1)+3]
           T_1S = self.freeParams[M_1 * (M_2-1)+4]
           T_2S = self.freeParams[M_1 * (M_2-1)+5]

           CM1 = self._CM(k, M_1) 
           SM1 = self._SM(k, M_1) 

           phi_2_0 = self._phi_2(0)
           phi_2_1 = self._phi_2(1)
           phi_2_prime_1 = self._phi_2_prime(1)
        
           # c[k, -1]
           coefs[k] = coefs[k + 2*M_1] + (T_1N*CM1 + T_2N*SM1)/(M_2*phi_2_prime_1)
           
           # c[k, M_2+1]
           coefs[k + (M_2+2)*M_1] = coefs[k + M_2*M_1] - (T_1S*CM1 + T_2S*SM1)/(M_2*phi_2_prime_1)

           # c[k, 0]
           coefs[k + M_1] = CN/phi_2_0 - (coefs[k] + coefs[k + 2*M_1])*phi_2_1/phi_2_0
            
           # c[k, M_2]
           coefs[k + (M_2+1)*M_1] =  CS/phi_2_0 - (coefs[k + M_2*M_1] + coefs[k + (M_2+2)*M_1])*phi_2_1/phi_2_0
       
       self.coefs = coefs

    def _surfaceValue(self, u, v):
        assert 0 <= u and u <= 1
        assert 0 <= v and v <= 1

        # The calculation assumes a regular grid
        surfaceValue = Snake3DNode(0, 0, 0)
        for k in range(self.M_1):
            for l in range(self.M_2+3):
                phi_1 = self._phi_1(self.M_1*u-k) + self._phi_1(self.M_1*(u-1)-k) + self._phi_1(self.M_1*(u+1)-k)
                phi_2 = self._phi_2(self.M_2*v-l+1)
                coef = self.coefs[k + l*self.M_1] 
                surfaceValue += coef*phi_1*phi_2

        return surfaceValue

    def _updateContour(self):
        for k in range(self.MR_1):
            for l in range(self.MR_2):
                surfaceValue = self._surfaceValue(k/self.MR_1, l/(self.MR_2-1))
                if abs(surfaceValue.x) < 1e-10:
                    surfaceValue.x = 0
                if abs(surfaceValue.y) < 1e-10:
                    surfaceValue.y = 0
                if abs(surfaceValue.z) < 1e-10:
                    surfaceValue.z = 0
                self.snakeContour[k][l].updateFromPoint(surfaceValue)

    def _CM(self, k, M):
        PI_M = np.pi/M
        num = 2*(1-np.cos(2*PI_M))*np.cos(2*k*PI_M)
        den = np.cos(PI_M) - np.cos(3*PI_M)
        return num/den

    def _SM(self, k, M):
        PI_M = np.pi/M
        num = 2*(1-np.cos(2*PI_M))*np.sin(2*k*PI_M)
        den = np.cos(PI_M) - np.cos(3*PI_M)
        return num/den

    def _phi_1(self, u):
        u = np.abs(u)
        PI_M = np.pi/self.M_1
        if 0 <= u and u < 0.5:
            num = np.cos(2*PI_M*u)*np.cos(PI_M) - np.cos(2*PI_M)
            den = 1 - np.cos(2*PI_M)
            return num/den
        elif 0.5 <= u and u < 1.5:
            num = 1 - np.cos(2*PI_M*(1.5-u))
            den = 2*(1 - np.cos(2*PI_M))
            return num/den
        else:
            return 0

    def _phi_2(self, v):
        v = np.abs(v)
        PI_M = np.pi/(2*self.M_2)
        if 0 <= v and v < 0.5:
            num = np.cos(2*PI_M*v)*np.cos(PI_M) - np.cos(2*PI_M)
            den = 1 - np.cos(2*PI_M)
            return num/den
        elif 0.5 <= v and v < 1.5:
            num = 1 - np.cos(2*PI_M*(1.5-v))
            den = 2*(1 - np.cos(2*PI_M))
            return num/den
        else:
            return 0

    def _phi_2_prime(self, v):
        PI_M = np.pi/(2*self.M_2)
        if -0.5 < v and v < 0.5:
            return -2*PI_M*np.sin(2*PI_M*v)*np.cos(PI_M)/(1 - np.cos(2*PI_M))
        elif -1.5 < v and v <= -0.5:
            return 2*PI_M*np.sin(2*PI_M*(1.5+v))/(2 - 2*np.cos(2*PI_M))
        elif 0.5 <= v and v < 1.5:
            return -2*PI_M*np.sin(2*PI_M*(1.5-v))/(2 - 2*np.cos(2*PI_M))
        else:
            return 0

    # Initalize snake free parameters with a sphere shape
    def initializeDefaultShape(self):
        freeParams = [Snake3DNode(0, 0, 0) for _ in range(self.nFreeParams)]

        # Coefficients c[k,l] k=0, ..., M_1-1, l=1, ..., M_2-1
        for l in range(1, self.M_2):
            for k in range(self.M_1):
                x = self._CM(k, self.M_1)*self._SM(l, 2*self.M_2)
                y = self._SM(k, self.M_1)*self._SM(l, 2*self.M_2)
                z = self._CM(l, 2*self.M_2)
                freeParams[k + (l-1)*self.M_1] = Snake3DNode(x, y, z)

        # North Pole
        freeParams[self.M_1 * (self.M_2-1)] = Snake3DNode(0, 0, 1)
        # South Pole
        freeParams[self.M_1 * (self.M_2-1)+1] = Snake3DNode(0, 0, -1)
        # North tangent plane
        freeParams[self.M_1 * (self.M_2-1)+2] = Snake3DNode(np.pi, 0, 0)
        freeParams[self.M_1 * (self.M_2-1)+3] = Snake3DNode(0, np.pi, 0)
        # South tangent plane
        freeParams[self.M_1 * (self.M_2-1)+4] = Snake3DNode(-np.pi, 0, 0)
        freeParams[self.M_1 * (self.M_2-1)+5] = Snake3DNode(0, -np.pi, 0)

        self.freeParams = freeParams
        self._updateCoefs()
        self._updateContour()

    def getNumScales(self):
        return self.MR_1 + self.MR_2

    """ Returns the ith scale """
    def _getScale(self, i):
        # Longitude scale
        if i >= 0 and i < self.MR_1:
            points = np.zeros((self.MR_2, 3))
            for v in range(self.MR_2):
                points[v, 0] = self.snakeContour[i][v].x
                points[v, 1] = self.snakeContour[i][v].y
                points[v, 2] = self.snakeContour[i][v].z              
            scale = Snake3DScale(points=points, closed=False, color=Color(128, 128, 128))

        # Latitude scale
        elif i >= self.MR_1 and i < self.MR_1+self.MR_2:
            points = np.zeros((self.MR_1, 3))
            for u in range(self.MR_1):
                points[u, 0] = self.snakeContour[u][i-self.MR_1].x
                points[u, 1] = self.snakeContour[u][i-self.MR_1].y
                points[u, 2] = self.snakeContour[u][i-self.MR_1].z
                
            scale = Snake3DScale(points=points, closed=True, color=Color(0, 0, 204))
        return scale

    """ Return list of scales """
    def getScales(self):
        scales = list()
        for i in range(self.getNumScales()):
            scales.append(self._getScale(i))
        return scales
