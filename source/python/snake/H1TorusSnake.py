# -*- coding: utf-8 -*-
"""
This class implements extension of Conti's et al paper to Torus-like surfaces
that is closed lines on longitudes and latitudes.

Designed to be run in Python 3 virtual environment 3.7_vtk

Three-dimensional exponential spline snake

@version: May 28, 2019
@author: Yoann Pradat
"""

import numpy as np
from auxiliary.aux import Color, Point3D
from process.SurfaceOracles import *
from snake3D.Snake3DNode import Snake3DNode
from snake3D.Snake3DScale import Snake3DScale

class H1TorusSnake(object):
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

        # Number of points on longitudes for scale (closed curve)
        self.MR_2 = nSamplesPerSeg*M_2

        # Number of coefficients of spline representation
        # i=1,...,4  
        # c_i[k,l] for k=0,..., M_1-1 and l=0,..., M_2
        self.nCoefs = 4*M_1*M_2
        self.coefs = [Snake3DNode(0, 0, 0) for _ in range(self.nCoefs)]
        
        # Samples of the coordinates of the snake contour 
        self.snakeContour = [[Point3D(0, 0, 0) for _ in range(self.MR_2)] for _ in range(self.MR_1)]

    def _surfaceValue(self, u, v):
        assert 0 <= u and u <= 1
        assert 0 <= v and v <= 1

        # The calculation assumes a regular grid
        surfaceValue = Snake3DNode(0, 0, 0)
        for k in range(self.M_1):
            for l in range(self.M_2):
                phi_1_w1_per = self._phi_1(self.w_1, self.M_1*u-k) + self._phi_1(self.w_1, self.M_1*(u-1)-k)
                phi_2_w1_per = self._phi_2(self.w_1, self.M_1*u-k) + self._phi_2(self.w_1, self.M_1*(u-1)-k)
                phi_1_w2_per = self._phi_1(self.w_2, self.M_2*v-l) + self._phi_1(self.w_2, self.M_2*(v-1)-l)
                phi_2_w2_per = self._phi_2(self.w_2, self.M_2*v-l) + self._phi_2(self.w_2, self.M_2*(v-1)-l)

                c_1 = self.coefs[k + l*self.M_1] 
                c_2 = self.coefs[k + l*self.M_1 + self.M_1*self.M_2] 
                c_3 = self.coefs[k + l*self.M_1 + 2*self.M_1*self.M_2] 
                c_4 = self.coefs[k + l*self.M_1 + 3*self.M_1*self.M_2] 
                surfaceValue += c_1*phi_1_w1_per*phi_1_w2_per
                surfaceValue += c_2*phi_1_w1_per*phi_2_w2_per
                surfaceValue += c_3*phi_2_w1_per*phi_1_w2_per
                surfaceValue += c_4*phi_2_w1_per*phi_2_w2_per

        #if abs(surfaceValue.x - np.cos(2*np.pi*u)*np.sin(np.pi*v)) > 0.1:
        #    print("error on x for u=%.2f and v=%.2f" % (u, v))
        #if abs(surfaceValue.y - np.sin(2*np.pi*u)*np.sin(np.pi*v)) > 0.1:
        #    print("error on y for u=%.2f and v=%.2f" % (u, v))
        #if abs(surfaceValue.z - np.cos(np.pi*v)) > 0.1:
        #    print("error on z for u=%.2f and v=%.2f" % (u, v))

        return surfaceValue

    def _updateContour(self):
        for k in range(self.MR_1):
            for l in range(self.MR_2):
                surfaceValue = self._surfaceValue(k/self.MR_1, l/self.MR_2)
                
                if abs(surfaceValue.x) < 1e-10:
                    surfaceValue.x = 0
                if abs(surfaceValue.y) < 1e-10:
                    surfaceValue.y = 0
                if abs(surfaceValue.z) < 1e-10:
                    surfaceValue.z = 0
                self.snakeContour[k][l].updateFromPoint(surfaceValue)

    def _phi_1(self, w, x):
        if 1 <= np.abs(x):
            return 0
        else:
            qw = np.exp(w*1j)*(w*1j-2) + w*1j+2
            aw = (w*1j + 1 + np.exp(w*1j)*(w*1j-1))/qw
            bw = -w*1j*(np.exp(w*1j)+1)/qw
            cw = 1/qw
            dw = -np.exp(w*1j)/qw
            return (aw + bw*np.abs(x) + cw*np.exp(w*np.abs(x)*1j) + dw*np.exp(-w*np.abs(x)*1j)).real

    def _phi_2(self, w, x):
        if 1 <= np.abs(x):
            return 0
        else:
            qw = np.exp(w*1j)*(w*1j-2) + w*1j+2
            pw = np.exp(2*w*1j)*(w*1j-1) + w*1j+1
            aw = pw/(w*1j*(np.exp(w*1j)-1)*qw)
            bw = -(np.exp(w*1j) -1)/qw
            cw = (np.exp(w*1j)-w*1j-1)/(w*1j*(np.exp(w*1j)-1)*qw)
            dw = -np.exp(w*1j)*(np.exp(w*1j)*(w*1j-1)+1)/(w*1j*(np.exp(w*1j)-1)*qw)
            
            if 0 <= x:
                return (aw + bw*x + cw*np.exp(w*x*1j) + dw*np.exp(-w*x*1j)).real
            else:
                return (-aw + bw*x + -cw*np.exp(-w*x*1j) - dw*np.exp(w*x*1j)).real

    def initializeDefaultShape(self):
        # Initalize snake coefs for a torus shape
        oracle = OracleTorus(R=2, r=0.5)
        self.w_1 = 2*np.pi/self.M_1
        self.w_2 = 2*np.pi/self.M_2

        # Coefficients c_i[k,l] k=0,.., M_1-1, l=0,.., M_2
        # i=1,..,4
        for l in range(0, self.M_2):
            for k in range(self.M_1):
                # Coefficients c_1. Surface values
                value = oracle.value(k/self.M_1, l/self.M_2)
                self.coefs[k + l*self.M_1] = value 
                
                # Coefficients c_2. Surface v-derivatives
                v_deriv = 1/(1.*self.M_2)*oracle.v_deriv(k/self.M_1,l/self.M_2)
                self.coefs[k + l*self.M_1 + self.M_1*self.M_2] = v_deriv

                # Coefficients c_3. Surface u-derivatives
                u_deriv = 1/(1.*self.M_1)*oracle.u_deriv(k/self.M_1,l/self.M_2)
                self.coefs[k + l*self.M_1 + 2*self.M_1*self.M_2] = u_deriv

                # Coefficients c_4. Surface uv-derivatives or "twist vector"
                uv_deriv = 1/(1.*self.M_1*self.M_2)*oracle.uv_deriv(k/self.M_1,l/self.M_2)
                self.coefs[k + l*self.M_1 + 3*self.M_1*self.M_2] = uv_deriv
        
        self._updateContour()

    def setNullTwist(self):
        for l in range(0, self.M_2+1):
            for k in range(self.M_1):
                # Coefficients c_4. Surface uv-derivatives or "twist vector"
                self.coefs[k + l*self.M_1 + 3*self.M_1*self.M_2] = Snake3DNode(0, 0, 0)

        self._updateContour()

    def setRandomPertTwist(self, mu, sigma):
        for l in range(0, self.M_2):
            for k in range(self.M_1):
                # Coefficients c_4. Surface uv-derivatives or "twist vector"
                x = np.random.randn()*sigma + mu
                y = np.random.randn()*sigma + mu
                z = np.random.randn()*sigma + mu
                self.coefs[k + l*self.M_1 + 3*self.M_1*self.M_2] += Snake3DNode(x, y, z)

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
            scale = Snake3DScale(points=points, closed=True, color=Color(255, 128, 128))

        # Latitude scale
        elif i >= self.MR_1 and i < self.MR_1+self.MR_2:
            points = np.zeros((self.MR_1, 3))
            for u in range(self.MR_1):
                points[u, 0] = self.snakeContour[u][i-self.MR_1].x
                points[u, 1] = self.snakeContour[u][i-self.MR_1].y
                points[u, 2] = self.snakeContour[u][i-self.MR_1].z
                
            scale = Snake3DScale(points=points, closed=True, color=Color(128, 128, 128))
        return scale

    """ Return list of scales """
    def getScales(self):
        scales = list()
        for i in range(self.getNumScales()):
            scales.append(self._getScale(i))
        return scales


