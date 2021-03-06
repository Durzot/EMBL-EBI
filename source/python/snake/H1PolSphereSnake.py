# -*- coding: utf-8 -*-
"""
This class implements extension of Uhlmann's et al paper to sphere-like surfaces
that is open lines on longitudes and closed lines on latitudes.

Designed to be run in Python 3 virtual environment 3.7_vtk

Three-dimensional Hermite polynomial spline snake

@version: July 16, 2019
@author: Yoann Pradat
"""

import numpy as np
from auxiliary.aux import Color, Point3D
from process.SurfaceOracles import *
from process.BSpline import SlopeCSNotAKnot
from snake3D.Snake3DNode import Snake3DNode
from snake3D.Snake3DScale import Snake3DScale
from scipy import integrate

class H1PolSphereSnake(object):
    def __init__(self, M_1, M_2, nSamplesPerSeg=11, hidePoints=True):
        # Number of control points on latitudes
        self.M_1 = M_1

        # Number of control points on longitudes
        self.M_2 = M_2
            
        # Longitudes are open curves in the model
        # However in case c[k, 0] = c[k, M_2] then the longitude will be closed
        self.lon_closed=False
        self.lat_closed=True

        # Number of samples between consecutive control points 
        # for drawing sking of the snake
        self.nSamplesPerSeg = nSamplesPerSeg

        # Number of points on latitudes for scale (closed curve)
        self.MR_1 = nSamplesPerSeg*M_1

        # Number of points on longitudes for scale (open curve)
        self.MR_2 = nSamplesPerSeg*M_2

        # Number of coefficients of spline representation
        # i=1,...,4  
        # c_i[k,l] for k=0,..., M_1-1 and l=0,..., M_2
        self.nCoefs = 4*M_1*(M_2+1)
        self.coefs = [Snake3DNode(0, 0, 0, hidden=hidePoints) for _ in range(self.nCoefs)]
        
        # Samples of the coordinates of the snake contour 
        self.snakeContour = [[Point3D(0, 0, 0) for _ in range(self.MR_2+1)] for _ in range(self.MR_1)]

        self.hidePoints = hidePoints

    def _surfaceValue(self, u, v):
        assert 0 <= u and u <= 1
        assert 0 <= v and v <= 1

        # The calculation assumes a regular grid
        surfaceValue = Point3D(0, 0, 0)
        for k in range(self.M_1):
            for l in range(self.M_2+1):
                phi_1_M1_per = self._phi_1(self.M_1*u-k) + self._phi_1(self.M_1*(u-1)-k)
                phi_2_M1_per = self._phi_2(self.M_1*u-k) + self._phi_2(self.M_1*(u-1)-k)
                phi_1_M2 = self._phi_1(self.M_2*v-l)
                phi_2_M2 = self._phi_2(self.M_2*v-l)

                c_1 = self.coefs[k + l*self.M_1] 
                c_2 = self.coefs[k + l*self.M_1 + self.M_1*(self.M_2+1)] 
                c_3 = self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)] 
                c_4 = self.coefs[k + l*self.M_1 + 3*self.M_1*(self.M_2+1)] 
                surfaceValue += c_1*phi_1_M1_per*phi_1_M2
                surfaceValue += c_2*phi_1_M1_per*phi_2_M2
                surfaceValue += c_3*phi_2_M1_per*phi_1_M2 
                surfaceValue += c_4*phi_2_M1_per*phi_2_M2

        return surfaceValue

    def _updateContour(self):
        for k in range(self.MR_1):
            for l in range(self.MR_2+1):
                surfaceValue = self._surfaceValue(k/self.MR_1, l/self.MR_2)
                surfaceValue.clip()
                self.snakeContour[k][l].updateFromPoint(surfaceValue)


    def _Df_1(self, x, order=1):
        assert 0 <= x
        if 1 < x:
            return 0
        else:
            if order==0:
                return 1 - 3*x**2 + 2*x**3
            elif order==1:
                return - 6*x + 6*x**2
            else:
                return -6 + 12*x
    
    def _phi_1(self, x):
        if 1 <= np.abs(x):
            return 0
        else:
            return self._Df_1(np.abs(x), order=0)

    def _Df_2(self, x, order=1):
        assert 0 <= x
        if 1 < x:
            return 0
        else:
            if order==0:
                return x - 2*x**2 + x**3             
            elif order==1:
                return 1 - 4*x + 3*x**2
            elif order==2:
                return -4 + 6*x

    def _phi_2(self, x):
        if 1 <= np.abs(x):
            return 0
        else:
            if 0 <= x:
                return self._Df_2(x, order=0)
            else:
                return -self._Df_2(-x, order=0)

    def initializeDefaultShape(self, shape):
        self.shape = shape
        if shape=='sphere':
            # Initialize snake coefs for a sphere shape
            oracle = OracleSphere(R=1)
        elif shape=='poly_sphere':
            # Initialize snake coefs for a poly sphere shape
            oracle = OraclePolySphere(R=1, M_1=self.M_1, M_2=self.M_2)
        elif shape=='half-sphere':
            # Initialize snake coefs for a half sphere shape
            oracle = OracleHalfSphere(R=1)
        elif shape=='torus':
            # Initialize snake coefs for a torus shape
            oracle = OracleTorus(R=8, r=2)
            #self.w_1 = 2*np.pi/self.M_1
            #self.w_2 = 2*np.pi/self.M_2
        elif shape=='klein_bottle':
            # Initialize snake coefs for Klein bottle
            oracle = OracleKleinBottle()
        elif shape=='klein_8':
            # Initialize snake coefs for Klein bottle
            oracle = OracleKlein8(a=4)
        elif shape=='x':
            oracle = OracleX(R=4)
        else:
            raise ValueError("Choose between 'sphere', 'half-sphere' and 'torus'")
        self.oracle = oracle

        # Coefficients c_i[k,l] k=0,.., M_1-1, l=0,.., M_2
        # i=1,..,4
        for l in range(0, self.M_2+1):
            for k in range(self.M_1):
                # Coefficients c_1. Surface values
                value = oracle.value(k/self.M_1, l/self.M_2)
                self.coefs[k + l*self.M_1].updateFromPoint(value)
                
                # Coefficients c_2. Surface v-derivatives
                v_deriv = 1/(1.*self.M_2)*oracle.v_deriv(k/self.M_1,l/self.M_2)
                self.coefs[k + l*self.M_1 + self.M_1*(self.M_2+1)].updateFromPoint(v_deriv)

                # Coefficients c_3. Surface u-derivatives
                u_deriv = 1/(1.*self.M_1)*oracle.u_deriv(k/self.M_1,l/self.M_2)
                self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)].updateFromPoint(u_deriv)

                # Coefficients c_4. Surface uv-derivatives or "twist vector"
                uv_deriv = 1/(1.*self.M_1*self.M_2)*oracle.uv_deriv(k/self.M_1,l/self.M_2)
                self.coefs[k + l*self.M_1 + 3*self.M_1*(self.M_2+1)].updateFromPoint(uv_deriv)
        
        self._updateContour()

    def getNumScales(self):
        return self.MR_1 + self.MR_2 + 1

    """ Returns the ith scale """
    def _getScale(self, i):
        # Longitude scale
        if i >= 0 and i < self.MR_1:
            points = np.zeros((self.MR_2+1, 3))
            for v in range(self.MR_2+1):
                points[v, 0] = self.snakeContour[i][v].x
                points[v, 1] = self.snakeContour[i][v].y
                points[v, 2] = self.snakeContour[i][v].z     
            if i == 0:
                scale = Snake3DScale(points=points, closed=self.lon_closed, color=Color(0, 0, 255))
            else:
                scale = Snake3DScale(points=points, closed=self.lon_closed, color=Color(128, 128, 128))

        # Latitude scale
        elif i >= self.MR_1 and i < self.MR_1+self.MR_2+1:
            points = np.zeros((self.MR_1, 3))
            for u in range(self.MR_1):
                points[u, 0] = self.snakeContour[u][i-self.MR_1].x
                points[u, 1] = self.snakeContour[u][i-self.MR_1].y
                points[u, 2] = self.snakeContour[u][i-self.MR_1].z
            scale = Snake3DScale(points=points, closed=self.lat_closed, color=Color(220, 20, 60))
        return scale

    """ Return list of nodes """
    def getNodes(self):
        nodes = self.coefs[:self.M_1*(self.M_2+1)]
        return nodes

    """ Return list of scales """
    def getScales(self):
        scales = list()
        for i in range(self.getNumScales()):
            scales.append(self._getScale(i))
        return scales

    def setTwist(self, method):
        if method=='null':
            for l in range(0, self.M_2+1):
                for k in range(self.M_1):
                    self.coefs[k + l*self.M_1 + 3*self.M_1*(self.M_2+1)].updateFromCoords(0, 0, 0)
        elif method=='random':
            mu = 0
            sigma = 0.2
            for l in range(0, self.M_2+1):
                for k in range(self.M_1):
                    x = np.random.randn()*sigma + mu
                    y = np.random.randn()*sigma + mu
                    z = np.random.randn()*sigma + mu
                    self.coefs[k + l*self.M_1 + 3*self.M_1*(self.M_2+1)] += Snake3DNode(x, y, z)
        else:
            raise ValueError("Unrecognized value for argument 'method'. Choose between 'null' and 'random'")

        self._updateContour()

    def estimateTwist(self, method='naive'):
        if method=='naive':
            # Estimate twist vector by interpolating coordinates of derivatives at control points
            
            # Derivative wtr to 2 of each coord of sigma_1
            sigma_12 = [Point3D(0,0,0) for _ in range(self.M_1*(self.M_2+1))]
            sites = np.array([l/self.M_2 for l in range(self.M_2+1)]) 
            for k in range(self.M_1):
                valsx = np.array([self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)].x for l in range(self.M_2+1)])
                valsy = np.array([self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)].y for l in range(self.M_2+1)])
                valsz = np.array([self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)].z for l in range(self.M_2+1)])
                
                # Retrieve slopes of not-a-knot cubic spline 
                slopex = SlopeCSNotAKnot(sites, valsx)
                slopey = SlopeCSNotAKnot(sites, valsy)
                slopez = SlopeCSNotAKnot(sites, valsz)

                for l in range(self.M_2+1):
                    sigma_12[k + l*self.M_1].updateFromCoords(slopex[l]/self.M_2, slopey[l]/self.M_2, slopez[l]/self.M_2)

            # Derivative wtr to 1 of each coord of sigma_2
            sigma_21 = [Point3D(0,0,0) for _ in range(self.M_1*(self.M_2+1))]
            sites = np.array([k/self.M_1 for k in range(self.M_1)]) 
            for l in range(self.M_2+1):
                valsx = np.array([self.coefs[k + l*self.M_1 + 1*self.M_1*(self.M_2+1)].x for k in range(self.M_1)])
                valsy = np.array([self.coefs[k + l*self.M_1 + 1*self.M_1*(self.M_2+1)].y for k in range(self.M_1)])
                valsz = np.array([self.coefs[k + l*self.M_1 + 1*self.M_1*(self.M_2+1)].z for k in range(self.M_1)])
                
                # Retrieve slopes of not-a-knot cubic spline 
                slopex = SlopeCSNotAKnot(sites, valsx)
                slopey = SlopeCSNotAKnot(sites, valsy)
                slopez = SlopeCSNotAKnot(sites, valsz)

                for k in range(self.M_1):
                    sigma_21[k + l*self.M_1].updateFromCoords(slopex[k]/self.M_1, slopey[k]/self.M_1, slopez[k]/self.M_1)

            error = 0
            for k in range(self.M_1):
                for l in range(self.M_2+1):
                    twist_hat = sigma_12[k + l*self.M_1]                  
                    twist = self.coefs[k + l*self.M_1 + 3*self.M_1*(self.M_2+1)]
                    print("At u=%.3g, v=%.3g estimated %s vs true %s" % (k/self.M_1, l/self.M_2, twist_hat, twist))
                    error += (twist-twist_hat).norm()
            print("Average error %.3g" % (error/(self.M_1*(self.M_2+1))))

        elif method=='oscillation':
            # The following estimation is an implementation of minimum oscillation principle introduced by Guo et Han to 
            # determine the twist components in the article "A New Computational Approach to the Twists of Bicubic Coons 
            # Surfaces" of 2017
            
            # The system slightly differs from that of the article because of the
            # periodicity on latitudes. All details can be found in my note on twist vectors.

            # The matrix equation is (K \bigotimeplus L) R = C with unknown R while K and L depend on the knot
            # locations and C depends in addition to that on surface values and first-order derivatives at knots

            K = np.zeros((self.M_2+1, self.M_2+1))
            L = np.zeros((self.M_1, self.M_1))
            C = np.zeros(self.M_1*(self.M_2+1))

            # As the spacing is uniform in our scheme, the formulas for K and L are greatly simplified
            for i in range(1, self.M_2):
                K[i,i-1] = -3./self.M_2**3
                K[i,i] = 8./self.M_2**3
                K[i,i+1] = -3./self.M_2**3

            for i in range(self.M_1):
                L[i,i-1] = -3./self.M_1**3
                L[i,i] = 8./self.M_1**3
                L[i,(i+1)%self.M_1] = -3./self.M_1**3

            K[0,0] = 4./self.M_2**3
            K[0,1] = -3/self.M_2**3
            K[self.M_2,self.M_2-1] = -3/self.M_2**3
            K[self.M_2,self.M_2] = 4./self.M_2**3

            # Compute kronecker product to create M_1*(M_2+1) matrix
            M = np.kron(K, L)

            # In order to compute C we need to compute B_{i,j} and G_0, ..., G_3 as defined in my notes
            # List of integral value of product g_0/1 * f_0/1 or g_0/1
            # Order is g_0^h*[f_0^h, f_1^h, g_0^h, g_1^h] then g_1^h*[f_0^h, f_1^h, g_0^h, g_1^h]
            I = np.empty(2, dtype=object)
            I[0] = np.array([[11/120, 13/420, 1/105, -1/140]])
            I[1] = np.array([[-13/240, -11/120, -1/140, 1/105]])
            
            G = np.empty(4, dtype=object)
            G[0] = 176400*I[0].T.dot(I[0])
            G[1] = 176400*I[1].T.dot(I[0])
            G[2] = 176400*I[0].T.dot(I[1])
            G[3] = 176400*I[1].T.dot(I[1])

            # Don't forget normalization factor appearing in g_0, g_1
            for q in range(len(G)):
                for i in range(4):
                    for j in range(4):
                        if i >= 2:
                            G[q][i,j] /= self.M_1
                        if j >= 2:
                            G[q][i,j] /= self.M_2

            # Link coefficients c_1, c_2, c_3 to quantities p, g, f
            p = np.empty((self.M_1+1, self.M_2+1), dtype=Point3D)
            g = np.empty((self.M_1+1, self.M_2+1), dtype=Point3D)
            f = np.empty((self.M_1+1, self.M_2+1), dtype=Point3D)
            r = np.empty((self.M_1+1, self.M_2+1), dtype=Point3D)
             
            for i in range(self.M_1+1):
                for j in range(self.M_2+1):
                    # p[i,j] = c_1[i,j]
                    p[i,j] = self.coefs[i%self.M_1 + j*self.M_1]
                    # g[i,j] = c_2[i,j]/l_j
                    g[i,j] = self.coefs[i%self.M_1 + j*self.M_1 + self.M_1*(self.M_2+1)]*self.M_2
                    # f[i,j] = c_3[i,j]/h_i
                    f[i,j] = self.coefs[i%self.M_1 + j*self.M_1 + 2*self.M_1*(self.M_2+1)]*self.M_1
                    # r[i,j] = c_3[i,j]/(h_i l_j)
                    r[i,j] = self.coefs[i%self.M_1 + j*self.M_1 + 3*self.M_1*(self.M_2+1)]*self.M_1*self.M_2

            # Compute quantities defining matrices B_{i,j} as defined in Guo et Han article except for 
            # \tilde{r} which carries a mistake
            bar_f = np.empty((self.M_1+1, self.M_2+1), dtype=Point3D)
            bar_g = np.empty((self.M_1+1, self.M_2+1), dtype=Point3D)
            bar_r = np.empty((self.M_1+1, self.M_2+1), dtype=Point3D)

            # Compute matrices B_{i,j} i = 0, ..., M_1-1, j=0, ..., M_2-1
            B = np.empty((self.M_1,self.M_2), dtype=object)
            for i in range(self.M_1):
                for j in range(self.M_2):
                    # Be aware of the successive rewriting of values
                    # in the matrices bar_f, bar_g, bar_r
            
                    bar_f[i,j] = (p[i+1,j]-p[i,j])*self.M_1
                    bar_f[i+1,j] = (p[i+1,j]-p[i,j])*self.M_1
                    bar_f[i,j+1] = (p[i+1,j+1]-p[i,j+1])*self.M_1
                    bar_f[i+1,j+1] = (p[i+1,j+1]-p[i,j+1])*self.M_1
            
                    bar_g[i,j] = (p[i,j+1]-p[i,j])*self.M_2
                    bar_g[i,j+1] = (p[i,j+1]-p[i,j])*self.M_2
                    bar_g[i+1,j] = (p[i+1,j+1]-p[i+1,j])*self.M_2
                    bar_g[i+1,j+1] = (p[i+1,j+1]-p[i+1,j])*self.M_2
            
                    qty = (p[i,j] - p[i,j+1] - p[i+1,j] + p[i+1,j+1])*self.M_1*self.M_2
                    bar_r[i,j] = qty
                    bar_r[i+1,j] = qty
                    bar_r[i,j+1] = qty
                    bar_r[i+1,j+1] = qty
                    
                    B[i,j] = np.empty((4,4), dtype=Point3D)
                    for s in range(4):
                        for t in range(4):
                            if s >= 2 and t >= 2:
                                B[i,j][s,t] = -bar_r[i+s-2,j+t-2]
                            elif s >= 2:
                                B[i,j][s,t] = f[i+s-2,j+t] - bar_f[i+s-2,j+t]
                            elif t >= 2:
                                B[i,j][s,t] = g[i+s,j+t-2] - bar_g[i+s,j+t-2]
                            else:
                                B[i,j][s,t] = Point3D(0,0,0)

            # The energy on patch I_{i,j} is E_{i,j} = h_i l_j \int \int Tr(H(u,v) B_tilde{i,j})
            # Fu = [f_0(u), f_1(u), g_0(u), g_1(u)]
            # Fv = [f_0(v), f_1(v), g_0(v), g_1(v)]
            Fu = lambda u: np.array([[1-3*u**2 + 2*u**3], 
                                    [3*u**2 - 2*u**3], 
                                    [u*(u-1)**2/self.M_1],
                                    [u**2*(u-1)/self.M_1]])
            Fv = lambda v: np.array([[1-3*v**2 + 2*v**3], 
                                    [3*v**2 - 2*v**3], 
                                    [v*(v-1)**2/self.M_2],
                                    [v**2*(v-1)/self.M_2]])
            H = lambda u,v: Fu(u).dot(Fv(v).T)

            # Compute energy on each patch
            B_tilde = np.copy(B)
            E = np.zeros((self.M_1,self.M_2))
            for i in range(self.M_1):
                for j in range(self.M_2):
                    for s in range(2,4):
                        for t in range(2,4):
                            B_tilde[i,j][s,t] += r[i+s-2,j+t-2]
                    
                    func = lambda u,v : np.trace(H(u,v).dot(B_tilde[i,j])).norm()**2
                    E[i,j], _ = integrate.dblquad(func, 0, 1, lambda x: 0, lambda x: 1)
                    E[i,j] /= self.M_1*self.M_2
            print("Energy with defaut twist is %.3g" % (E.sum().sum()))

            # Compute long column vector C
            C = np.empty(self.M_1*(self.M_2+1), dtype=Point3D)
            for i in range(self.M_1):
                # First M_1 elements of C
                C[i] = (np.trace(G[1].dot(B[i-1, 0])) + np.trace(G[0].dot(B[i,0])))/(self.M_1**2*self.M_2**2)

                # Last M_1 elements of C
                C[i+self.M_1*self.M_2] = (np.trace(G[3].dot(B[i-1, self.M_2-1])) \
                                          + np.trace(G[2].dot(B[i,self.M_2-1])))/(self.M_1**2*self.M_2**2)
                
                # All in-between elements (4 patches)
                for j in range(1, self.M_2):
                    C[i+self.M_1*j] = (np.trace(G[3].dot(B[i-1, j-1])) \
                                      + np.trace(G[2].dot(B[i,j-1])) \
                                      + np.trace(G[1].dot(B[i-1,j])) \
                                      + np.trace(G[0].dot(B[i,j])))/(self.M_1**2*self.M_2**2)

            # Inverse the system
            R = np.linalg.inv(M).dot(C)

            for j in range(self.M_2+1):
                for i in range(self.M_1):
                    print("At i=%d, j=%d" % (i,j))
                    print("optimal twist %s" % (R[i+j*self.M_1]/(self.M_1*self.M_2)))
                    print("real twist %s" % self.coefs[i + j*self.M_1 + 3*self.M_1*(self.M_2+1)])
                    self.coefs[i + j*self.M_1 + 3*self.M_1*(self.M_2+1)] = R[i+j*self.M_1]/(self.M_1*self.M_2)

            B_tilde = np.copy(B)
            E = np.zeros((self.M_1,self.M_2))
            for i in range(self.M_1):
                for j in range(self.M_2):
                    for s in range(2,4):
                        for t in range(2,4):
                            B_tilde[i,j][s,t] += r[i+s-2,j+t-2]
                    
                    func = lambda u,v : np.trace(H(u,v).dot(B_tilde[i,j])).norm()**2
                    E[i,j], _ = integrate.dblquad(func, 0, 1, lambda x: 0, lambda x: 1)
                    E[i,j] /= self.M_1*self.M_2
            print("Energy with optimized twist is %.3g" % (E.sum().sum()))

            for i in range(self.M_1+1):
                for j in range(self.M_2+1):
                    # r[i,j] = c_3[i,j]/(h_i l_j)
                    r[i,j] = self.coefs[i%self.M_1 + j*self.M_1 + 3*self.M_1*(self.M_2+1)]*self.M_1*self.M_2

            self._updateContour()

        elif method=='selesnick':
            # The following estimation of the twist vector is an implementation of what S.A Selesnick 
            # describes in his article "Local invariants and twist vectors in computer-aided geometric design" of 1981
            #
            # The idea is to estimate first the normal component n.sigma_12 of the twist vector from knowledge of 
            # the Gaussian curvature. Then the tangential components are estimated by estimating the variation 
            # of the length of the tangential vector sigma_1 as the second parameter changes and vice-versa

            def F(u):
                return np.array([self._Df_1(u, 2), self._Df_1(1-u, 2), 
                                 self._Df_2(u, 2), -self._Df_2(1-u, 2)])

            # Estimate derivatives of |sigma_1| with respect to var 2 and |sigma_2| with respect ot var 1 
            # at control points. This is done through cubic spline interpolation with not-a-knot condition. 
            # Refer to chapter IV of A Practical Guide To Splines by De Boor for more details on this interpolation.
            
            # Estimation of derivative wrt to 2 of |sigma_1| * 1/(M_1*M_2)
            dnorm_sigma_1 = [0 for _ in range(self.M_1*(self.M_2+1))]

            #  Estimation of derivative wrt to 1 of |sigma_2| * 1/(M_1*M_2)
            dnorm_sigma_2 = [0 for _ in range(self.M_1*(self.M_2+1))]

            # Derivatives of |sigma_1|
            for k in range(self.M_1):
                tau = np.array([l/self.M_2 for l in range(self.M_2+1)]) # Independent of k
                gtau = np.array([self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)].norm() for l in range(self.M_2+1)])
                
                # Retrieve slopes of not-a-knot cubic spline 
                slope = SlopeCSNotAKnot(tau, gtau)
                for l in range(self.M_2+1):
                    dnorm_sigma_1[k + l*self.M_1] = slope[l]/self.M_2
             
            # Derivatives of |sigma_2|
            for l in range(self.M_2+1):
                # At poles assign to 0
                if l==0 or l==self.M_2:
                    for k in range(self.M_1):
                        dnorm_sigma_2[k + l*self.M_1] = 0

                else:
                    tau = np.array([l/self.M_2 for l in range(self.M_2+1)]) # Independent of k
                    gtau = np.array([self.coefs[k + l*self.M_1 + self.M_1*(self.M_2+1)].norm() for l in range(self.M_2+1)])
                
                    # Retrieve slopes of not-a-knot cubic spline 
                    slope = SlopeCSNotAKnot(tau, gtau)
                    for k in range(self.M_1):
                        dnorm_sigma_2[k + l*self.M_1] = slope[k]/self.M_1

            
            for k in range(self.M_1):
                for l in range(self.M_2):
                    # We are working patch by patch independently
                    # u=0,1 and v=0,1 correspond to the corners of the patch [k/M_1, k+1/M_1]x[l/M_2, l+1/M_2]
                    # We have M_1 coefs in u, M_2+1 coefs in v
                    # In case k = M_1-1, c[k+1 = M_1, l] = c[0, l] by periodicity

                    # Surface values 
                    sigma = np.empty((2,2), dtype=object)
                    sigma[0,0] = self.coefs[k + l*self.M_1]
                    sigma[0,1] = self.coefs[k + (l+1)*self.M_1]
                    sigma[1,0] = self.coefs[(k+1)%self.M_1 + l*self.M_1]
                    sigma[1,1] = self.coefs[(k+1)%self.M_1 + (l+1)*self.M_1]
                
                    # First order u-derivatives
                    sigma_1 = np.empty((2,2), dtype=object)
                    sigma_1[0,0] = self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)]
                    sigma_1[0,1] = self.coefs[k + (l+1)*self.M_1 + 2*self.M_1*(self.M_2+1)]
                    sigma_1[1,0] = self.coefs[(k+1)%self.M_1 + l*self.M_1 + 2*self.M_1*(self.M_2+1)]
                    sigma_1[1,1] = self.coefs[(k+1)%self.M_1 + (l+1)*self.M_1 + 2*self.M_1*(self.M_2+1)]

                    # First order v-derivatives
                    sigma_2 = np.empty((2,2), dtype=object)
                    sigma_2[0,0] = self.coefs[k + l*self.M_1 + self.M_1*(self.M_2+1)]
                    sigma_2[0,1] = self.coefs[k + (l+1)*self.M_1 + self.M_1*(self.M_2+1)]
                    sigma_2[1,0] = self.coefs[(k+1)%self.M_1 + l*self.M_1 + self.M_1*(self.M_2+1)]
                    sigma_2[1,1] = self.coefs[(k+1)%self.M_1 + (l+1)*self.M_1 + self.M_1*(self.M_2+1)]

                    # First and second columns of the Q matrix
                    Qc1 = np.array([sigma[0,0], sigma[1,0], sigma_1[0,0], sigma_1[1,0]])
                    Qc2 = np.array([sigma[0,1], sigma[1,1], sigma_1[0,1], sigma_1[1,1]])

                    # Compute 2nd-order u-deriv at the corners of the patch 
                    sigma_11 = np.empty((2,2), dtype=object)
                    sigma_11[0,0] = F(0).dot(Qc1)
                    sigma_11[0,1] = F(0).dot(Qc2)
                    sigma_11[1,0] = F(1).dot(Qc1)
                    sigma_11[1,1] = F(1).dot(Qc2)
                    
                    for u in [0,1]:
                        for v in [0,1]:
                            uc = (k+u)%self.M_1/self.M_1
                            vc = (l+v)/self.M_2
                            sigma_11_hat = sigma_11[u,v]
                            sigma_11_true = self.oracle.uu_deriv(uc,vc)/self.M_1**2
                            print("At u=%.3g, v=%.3g est. sigma_11 %s vs true sigma_11 %s" % (uc, vc, sigma_11_hat,
                                                                                              sigma_11_true))
                    # First and second rows of the Q matrix
                    Qr1 = np.array([sigma[0,0], sigma[0,1], sigma_2[0,0], sigma_2[0,1]])
                    Qr2 = np.array([sigma[1,0], sigma[1,1], sigma_2[1,0], sigma_2[1,1]])

                    # Compute 2nd-order v-deriv at the corners of the patch 
                    sigma_22 = np.empty((2,2), dtype=object)
                    sigma_22[0,0] = Qr1.dot(F(0))
                    sigma_22[0,1] = Qr1.dot(F(1)) 
                    sigma_22[1,0] = Qr2.dot(F(0))
                    sigma_22[1,1] = Qr2.dot(F(1))

                    for u in [0,1]:
                        for v in [0,1]:
                            uc = (k+u)%self.M_1/self.M_1
                            vc = (l+v)/self.M_2
                            sigma_22_hat = sigma_22[u,v]
                            sigma_22_true = self.oracle.vv_deriv(uc,vc)/self.M_2**2
                            print("At u=%.3g, v=%.3g est. sigma_22 %s vs true sigma_22 %s" % (uc, vc, sigma_22_hat,
                                                                                              sigma_22_true))

                    # Normal vector
                    n_hat = np.empty((2,2), dtype=object)
                    for u in [0,1]:
                        for v in [0,1]:
                            n_hat[u,v] = sigma_1[u,v].crossProduct(sigma_2[u,v])
                            if np.abs(n_hat[u,v].norm()) < 1e-10:
                                print("Issue at  u=%.3g, v=%.3g" % ((k+u)%self.M_1/self.M_1,(l+v)/self.M_2))
                                print("The cross-product between sigma_1 and sigma_2 is 0")
                            else:
                                n_hat[u,v] /= n_hat[u,v].norm()

                    # Matrice of twist vector coordinates at each of the 4 corners in 
                    # the tangential basis (n, r_1, r_2). In order (0,0), (0,1), (1,0), (1,1)
                    sigma_12 = np.zeros((3,4), dtype=float)

                    j = 0
                    for u in [0,1]:
                        for v in [0,1]:
                            uc = (k+u)%self.M_1/self.M_1
                            vc = (l+v)/self.M_2

                            # The gaussian curvature
                            K = self.oracle.gauss_curv(uc, vc)
                            H_2 = sigma_1[u,v].norm()**2*sigma_2[u,v].norm()**2 - sigma_1[u,v].dot(sigma_2[u,v])

                            if H_2==0:
                                print("Issue at  u=%.3g, v=%.3g" % (uc, vc))
                                print("The formula cannot be used as det of 1st fund. form is 0")

                            else:
                                nhat_t_sq = (n_hat[u,v].dot(sigma_11[u,v]))*(n_hat[u,v].dot(sigma_22[u,v]))-K*H_2
                            
                                # In case the number below the sqrt is negative, the Gaussian curvature
                                # given is invalid. We choose to change it continuously until the number becomes positive 
                                # that is 0.
                                if np.abs(nhat_t_sq) < 1e-10:
                                    nhat_t_sq = 0
                                elif nhat_t_sq < 0:
                                    print("Issue at  u=%.3g, v=%.3g" % (uc, vc))
                                    print ("The squared value of n.sigma_12 is negative: %.3g" % (nhat_t_sq))
                                    nhat_t_sq = 0

                                # Normal component
                                # Choose positive sign at north pole and then alternating sign at patch ends
                                sigma_12[0, j] = (-1)**(k+l)*np.sqrt(nhat_t_sq)

                                # First (normalized) tangential component of twist vector
                                sigma_12[1, j] = dnorm_sigma_1[(k+u)%self.M_1 + (l+v)*self.M_1]

                                # Second (normalized) tangential component of the twist vector
                                sigma_12[2, j]  = dnorm_sigma_2[(k+u)%self.M_1 + (l+v)*self.M_1]

                                # Use the matrix to change from basis (n_hat, r_1_hat, r_2_hat) 
                                # to canonical base (e_1, e_2, e_3)
                                sc_1 = sigma_1[u,v].norm()
                                sc_2 = sigma_2[u,v].norm()
                                
                                P = np.array([[n_hat[u,v].x, n_hat[u,v].y, n_hat[u,v].z],
                                              [sigma_1[u,v].x/sc_1, sigma_1[u,v].y/sc_1, sigma_1[u,v].z/sc_1],
                                              [sigma_2[u,v].x/sc_2, sigma_2[u,v].y/sc_2, sigma_2[u,v].z/sc_2]]).T
                                sigma_12[:,j] = P.dot(sigma_12[:,j])

                                # Update the coefficients c_4
                                # Given shared edges between patches each coefficient will be updated up to 4 times

                                twist_hat = Point3D(sigma_12[0,j], sigma_12[1,j], sigma_12[2,j])
                                twist_hat.clip()
                                twist = self.coefs[(k+u)%self.M_1 + (l+v)*self.M_1 + 3*self.M_1*(self.M_2+1)]
                                print("At u=%.3g, v=%.3g est. sigma_12 %s vs true sigma_12 %s" % (uc, vc, twist_hat, twist))
                                
                                j += 1
                            
        else:
            raise ValueError("Unrecognized value for argument 'method'. Please choose 'naive', 'oscillation', and \
                             'selesnick' for the method")

