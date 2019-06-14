# -*- coding: utf-8 -*-
"""
This class implements extension of Conti's et al paper to Sphere-like surfaces
that is open lines on longitudes and closed lines on latitudes.

Designed to be run in Python 3 virtual environment 3.7_vtk

Three-dimensional Hermite exponential spline snake

@version: June 10, 2019
@author: Yoann Pradat
"""

import numpy as np
from auxiliary.aux import Color, Point3D
from process.SurfaceOracles import *
from process.BSpline import SlopeCSNotAKnot
from snake3D.Snake3DNode import Snake3DNode
from snake3D.Snake3DScale import Snake3DScale

class H1ExpSphereSnake(object):
    def __init__(self, M_1, M_2, nSamplesPerSeg=11, hidePoints=True):
        # Number of control points on latitudes
        self.M_1 = M_1
        self.w_1 = 2*np.pi/self.M_1

        # Number of control points on longitudes
        self.M_2 = M_2
        self.w_2 = np.pi/self.M_2
            
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
                phi_1_w1_per = self._phi_1(self.w_1, self.M_1*u-k) + self._phi_1(self.w_1, self.M_1*(u-1)-k)
                phi_2_w1_per = self._phi_2(self.w_1, self.M_1*u-k) + self._phi_2(self.w_1, self.M_1*(u-1)-k)
                phi_1_w2 = self._phi_1(self.w_2, self.M_2*v-l)
                phi_2_w2 = self._phi_2(self.w_2, self.M_2*v-l)

                c_1 = self.coefs[k + l*self.M_1] 
                c_2 = self.coefs[k + l*self.M_1 + self.M_1*(self.M_2+1)] 
                c_3 = self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)] 
                c_4 = self.coefs[k + l*self.M_1 + 3*self.M_1*(self.M_2+1)] 
                surfaceValue += c_1*phi_1_w1_per*phi_1_w2
                surfaceValue += c_2*phi_1_w1_per*phi_2_w2
                surfaceValue += c_3*phi_2_w1_per*phi_1_w2 
                surfaceValue += c_4*phi_2_w1_per*phi_2_w2

        return surfaceValue

    def _updateContour(self):
        for k in range(self.MR_1):
            for l in range(self.MR_2+1):
                surfaceValue = self._surfaceValue(k/self.MR_1, l/self.MR_2)
                surfaceValue.clip()
                self.snakeContour[k][l].updateFromPoint(surfaceValue)

    def _phi_1(self, w, x):
        if 1 <= np.abs(x):
            return 0
        else:
            return self._Dg_1(w, np.abs(x), order=0)

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
            #points = np.zeros((self.MR_1+1, 3)) # For disentangling Klein bottle
            points = np.zeros((self.MR_1, 3))
            for u in range(self.MR_1):
                points[u, 0] = self.snakeContour[u][i-self.MR_1].x
                points[u, 1] = self.snakeContour[u][i-self.MR_1].y
                points[u, 2] = self.snakeContour[u][i-self.MR_1].z
            #points[self.MR_1, 0] = self.snakeContour[0][self.MR_2-i+self.MR_1].x
            #points[self.MR_1, 1] = self.snakeContour[0][self.MR_2-i+self.MR_1].y
            #points[self.MR_1, 2] = self.snakeContour[0][self.MR_2-i+self.MR_1].z    
            #scale = Snake3DScale(points=points, closed=True, color=Color(220, 20, 60))
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

    def setNullTwist(self):
        for l in range(0, self.M_2+1):
            for k in range(self.M_1):
                # Coefficients c_4. Surface uv-derivatives or "twist vector"
                self.coefs[k + l*self.M_1 + 3*self.M_1*(self.M_2+1)].updateFromCoords(0, 0, 0)

        self._updateContour()

    def setRandTwist(self, mu, sigma):
        for l in range(0, self.M_2+1):
            for k in range(self.M_1):
                # Coefficients c_4. Surface uv-derivatives or "twist vector"
                x = np.random.randn()*sigma + mu
                y = np.random.randn()*sigma + mu
                z = np.random.randn()*sigma + mu
                self.coefs[k + l*self.M_1 + 3*self.M_1*(self.M_2+1)] += Snake3DNode(x, y, z)

        self._updateContour()

    def estimateTwist(self, method='brutal'):
        if method=='brutal':
            # Estimate twist vector by interpolation coordinates of each derivatives
            # at control points
            
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

            for k in range(self.M_1):
                for l in range(self.M_2):
                    twist_hat = sigma_12[k + l*self.M_1]                  
                    twist = self.coefs[k + l*self.M_1 + 3*self.M_1*(self.M_2+1)]
                    print("At u=%.3g, v=%.3g estimated %s vs true %s" % (k/self.M_1, l/self.M_2, twist_hat, twist))


    #def estimateTwist(self, shape, method='Selesnick'):
    #    if method=='Selesnick':
    #        # The following estimation of the twist vector is an implementation of what S.A Selesnick 
    #        # describes in his article "Local invariants and twist vectors in computer-aided geometric design" of 1981
    #        #
    #        # The idea is to estimate first the normal component n.sigma_12 of the twist vector from knowledge of 
    #        # the Gaussian curvature. Then the tangential components are estimated by estimating the variation 
    #        # of the length of the tangential vector sigma_1 as the second parameter changes and vice-versa

    #        # Gaussian curvature
    #        if shape=='sphere':
    #            def GaussCurv(u,v):
    #                return 1/16
    #        elif shape=='torus':
    #            def GaussCurv(u,v):
    #                return np.cos(2*np.pi*v)/(2*(8+2*np.cos(2*np.pi*v)))

    #        # Estimate sigma_11 at control points
    #        sigma_11 = [Point3D(0,0,0) for _ in range(self.M_1*(self.M_2+1)]


    #        def F(u):
    #            return np.array([self._Dg_1(self.w_1, u, 2), self._Dg_1(self.w_1, 1-u, 2), 
    #                             self._Dg_2(self.w_1, u, 2), -self._Dg_2(self.w_1, 1-u, 2)])
    #        
    #        def G(v):
    #            return np.array([self._Dg_1(self.w_2, v, 2), self._Dg_1(self.w_2, 1-v, 2), 
    #                             self._Dg_2(self.w_2, v, 2), -self._Dg_2(self.w_2, 1-v, 2)])

    #        # Estimate derivatives of |sigma_1| with respect to var 2 and |sigma_2| with respect ot var 1 
    #        # at control points. This is done through cubic spline interpolation with not-a-knot condition. 
    #        # Refer to chapter IV of A Practical Guide To Splines by De Boor for more details on this interpolation.
    #        
    #        # Estimation of d|sigma_1(k/M_1, l/M_2)|/dv * 1/(M_1*M_2)
    #        dnorm_sigma_1 = [0 for _ in range(self.M_1*(self.M_2+1))]

    #        # Estimation of |sigma_2(k/M_1, l/M_2)| * 1/(M_2)
    #        dnorm_sigma_2 = [0 for _ in range(self.M_1*(self.M_2+1))]

    #        # Derivatives of |sigma_1|
    #        for k in range(self.M_1):
    #            tau = np.array([l/self.M_2 for l in range(self.M_2+1)]) # Independent of k
    #            gtau = np.array([self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)].norm() for l in range(self.M_2+1)])
    #            
    #            # Retrieve slopes of not-a-knot cubic spline 
    #            slope = SlopeCSNotAKnot(tau, gtau)
    #            for l in range(self.M_2+1):
    #                dnorm_sigma_1[k + l*self.M_1] = slope[l]/self.M_2
    #         
    #        # Derivatives of |sigma_2|
    #        for l in range(self.M_2+1):
    #            # At poles assign to 0
    #            if l==0 or l==self.M_2:
    #                for k in range(self.M_1):
    #                    dnorm_sigma_2[k + l*self.M_1] = 0

    #            else:
    #                tau = np.array([l/self.M_2 for l in range(self.M_2+1)]) # Independent of k
    #                gtau = np.array([self.coefs[k + l*self.M_1 + self.M_1*(self.M_2+1)].norm() for l in range(self.M_2+1)])
    #            
    #                # Retrieve slopes of not-a-knot cubic spline 
    #                slope = SlopeCSNotAKnot(tau, gtau)
    #                for k in range(self.M_1):
    #                    dnorm_sigma_2[k + l*self.M_1] = slope[k]/self.M_1

    #        
    #        for k in range(self.M_1):
    #            for l in range(1, self.M_2):
    #                # We are working patch by patch independently
    #                # u=0,1 and v=0,1 correspond to the corners of the patch [k/M_1, k+1/M_1]x[l/M_2, l+1/M_2]
    #                # We have M_1 coefs in u, M_2+1 coefs in v
    #                # In case k = M_1-1, c[k+1 = M_1, l] = c[0, l] by periodicity

    #                # Surface values 
    #                sigma = np.empty((2,2), dtype=object)
    #                sigma[0,0] = self.coefs[k + l*self.M_1]
    #                sigma[0,1] = self.coefs[k + (l+1)*self.M_1]
    #                sigma[1,0] = self.coefs[(k+1)%self.M_1 + l*self.M_1]
    #                sigma[1,1] = self.coefs[(k+1)%self.M_1 + (l+1)*self.M_1]
    #            
    #                # First order u-derivatives
    #                sigma_1 = np.empty((2,2), dtype=object)
    #                sigma_1[0,0] = self.coefs[k + l*self.M_1 + 2*self.M_1*(self.M_2+1)]
    #                sigma_1[0,1] = self.coefs[k + (l+1)*self.M_1 + 2*self.M_1*(self.M_2+1)]
    #                sigma_1[1,0] = self.coefs[(k+1)%self.M_1 + l*self.M_1 + 2*self.M_1*(self.M_2+1)]
    #                sigma_1[1,1] = self.coefs[(k+1)%self.M_1 + (l+1)*self.M_1 + 2*self.M_1*(self.M_2+1)]

    #                # First order v-derivatives
    #                sigma_2 = np.empty((2,2), dtype=object)
    #                sigma_2[0,0] = self.coefs[k + l*self.M_1 + self.M_1*(self.M_2+1)]
    #                sigma_2[0,1] = self.coefs[k + (l+1)*self.M_1 + self.M_1*(self.M_2+1)]
    #                sigma_2[1,0] = self.coefs[(k+1)%self.M_1 + l*self.M_1 + self.M_1*(self.M_2+1)]
    #                sigma_2[1,1] = self.coefs[(k+1)%self.M_1 + (l+1)*self.M_1 + self.M_1*(self.M_2+1)]

    #                # First and second columns of the Q matrix
    #                Qc1 = np.array([sigma[0,0], sigma[1,0], sigma_1[0,0], sigma_1[1,0]])
    #                Qc2 = np.array([sigma[0,1], sigma[1,1], sigma_1[0,1], sigma_1[1,1]])

    #                # Compute 2nd-order u-deriv at the corners of the patch 
    #                sigma_11 = np.empty((2,2), dtype=object)
    #                sigma_11[0,0] = F(0).dot(Qc1)
    #                sigma_11[0,1] = F(0).dot(Qc2)
    #                sigma_11[1,0] = F(1).dot(Qc1)
    #                sigma_11[1,1] = F(1).dot(Qc2) 
    #                
    #                # First and second rows of the Q matrix
    #                Qr1 = np.array([sigma[0,0], sigma[0,1], sigma_2[0,0], sigma_2[0,1]])
    #                Qr2 = np.array([sigma[1,0], sigma[1,1], sigma_2[1,0], sigma_2[1,1]])

    #                # Compute 2nd-order v-deriv at the corners of the patch 
    #                sigma_22 = np.empty((2,2), dtype=object)
    #                sigma_22[0,0] = Qr1.dot(G(0))
    #                sigma_22[0,1] = Qr1.dot(G(1)) 
    #                sigma_22[1,0] = Qr2.dot(G(0))
    #                sigma_22[1,1] = Qr2.dot(G(1))

    #                # Normal vector
    #                n_hat = np.empty((2,2), dtype=object)
    #                for u in [0,1]:
    #                    for v in [0,1]:
    #                        n_hat[u,v] = sigma_1[u,v].crossProduct(sigma_2[u,v])
    #                        n_hat[u,v] /= n_hat[u,v].norm()

    #                
    #                # Matrice of twist vector coordinates at each of the 4 corners in 
    #                # the tangential basis (n, r_1, r_2). In order (0,0), (0,1), (1,0), (1,1)
    #                sigma_12 = np.zeros((3,4), dtype=float)

    #                j = 0
    #                for u in [0,1]:
    #                    for v in [0,1]:
    #                        # The gaussian curvature
    #                        K = GaussCurv((k+u)/self.M_1, (l+v)/self.M_2)
    #                
    #                        H_2 = sigma_1[u,v].norm()**2*sigma_2[u,v].norm()**2 - sigma_1[u,v].dot(sigma_2[u,v])
    #                        sq_comp = (n_hat[u,v].dot(sigma_11[u,v]))*(n_hat[u,v].dot(sigma_22[u,v]))-K*H_2
    #                        
    #                        # In case the number below the sqrt is negative, the Gaussian curvature
    #                        # given is invalid. We choose to change it continuously until the number becomes positive 
    #                        # that is 0.
    #                        if np.abs(sq_comp) < 1e-10:
    #                            sq_comp = 0
    #                        elif sq_comp < 0:
    #                            print ("Error for k=%d, l=%d, the value below sqrt is invalid: %.3g" % (k,l,sq_comp))
    #                            sq_comp = 0


    #                        # Normal component
    #                        # Choose positive sign at north pole and then alternating sign at patch ends
    #                        sigma_12[0, j] = (-1)**(k+l)*np.sqrt(sq_comp)

    #                        # First (normalized) tangential component of twist vector
    #                        sigma_12[1, j] = dnorm_sigma_1[(k+u)%self.M_1 + (l+v)*self.M_1]

    #                        # Second (normalized) tangential component of the twist vector
    #                        sigma_12[2, j]  = dnorm_sigma_2[(k+u)%self.M_1 + (l+v)*self.M_1]

    #                        # Use the matrix to change from basis (n_hat, r_1_hat, r_2_hat) 
    #                        # to canonical base (e_1, e_2, e_3)
    #                        sc_1 = sigma_1[u,v].norm()
    #                        sc_2 = sigma_2[u,v].norm()
    #                        
    #                        P = np.array([[n_hat[u,v].x, n_hat[u,v].y, n_hat[u,v].z],
    #                                      [sigma_1[u,v].x/sc_1, sigma_1[u,v].y/sc_1, sigma_1[u,v].z/sc_1],
    #                                      [sigma_2[u,v].x/sc_2, sigma_2[u,v].y/sc_2, sigma_2[u,v].z/sc_2]]).T
    #                        sigma_12[:,j] = P.dot(sigma_12[:,j])

    #                        # Update the coefficients c_4
    #                        # Given shared edges between patches each coefficient will be updated up to 4 times
    #                        #twist_hat = Point3D(sigma_12[0,j], sigma_12[1,j], sigma_12[2,j])
    #                        #twist_hat.clip()
    #                        #self.coefs[(k+u)%self.M_1 + (l+v)*self.M_1 + 3*self.M_1*(self.M_2+1)].updateFromPoint(twist_hat)

    #                        twist_hat = Point3D(sigma_12[0,j], sigma_12[1,j], sigma_12[2,j])
    #                        twist_hat.clip()
    #                        twist = self.coefs[(k+u)%self.M_1 + (l+v)*self.M_1 + 3*self.M_1*(self.M_2+1)]
    #                        print("At u=%.3g, v=%.3g estimated %s vs true %s" % ((k+u)%self.M_1/self.M_1,
    #                                                                             (l+v)/self.M_2, twist_hat, twist))
    #                        
    #                        j += 1
    #                        
    #    else:
    #        raise ValueError("Please choose 'Selesnick' for the method")
