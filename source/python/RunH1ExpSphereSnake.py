# -*- coding: utf-8 -*-
"""
Class that produces a snake from the H1SphereSnake class and produces a VTK rendering

Designed to be run in Python 3 virtual environment 3.7_vtk

@version: June 10, 2019
@author: Yoann Pradat
"""

import argparse
from snake.H1ExpSphereSnake import H1ExpSphereSnake
from roi.ROI3DSnake import ROI3DSnake

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# =========================================== PARAMETERS =========================================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--M_1', type=int, default=4, help='number of control points on latitudes')
parser.add_argument('--M_2', type=int, default=4, help='number of control points on longitudes')
parser.add_argument('--hidePoints', type=boolean_string, default='True', help='False for displaying control points')
parser.add_argument('--shape', type=str, default='sphere', help='shape to be represented')
parser.add_argument('--twist', type=str, default='default', help='mode for the twist value')
parser.add_argument('--nSamplesPerSeg', type=int, default=7, help='number of scales between consecutive control points')
parser.add_argument('--renWinSizeX', type=int, default=900, help='size of display window width in pixels')
parser.add_argument('--renWinSizeY', type=int, default=900, help='size of display window height in pixels')
opt = parser.parse_args()

# =========================================== SNAKE DISPLAY =========================================== # 
# Create SphereSnake and intialize
snake = H1ExpSphereSnake(opt.M_1, opt.M_2, opt.nSamplesPerSeg, opt.hidePoints)
snake.initializeDefaultShape(shape=opt.shape)

if opt.twist=='null':
    snake.setNullTwist
elif opt.twist=='rand':
    snake.setRandTwist(mu=0, sigma=1)
elif opt.twist=='Selesnick':
    snake.estimateTwist('Selesnick')
elif opt.twist=='default':
    pass
else:
    raise ValueError('Invalid value of twist')

# Create 3D painter
roi3dsnake = ROI3DSnake(snake)

# Display the snake
roi3dsnake.displaySnake(renWinSize=(opt.renWinSizeX, opt.renWinSizeY))



## The following estimation of the twist vector is an implementation of what S.A Selesnick 
## describes in his article "Local invariants and twist vectors in computer-aided geometric design" of 1981
##
## The idea is to estimate first the normal component n.sigma_12 of the twist vector from knowledge of 
## the Gaussian curvature. Then the tangential components are estimated by estimating the variation 
## of the length of the tangential vector sigma_1 as the second parameter changes and vice-versa
#
## We make use of the fact that phi_1(0)=1, phi_1(1)=0 and phi_2(0)=phi_2(1)=0
## Indeed this allows us to estimate second derivates without knowing the cross-derivative
#
#import numpy as np
#from auxiliary.aux import Point3D
#from process.BSpline import SlopeCSNotAKnot
#
#from process import BSpline
#from auxiliary import aux
#importlib.reload(BSpline)
#importlib.reload(aux)
#from process.BSpline import SlopeCSNotAKnot
#from auxiliary.aux import Point3D
#
#def r_1(u,v):
#    return Point3D(-2*np.pi*np.sin(2*np.pi*u)*np.sin(np.pi*v), 2*np.pi*np.cos(2*np.pi*u)*np.sin(np.pi*v), 0)/snake.M_1
#
#def r_11(u,v):
#    return -4*np.pi**2*Point3D(np.cos(2*np.pi*u)*np.sin(np.pi*v), np.sin(2*np.pi*u)*np.sin(np.pi*v), 0)/snake.M_1**2
#
#def r_2(u,v):
#    return Point3D(np.pi*np.cos(2*np.pi*u)*np.cos(np.pi*v), np.pi*np.sin(2*np.pi*u)*np.cos(np.pi*v),
#                   -np.pi*np.sin(np.pi*v))/snake.M_2
#def r_22(u,v):
#    return -np.pi**2*Point3D(np.cos(2*np.pi*u)*np.sin(np.pi*v), np.sin(2*np.pi*u)*np.sin(np.pi*v),
#                   np.cos(np.pi*v))/snake.M_2**2
#def r_12(u,v):
#    return 2*np.pi**2*Point3D(-np.sin(2*np.pi*u)*np.cos(np.pi*v), np.cos(2*np.pi*u)*np.cos(np.pi*v),
#                              0)/(snake.M_1*snake.M_2)
#
#def n_norm(u,v):
#    return Point3D(-np.cos(2*np.pi*u)*np.sin(np.pi*np.abs(v)), -np.sin(2*np.pi*u)*np.sin(np.pi*np.abs(v)),
#                    -np.cos(np.pi*v)*np.sign(v))
#
#
## Gaussian curvature
#K = 1
#
#def F(u):
#    return np.array([snake._Dg_1(snake.w_1, u, 2), snake._Dg_1(snake.w_1, 1-u, 2), 
#                     snake._Dg_2(snake.w_1, u, 2), -snake._Dg_2(snake.w_1, 1-u, 2)])
#
#def G(v):
#    return np.array([snake._Dg_1(snake.w_2, v, 2), snake._Dg_1(snake.w_2, 1-v, 2), 
#                     snake._Dg_2(snake.w_2, v, 2), -snake._Dg_2(snake.w_2, 1-v, 2)])
#
#
## Estimate derivatives of |sigma_1| with respect to var 2 and |sigma_2| with respect ot var 1 
## at control points. This is done through cubic spline interpolation with not-a-knot condition. 
## Refer to chapter IV of A Practical Guide To Splines by De Boor for more details on this interpolation.
#
#dnorm_sigma_1 = [0 for _ in range(snake.M_1*(snake.M_2+1))]
#dnorm_sigma_2 = [0 for _ in range(snake.M_1*(snake.M_2+1))]
#
## Derivatives of |sigma_1|
#for k in range(snake.M_1):
#    tau = np.array([l/snake.M_2 for l in range(snake.M_2+1)]) # Independent of k
#    gtau = np.array([snake.coefs[k + l*snake.M_1 + 2*snake.M_1*(snake.M_2+1)].norm() for l in range(snake.M_2+1)])
#    
#    # Retrieve slopes of not-a-knot cubic spline 
#    slope = SlopeCSNotAKnot(tau, gtau)
#    for l in range(snake.M_2+1):
#        dnorm_sigma_1[k + l*snake.M_1] = slope[l]/snake.M_2
# 
## Derivatives of |sigma_2|
#for l in range(snake.M_2+1):
#    # At poles assign to 0
#    if l==0 or l==snake.M_2:
#        for k in range(snake.M_1):
#            dnorm_sigma_2[k + l*snake.M_1] = 0
#
#    else:
#        tau = np.array([l/snake.M_2 for l in range(snake.M_2+1)]) # Independent of k
#        gtau = np.array([snake.coefs[k + l*snake.M_1 + snake.M_1*(snake.M_2+1)].norm() for l in range(snake.M_2+1)])
#    
#        # Retrieve slopes of not-a-knot cubic spline 
#        slope = SlopeCSNotAKnot(tau, gtau)
#        for k in range(snake.M_1):
#            dnorm_sigma_2[k + l*snake.M_1] = slope[k]/snake.M_1
#
#
#for k in range(snake.M_1):
#    for l in range(1, snake.M_2):
#        # We are working patch by patch independently
#        # u=0,1 and v=0,1 correspond to the corners of the patch [k/M_1, k+1/M_1]x[l/M_2, l+1/M_2]
#        # We have M_1 coefs in u, M_2+1 coefs in v
#        # In case k = M_1-1, c[k+1 = M_1, l] = c[0, l] by periodicity
#        
#        # Surface values 
#        sigma = np.empty((2,2), dtype=object)
#        sigma[0,0] = snake.coefs[k + l*snake.M_1]
#        sigma[0,1] = snake.coefs[k + (l+1)*snake.M_1]
#        sigma[1,0] = snake.coefs[(k+1)%snake.M_1 + l*snake.M_1]
#        sigma[1,1] = snake.coefs[(k+1)%snake.M_1 + (l+1)*snake.M_1]
#    
#        # First order u-derivatives
#        sigma_1 = np.empty((2,2), dtype=object)
#        sigma_1[0,0] = snake.coefs[k + l*snake.M_1 + 2*snake.M_1*(snake.M_2+1)]
#        sigma_1[0,1] = snake.coefs[k + (l+1)*snake.M_1 + 2*snake.M_1*(snake.M_2+1)]
#        sigma_1[1,0] = snake.coefs[(k+1)%snake.M_1  + l*snake.M_1 + 2*snake.M_1*(snake.M_2+1)]
#        sigma_1[1,1] = snake.coefs[(k+1)%snake.M_1 + (l+1)*snake.M_1 + 2*snake.M_1*(snake.M_2+1)]
#
#        # First order v-derivatives
#        sigma_2 = np.empty((2,2), dtype=object)
#        sigma_2[0,0] = snake.coefs[k + l*snake.M_1 + snake.M_1*(snake.M_2+1)]
#        sigma_2[0,1] = snake.coefs[k + (l+1)*snake.M_1 + snake.M_1*(snake.M_2+1)]
#        sigma_2[1,0] = snake.coefs[(k+1)%snake.M_1  + l*snake.M_1 + snake.M_1*(snake.M_2+1)]
#        sigma_2[1,1] = snake.coefs[(k+1)%snake.M_1 + (l+1)*snake.M_1 + snake.M_1*(snake.M_2+1)]
#
#        # First and second columns of the Q matrix
#        Qc1 = np.array([sigma[0,0], sigma[1,0], sigma_1[0,0], sigma_1[1,0]])
#        Qc2 = np.array([sigma[0,1], sigma[1,1], sigma_1[0,1], sigma_1[1,1]])
#
#        # 2nd-order u-deriv at the corners of the patch [0,1]x[0,1]=[k/M_1, k+1/M_1]x[l/M_2, l+1/M_2]
#        sigma_11 = np.empty((2,2), dtype=object)
#        sigma_11[0,0] = F(0).dot(Qc1)
#        sigma_11[0,1] = F(0).dot(Qc2)
#        sigma_11[1,0] = F(1).dot(Qc1)
#        sigma_11[1,1] = F(1).dot(Qc2) 
#        
#        # First and second rows of the Q matrix
#        Qr1 = np.array([sigma[0,0], sigma[0,1], sigma_2[0,0], sigma_2[0,1]])
#        Qr2 = np.array([sigma[1,0], sigma[1,1], sigma_2[1,0], sigma_2[1,1]])
#
#        # 2nd-order v-deriv at the corners of the pacth [0,1]x[0,1] ([k/M_1, k+1/M_1]x[l/M_2, l+1/M_2]
#        sigma_22 = np.empty((2,2), dtype=object)
#        sigma_22[0,0] = Qr1.dot(G(0))
#        sigma_22[0,1] = Qr1.dot(G(1)) 
#        sigma_22[1,0] = Qr2.dot(G(0))
#        sigma_22[1,1] = Qr2.dot(G(1))
#
#        # Normal vector
#        n_hat = np.empty((2,2), dtype=object)
#        for u in [0,1]:
#            for v in [0,1]:
#                n_hat[u,v] = sigma_1[u,v].crossProduct(sigma_2[u,v])
#                n_hat[u,v] /= n_hat[u,v].norm()
#
#        # Matrice of twist vector coordinates at each of the 4 corners in 
#        # the tangential basis (n, r_1, r_2). In order (0,0), (0,1), (1,0), (1,1)
#        sigma_12 = np.zeros((3,4), dtype=float)
#
#        j = 0
#        for u in [0,1]:
#            for v in [0,1]:
#                H_2 = sigma_1[u,v].norm()**2*sigma_2[u,v].norm()**2 - sigma_1[u,v].dot(sigma_2[u,v])
#                sq_comp = (n_hat[u,v].dot(sigma_11[u,v]))*(n_hat[u,v].dot(sigma_22[u,v]))-K*H_2
#                
#                # In case the number below the sqrt is negative, the Gaussian curvature
#                # given is invalid. We choose to change it continuously until the number becomes positive 
#                # that is 0.
#                if np.abs(sq_comp) < 1e-8:
#                    sq_comp = 0
#                elif sq_comp < 0:
#                    print ("Error for k=%d, l=%d, the value below sqrt is invalid: %.3g" % (k,l,sq_comp))
#                    sq_comp = 0
#                
#                # Normal component
#                # Choose positive sign at north pole and then alternating sign at patch ends
#                sigma_12[0, j] = (-1)**(k+l)*np.sqrt(sq_comp)
#
#                # First (normalized) tangential component of twist vector
#                sigma_12[1, j] = dnorm_sigma_1[(k+u)%snake.M_1 + (l+v)*snake.M_1]
#
#                # Second (normalized) tangential component of the twist vector
#                sigma_12[2, j]  = dnorm_sigma_2[(k+u)%snake.M_1 + (l+v)*snake.M_1]
#
#                # Use the matrix to change from basis (n_hat, r_1_hat, r_2_hat) to canonical base (e_1, e_2, e_3)
#                sc_1 = sigma_1[u,v].norm()
#                sc_2 = sigma_2[u,v].norm()
#                
#                P = np.array([[n_hat[u,v].x, n_hat[u,v].y, n_hat[u,v].z],
#                              [sigma_1[u,v].x/sc_1, sigma_1[u,v].y/sc_1, sigma_1[u,v].z/sc_1],
#                              [sigma_2[u,v].x/sc_2, sigma_2[u,v].y/sc_2, sigma_2[u,v].z/sc_2]]).T
#                
#                sigma_12[:,j] = P.dot(sigma_12[:,j])
#                
#                twist_hat = Point3D(sigma_12[0,j], sigma_12[1,j], sigma_12[2,j])
#                twist_hat.clip()
#
#                twist = snake.coefs[(k+u)%snake.M_1 + (l+v)*snake.M_1 + 3*snake.M_1*(snake.M_2+1)]
#                print("At u=%.3g, v=%.3g" % ((k+u)%snake.M_1/snake.M_1, (l+v)/snake.M_2))
#                print("estimated %s vs true %s" % (twist_hat, twist))
#                
#                j += 1

