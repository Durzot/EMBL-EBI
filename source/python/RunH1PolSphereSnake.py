# -*- coding: utf-8 -*-
"""
Class that produces a snake from the H1SphereSnake class and produces a VTK rendering

Designed to be run in Python 3 virtual environment 3.7_vtk

@version: July 16, 2019
@author: Yoann Pradat
"""
import argparse
from snake.H1PolSphereSnake import H1PolSphereSnake
from roi.ROI3DSnake import ROI3DSnake
from snake3D.Snake3DNode import Snake3DNode
import numpy as np

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
parser.add_argument('--set_twist', type=str, default=None, help='Optional: choose random or null')
parser.add_argument('--est_twist', type=str, default=None, help='Optional: choose naive, selesnick or oscillation')
parser.add_argument('--nSamplesPerSeg', type=int, default=7, help='number of scales between consecutive control points')
parser.add_argument('--renWinSizeX', type=int, default=900, help='size of display window width in pixels')
parser.add_argument('--renWinSizeY', type=int, default=900, help='size of display window height in pixels')
opt = parser.parse_args()

# =========================================== SNAKE DISPLAY =========================================== # 

# Create SphereSnake and intialize
snake = H1PolSphereSnake(opt.M_1, opt.M_2, opt.nSamplesPerSeg, opt.hidePoints)
snake.initializeDefaultShape(shape=opt.shape)

if opt.set_twist is not None:
    snake.setTwist(opt.set_twist)

if opt.est_twist is not None:
    snake.estimateTwist(opt.est_twist)

## Create shape with volcano-like aperture at north pole
#for k in range(snake.M_1):
#    snake.coefs[k].updateFromCoords(0,0,0.5) # Move north pole downward
#    snake.coefs[k + snake.M_1 + snake.M_1*(snake.M_2+1)].updateFromCoords(0, 0, 0) # Set v-deriv to 0
#    snake.coefs[k + snake.M_1 + 3*snake.M_1*(snake.M_2+1)].updateFromCoords(np.random.rand(1), np.random.rand(1), 
#                                                                            np.random.rand(1)) 
#    snake._updateContour()

# Create 3D painter
roi3dsnake = ROI3DSnake(snake)

# Display the snake
roi3dsnake.displaySnake(renWinSize=(opt.renWinSizeX, opt.renWinSizeY))

import numpy as np
from auxiliary.aux import Color, Point3D
from process.SurfaceOracles import *
from process.BSpline import SlopeCSNotAKnot
from snake3D.Snake3DNode import Snake3DNode
from snake3D.Snake3DScale import Snake3DScale
from scipy import integrate

# The matrix equation is (K \bigotimeplus L) R = C with unknown R while K and L depend on the knot
# locations and C depends in addition to that on surface values and first-order derivatives at knots

K = np.zeros((snake.M_2+1, snake.M_2+1))
L = np.zeros((snake.M_1, snake.M_1))
C = np.zeros(snake.M_1*(snake.M_2+1))

# As the spacing is uniform in our scheme, the formulas for K and L are greatly simplified
for i in range(1, snake.M_2):
    K[i,i-1] = -3./snake.M_2**3
    K[i,i] = 8./snake.M_2**3
    K[i,i+1] = -3./snake.M_2**3

for i in range(snake.M_1):
    L[i,i-1] = -3./snake.M_1**3
    L[i,i] = 8./snake.M_1**3
    L[i,(i+1)%snake.M_1] = -3./snake.M_1**3

K[0,0] = 4./snake.M_2**3
K[0,1] = -3/snake.M_2**3
K[snake.M_2,snake.M_2-1] = -3/snake.M_2**3
K[snake.M_2,snake.M_2] = 4./snake.M_2**3

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
                G[q][i,j] /= snake.M_1
            if j >= 2:
                G[q][i,j] /= snake.M_2

# Link coefficients c_1, c_2, c_3 to quantities p, g, f
p = np.empty((snake.M_1+1, snake.M_2+1), dtype=Point3D)
g = np.empty((snake.M_1+1, snake.M_2+1), dtype=Point3D)
f = np.empty((snake.M_1+1, snake.M_2+1), dtype=Point3D)
r = np.empty((snake.M_1+1, snake.M_2+1), dtype=Point3D)
 
for i in range(snake.M_1+1):
    for j in range(snake.M_2+1):
        # p[i,j] = c_1[i,j]
        p[i,j] = snake.coefs[i%snake.M_1 + j*snake.M_1]
        # g[i,j] = c_2[i,j]/l_j
        g[i,j] = snake.coefs[i%snake.M_1 + j*snake.M_1 + snake.M_1*(snake.M_2+1)]*snake.M_2
        # f[i,j] = c_3[i,j]/h_i
        f[i,j] = snake.coefs[i%snake.M_1 + j*snake.M_1 + 2*snake.M_1*(snake.M_2+1)]*snake.M_1
        # f[i,j] = c_3[i,j]/(h_il_j)
        r[i,j] = snake.coefs[i%snake.M_1 + j*snake.M_1 + 3*snake.M_1*(snake.M_2+1)]*snake.M_1*snake.M_2

# Compute quantities defining matrices B_{i,j} as defined in Guo et Han article except for 
# \tilde{r} which carries a mistake
bar_f = np.empty((snake.M_1+1, snake.M_2+1), dtype=Point3D)
bar_g = np.empty((snake.M_1+1, snake.M_2+1), dtype=Point3D)
bar_r = np.empty((snake.M_1+1, snake.M_2+1), dtype=Point3D)

# Compute matrices B_{i,j} i = 0, ..., M_1-1, j=0, ..., M_2-1
B = np.empty((snake.M_1,snake.M_2), dtype=object)
Q = np.empty((snake.M_1,snake.M_2), dtype=object)
for i in range(snake.M_1):
    for j in range(snake.M_2):
        # Be aware of the successive rewriting of values
        # in the matrices bar_f, bar_g, bar_r

        bar_f[i,j] = (p[i+1,j]-p[i,j])*snake.M_1
        bar_f[i+1,j] = (p[i+1,j]-p[i,j])*snake.M_1
        bar_f[i,j+1] = (p[i+1,j+1]-p[i,j+1])*snake.M_1
        bar_f[i+1,j+1] = (p[i+1,j+1]-p[i,j+1])*snake.M_1

        bar_g[i,j] = (p[i,j+1]-p[i,j])*snake.M_2
        bar_g[i,j+1] = (p[i,j+1]-p[i,j])*snake.M_2
        bar_g[i+1,j] = (p[i+1,j+1]-p[i+1,j])*snake.M_2
        bar_g[i+1,j+1] = (p[i+1,j+1]-p[i+1,j])*snake.M_2

        qty = (p[i,j] - p[i,j+1] - p[i+1,j] + p[i+1,j+1])*snake.M_1*snake.M_2
        bar_r[i,j] = qty
        bar_r[i+1,j] = qty
        bar_r[i,j+1] = qty
        bar_r[i+1,j+1] = qty
        
        B[i,j] = np.empty((4,4), dtype=Point3D)
        Q[i,j] = np.empty((4,4), dtype=Point3D)
        for s in range(4):
            for t in range(4):
                if s >= 2 and t >= 2:
                    B[i,j][s,t] = -bar_r[i+s-2,j+t-2]
                    Q[i,j][s,t] = r[i+s-2,j+t-2]
                elif s >= 2:
                    B[i,j][s,t] = f[i+s-2,j+t] - bar_f[i+s-2,j+t]
                    Q[i,j][s,t] = f[i+s-2,j+t]
                elif t >= 2:
                    B[i,j][s,t] = g[i+s,j+t-2] - bar_g[i+s,j+t-2]
                    Q[i,j][s,t] = g[i+s,j+t-2]
                else:
                    B[i,j][s,t] = Point3D(0,0,0)
                    Q[i,j][s,t] = p[i+s,j+t]

# The energy on patch I_{i,j} is E_{i,j} = h_i l_j \int \int Tr(H(u,v) B_tilde{i,j})
# Fu = [f_0(u), f_1(u), g_0(u), g_1(u)]
# Fv = [f_0(v), f_1(v), g_0(v), g_1(v)]
Fu = lambda u: np.array([[1-3*u**2 + 2*u**3], 
                        [3*u**2 - 2*u**3], 
                        [u*(u-1)**2/snake.M_1],
                        [u**2*(u-1)/snake.M_1]])
Fv = lambda v: np.array([[1-3*v**2 + 2*v**3], 
                        [3*v**2 - 2*v**3], 
                        [v*(v-1)**2/snake.M_2],
                        [v**2*(v-1)/snake.M_2]])
H = lambda u,v: Fu(u).dot(Fv(v).T)

h = 0.01
for i in range(snake.M_1):
    for j in range(snake.M_2):
        I = 0
        for u in np.r_[i/snake.M_1:(i+1)/snake.M_1:h]:
            for v in np.r_[j/snake.M_2:(j+1)/snake.M_2:h]:
                real_pt = Fu(4*u-i).T.dot(Q[i,j]).dot(Fv(4*v-j))[0,0]
                linear_pt = Fu(4*u-i)[:2].T.dot(Q[i,j][:2, :2]).dot(Fv(4*v-j)[:2])[0,0]
                I += (real_pt - linear_pt).norm()**2*h*h

        print("estim oscillation %.3g vs compute oscillation %.3g" % (I, E[i,j]))

# Compute energy on each patch
B_tilde = np.copy(B)
E = np.zeros((snake.M_1,snake.M_2))
for i in range(snake.M_1):
    for j in range(snake.M_2):
        for s in range(2,4):
            for t in range(2,4):
                B_tilde[i,j][s,t] += r[i+s-2,j+t-2]
        
        func = lambda u,v : np.trace(H(u,v).dot(B_tilde[i,j])).norm()**2
        E[i,j], _ = integrate.dblquad(func, 0, 1, lambda x: 0, lambda x: 1)
        E[i,j] /= snake.M_1*snake.M_2

# Compute long column vector C
C = np.empty(snake.M_1*(snake.M_2+1), dtype=Point3D)
for i in range(snake.M_1):
    # First M_1 elements of C
    C[i] = (np.trace(G[1].dot(B[i-1, 0])) + np.trace(G[0].dot(B[i,0])))/(snake.M_1**2*snake.M_2**2)

    # Last M_1 elements of C
    C[i+snake.M_1*snake.M_2] = (np.trace(G[3].dot(B[i-1, snake.M_2-1])) \
                              + np.trace(G[2].dot(B[i,snake.M_2-1])))/(snake.M_1**2*snake.M_2**2)
    
    # All in-between elements (4 patches)
    for j in range(1, snake.M_2):
        C[i+snake.M_1*j] = (np.trace(G[3].dot(B[i-1, j-1])) \
                          + np.trace(G[2].dot(B[i,j-1])) \
                          + np.trace(G[1].dot(B[i-1,j])) \
                          + np.trace(G[0].dot(B[i,j])))/(snake.M_1**2*snake.M_2**2)

# Inverse the system
R = np.linalg.inv(M).dot(C)

#for j in range(snake.M_2+1):
#    for i in range(snake.M_1):
#        print("At i=%d, j=%d" % (i,j))
#        print("optimal twist %s" % (R[i+j*snake.M_1]/(snake.M_1*snake.M_2)))
#        print("real twist %s" % snake.coefs[i + j*snake.M_1 + 3*snake.M_1*(snake.M_2+1)])
#        snake.coefs[i + j*snake.M_1 + 3*snake.M_1*(snake.M_2+1)] = R[i+j*snake.M_1]/(snake.M_1*snake.M_2)

# update r on the grid
for i in range(snake.M_1+1):
    for j in range(snake.M_2+1):
        # f[i,j] = c_3[i,j]/(h_il_j)
        r[i,j] = R[i%snake.M_1+j*snake.M_1] 

# Compute energy on each patch
B_tilde = np.copy(B)
E = np.zeros((snake.M_1,snake.M_2))
for i in range(snake.M_1):
    for j in range(snake.M_2):
        for s in range(2,4):
            for t in range(2,4):
                B_tilde[i,j][s,t] += r[i+s-2,j+t-2]
        
        func = lambda u,v : np.trace(H(u,v).dot(B_tilde[i,j])).norm()**2
        E[i,j], _ = integrate.dblquad(func, 0, 1, lambda x: 0, lambda x: 1)
        E[i,j] /= snake.M_1*snake.M_2

h = 0.01
for i in range(snake.M_1):
    for j in range(snake.M_2):
        for s in range(4):
            for t in range(4):
                if s >= 2 and t >= 2:
                    Q[i,j][s,t] = r[i+s-2,j+t-2]

        I = 0
        for u in np.r_[i/snake.M_1:(i+1)/snake.M_1:h]:
            for v in np.r_[j/snake.M_2:(j+1)/snake.M_2:h]:
                real_pt = Fu(4*u-i).T.dot(Q[i,j]).dot(Fv(4*v-j))[0,0]
                linear_pt = Fu(4*u-i)[:2].T.dot(Q[i,j][:2, :2]).dot(Fv(4*v-j)[:2])[0,0]
                I += (real_pt - linear_pt).norm()**2*h*h

        print("estim oscillation %.3g vs compute oscillation %.3g" % (I, E[i,j]))

