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
parser.add_argument('--M_1', type=int, default=5, help='number of control points on latitudes')
parser.add_argument('--M_2', type=int, default=5, help='number of control points on longitudes')
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
#
## Create 3D painter
#roi3dsnake = ROI3DSnake(snake)
#
## Display the snake
#roi3dsnake.displaySnake(renWinSize=(opt.renWinSizeX, opt.renWinSizeY))

