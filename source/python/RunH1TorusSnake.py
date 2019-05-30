# -*- coding: utf-8 -*-
"""
Class that produces a snake from the H1TorusSnake class and produces a VTK rendering

Designed to be run in Python 3 virtual environment 3.7_vtk

@version: May 28, 2019
@author: Yoann Pradat
"""

import argparse
from snake.H1TorusSnake import H1TorusSnake
from roi.ROI3DSnake import ROI3DSnake

# =========================================== PARAMETERS =========================================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--M_1', type=int, default=4, help='number of control points on latitudes')
parser.add_argument('--M_2', type=int, default=4, help='number of control points on longitudes')
parser.add_argument('--nullTwist', type=bool, default=False, help='set twist vector everywhere to 0')
parser.add_argument('--randTwist', type=bool, default=False, help='add random normal pert totwist vector everywhere')
parser.add_argument('--nSamplesPerSeg', type=int, default=7, help='number of scales between consecutive control points')
parser.add_argument('--renWinSizeX', type=int, default=900, help='size of display window width in pixels')
parser.add_argument('--renWinSizeY', type=int, default=900, help='size of display window height in pixels')
opt = parser.parse_args()

# =========================================== SNAKE DISPLAY =========================================== # 
# Create SphereSnake and intialize
snake = H1TorusSnake(opt.M_1, opt.M_2, opt.nSamplesPerSeg)
snake.initializeDefaultShape()

if opt.nullTwist:
    snake.setNullTwist()

if opt.randTwist:
    snake.setRandomPertTwist(mu=0, sigma=1)

# Create 3D painter
roi3dsnake = ROI3DSnake(snake)

# Display the snake
roi3dsnake.displaySnake(renWinSize=(opt.renWinSizeX, opt.renWinSizeY))

