# -*- coding: utf-8 -*-
"""
Class that produces a snake from the SphereSnake class and produces a VTK rendering

Designed to be run in Python 3 virtual environment 3.7_vtk

@version: May 16, 2019
@author: Yoann Pradat
"""

import argparse
from snake.SphereSnake import SphereSnake
from roi.ROI3DSnake import ROI3DSnake

# =========================================== PARAMETERS =========================================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--M_1', type=int, default=3, help='number of control points on latitudes')
parser.add_argument('--M_2', type=int, default=3, help='number of control points on longitudes')
parser.add_argument('--nSamplesPerSeg', type=int, default=7, help='number of scales between consecutive control points')
parser.add_argument('--renWinSizeX', type=int, default=900, help='size of display window width in pixels')
parser.add_argument('--renWinSizeY', type=int, default=900, help='size of display window height in pixels')
opt = parser.parse_args()

# =========================================== SNAKE DISPLAY =========================================== # 
# Create SphereSnake and intialize
snake = SphereSnake(opt.M_1, opt.M_2, opt.nSamplesPerSeg)
snake.initializeDefaultShape()

# Create 3D painter
roi3dsnake = ROI3DSnake(snake)

# Display the snake
roi3dsnake.displaySnake(renWinSize=(opt.renWinSizeX, opt.renWinSizeY))
