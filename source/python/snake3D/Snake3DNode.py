# -*- coding: utf-8 -*-
"""
Designed to be run in Python 3 virtual environment 3.7_vtk

This class is used to store the snake-defining anchor points

@version: May 14, 2019
@author: Yoann Pradat
"""

import numpy as np
from auxiliary.aux import Point3D, Color

class Snake3DNode(Point3D):
    def __init__(self, x, y, z, hidden=False, color=Color(0,0,255)):
        super(Snake3DNode, self).__init__(x, y, z)
        self.hidden = hidden
        self.color = color
    
    def setColor(self, color):
        self.color = color

    def getColor(self):
        return self.color

    def setPoint(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def getPoint(self, x, y, z):
        return Point3D(self.x, self.y, self.z)

    def getCoordinates(self):
        return np.array([self.x, self.y, self.z])

