# -*- coding: utf-8 -*-
"""
Designed to be run in Python 3 virtual environment 3.7_vtk

This class is used to store the snake-defining anchor points

@version: May 14, 2019
@author: Yoann Pradat
"""

from auxiliary.aux import Point3D

class Snake3DNode(Point3D):
    def __init__(self, x, y, z, hidden=False, color=None):
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

    # -–––––----------------------––-------
    # Overload useful operators
    # -–––––----------------------––-------
    
    def __str__(self):
        return "Point %.3f, %.3f, %.3f" % (self.x, self.y, self.z)

    def __add__(self, other):
        return Snake3DNode(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        return Snake3DNode(self.x-other.x, self.y-other.y, self.z-other.z)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Snake3DNode(self.x*scalar, self.y*scalar, self.z*scalar)
        else:
            return NotImplemented

    def __rmul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Snake3DNode(self.x*scalar, self.y*scalar, self.z*scalar)
        else:
            return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Snake3DNode(self.x/scalar, self.y/scalar, self.z/scalar)
        else:
            return NotImplemented

