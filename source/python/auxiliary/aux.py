# -*- coding: utf-8 -*-
"""
Designed to be run in Python 3 virtual environment 3.7_vtk

This class is used to define classes used throughout the code

@version: July 16, 2019
@author: Yoann Pradat
"""

import numpy as np

class Point3D(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def updateFromPoint(self, point):
        self.x = point.x
        self.y = point.y
        self.z = point.z

    def updateFromCoords(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def crossProduct(self, other):
        return Point3D(self.y*other.z - self.z*other.y, self.z*other.z - self.x*other.z, self.x*other.y-self.y*other.x)

    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def clip(self, lim=1e-10):
        if abs(self.x) < lim:
            self.x = 0
        if abs(self.y) < lim:
            self.y = 0
        if abs(self.z) < lim:
            self.z = 0

    def norm(self):
        return np.sqrt(self.x**2+self.y**2+self.z**2)

    def __str__(self):
        return "Point %.3f, %.3f, %.3f" % (self.x, self.y, self.z)

    def __add__(self, other):
        return Point3D(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        return Point3D(self.x-other.x, self.y-other.y, self.z-other.z)

    def __neg__(self):
        return Point3D(-self.x, -self.y, -self.z)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Point3D(self.x*scalar, self.y*scalar, self.z*scalar)
        else:
            return NotImplemented

    def __rmul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Point3D(self.x*scalar, self.y*scalar, self.z*scalar)
        else:
            return NotImplemented

    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Point3D(self.x/scalar, self.y/scalar, self.z/scalar)
        else:
            return NotImplemented


class Color(object):
    def __init__(self, red, green, blue):
        self.red_ = red
        self.green_ = green
        self.blue_ = blue

    def getRed(self):
        return self.red_

    def getGreen(self):
        return self.green_

    def getBlue(self):
        return self.blue_
