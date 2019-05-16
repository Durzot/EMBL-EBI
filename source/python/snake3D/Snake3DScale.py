# -*- coding: utf-8 -*-
"""
This class is inspired by Snake3DScale.java class of bigsnakeutils plugin
from Biomedical Imaging Group. 

Designed to be run in Python 3 virtual environment 3.7_vtk

This class is used to store the scales that are used to draw the skin

@version: May 14, 2019
@author: Yoann Pradat
"""

class Snake3DScale(object):
    """
    Attributes
    ----------
    color_: Color
    closed_ : boolean
    points_ : np.array (n_points, 3)
    
    Methods
    ----------
    getColor()
    isClosed()
    getCoordinates()
    """
    def __init__(self, color, closed, points):
        self.color = color
        self.closed = closed
        self.points = points

    def getColor(self):
        return self.color

    def isClosed(self):
        return self.closed

    def getCoordinates(self):
        return self.points


