# -*- coding: utf-8 -*-
"""
Useful functions for making links between Python and VTK objects.
Designed to be run in Python 3 virtual environment 3.7_vtk

@version: May 15, 2019
@author: Yoann Pradat
"""

import numpy as np
from vtk import vtkCellArray
from vtk import vtkPoints
from vtk import vtkActor
from vtk import vtkIdTypeArray
from vtk import vtkIntArray
from vtk import vtkPolyData
from vtk import vtkPolyDataMapper
from vtk import vtkSphereSource
import vtk.util.numpy_support as vtk_np

def prepareCells(indexes):
    """
    Parameters
    ----------
    array: np.array(n, p) dtype=int
    
    Return
    ------
    result: np.array(total_length,) dtype=int32
    """
    n, p = indexes.shape
    total_len = n*(p+1)
    
    result = np.zeros((total_len), dtype=np.int32)
    offset = 0
    for i in range(n):
        s_cells = indexes[i]
        s_len = s_cells.shape[0]
        
        result[offset] = s_len
        offset += 1
        for j in range(s_len):
            result[offset] = s_cells[j]
            offset += 1
    return result
    
def getIdTypedArray(array):
    """
    Parameters
    ----------
    array: np.array(n, ) dtype=int32
    
    Return
    ------
    result: vtkIdTypedArray
    """
    result = vtkIdTypeArray()
    iarray = vtk_np.numpy_to_vtk(array)
    result.DeepCopy(iarray)
    return result

def getCells(numCell, cells):
    """
    Parameters
    ----------
    numCell: int
    cells: np.array(n,) dtype=int32
    
    Return
    ------
    result: vtkCellArray
    """
    result = vtkCellArray()
    result.SetCells(numCell, getIdTypedArray(cells))
    return result
