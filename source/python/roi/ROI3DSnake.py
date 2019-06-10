# -*- coding: utf-8 -*-
"""
This class is inspired by ROI3DSnake.java class of bigsnake3d plugin
from Biomedical Imaging Group. 

Designed to be run in Python 3 virtual environment 3.7_vtk

Rendering of a Snake surface in VTK

@version: June 10, 2019
@author: Yoann Pradat
"""

from vtk import vtkRenderer
from vtk import vtkRenderWindow
from vtk import vtkRenderWindowInteractor

from auxiliary.auxVtk import *

class ROI3DSnake(object):
    def __init__(self, snake, scaleSubsampling=1):
        self.snake = snake
        self.scaleSubsampling = scaleSubsampling

        self.pixelSizeX = 1
        self.pixelSizeY = 1
        self.pixelSizeZ = 1

        # vtkRenderer
        self.renderer = vtkRenderer()

    def _getScaledPoints(self, coordinates, scaleX, scaleY, scaleZ):
        """
        Parameters
        ----------
        coordinates: np.array (n_coordinates, 3)
        scaleX, scaleY, scaleZ: int
        
        Return
        ---------
        result: vktPoints vector of size (n_coordinates*3, 1)
        """
        n_coordinates = coordinates.shape[0]
        result = vtkPoints()
        if n_coordinates < 1:
            return result
        coordinatesVector = np.zeros((n_coordinates*3, 1), dtype=float)
        for i in range(n_coordinates):
            coordinatesVector[3*i, 0] = coordinates[i, 0]*scaleX
            coordinatesVector[3*i+1, 0] = coordinates[i, 1]*scaleY
            coordinatesVector[3*i+2, 0] = coordinates[i, 2]*scaleZ
            
        # vtkDoubleArray
        array = vtk_np.numpy_to_vtk(coordinatesVector)
        array.SetNumberOfComponents(3)
        result.SetData(array)
        return result

    def _nodeToWorldScale(self, coordinates, scaleX, scaleY, scaleZ):
        scaledPoint = np.array([coordinates[0]*scaleX, coordinates[1]*scaleY, coordinates[2]*scaleZ])
        return scaledPoint

    def _createNodesActors(self, renderer):
        nodes = self.snake.getNodes()
        for i in range(0, len(nodes)):
            if nodes[i].hidden:
                pass
            else:
                # Create sphere at nodes coordinates
                sphereNode = vtkSphereSource()
                nodePos = nodes[i].getCoordinates()
                sphereNode.SetCenter(self._nodeToWorldScale(nodePos, self.pixelSizeX, self.pixelSizeY, self.pixelSizeZ))

                sphereNode.SetRadius(self.pixelSizeX/20)
                sphereNode.SetThetaResolution(25)
                sphereNode.SetPhiResolution(25)

                sphereNode.Update()

                # Get vtkPolyData from sphere
                nodeData = sphereNode.GetOutput()

                # Set nodeData to mapper
                nodeMapper = vtkPolyDataMapper()
                nodeMapper.SetInputData(nodeData)

                # Set nodeMapper to actor and add actor to the renderer
                nodeActor = vtkActor()
                
                color = nodes[i].getColor()
                red = color.getRed()
                green = color.getGreen()
                blue = color.getBlue()
                nodeActor.GetProperty().SetColor(red/255., green/255., blue/255.)

                nodeActor.SetMapper(nodeMapper)
                renderer.AddActor(nodeActor)

    def _init3DRenderer(self, renderer):
        scales = self.snake.getScales()
        for i in range(0, len(scales), self.scaleSubsampling):
            scale = scales[i]
            scalePoints = scale.getCoordinates()

            # Scale points and Python conversion to vktPoints and scaling
            points = self._getScaledPoints(scalePoints, self.pixelSizeX, self.pixelSizeY, self.pixelSizeZ)

            cells = vtkCellArray()
            num_segments = scalePoints.shape[0]-1
            
            if scale.isClosed():
                num_segments += 1
            
            lineIdx = np.zeros((num_segments, 2), dtype=np.int32)
            for j in range(num_segments):
                lineIdx[j] = [j, j+1]

            if scale.isClosed():
                lineIdx[num_segments-1] = [0, num_segments-1]

            # Create cells and Python conversion to vktCellArray
            cells = getCells(num_segments, prepareCells(lineIdx))
            scaleData = vtkPolyData()

            # Set vertices and lines to scaleData
            scaleData.SetPoints(points)
            scaleData.SetLines(cells)

            # Set scaleData to mapper
            polyMapper = vtkPolyDataMapper()
            polyMapper.SetInputData(scaleData)

            # Set mapper to actor and add actor to the renderer
            lineActor = vtkActor()

            color = scale.getColor()
            red = color.getRed()
            green = color.getGreen()
            blue = color.getBlue()
            lineActor.GetProperty().SetColor(red/255., green/255., blue/255.)
            
            lineActor.SetMapper(polyMapper)
            renderer.AddActor(lineActor)

            # display control points on 3D vtk renderer
            self._createNodesActors(renderer)

            painter3Dintialized=True

    def displaySnake(self, renWinSize=(900, 900)):
        # Creates vtkRenderWindow and set size
        renWin = vtkRenderWindow()
        renWin.SetSize(renWinSize[0], renWinSize[1])
        
        # Creates interactive window
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        # Add actors to the renderer
        self._init3DRenderer(self.renderer)

        # Add renderer to the window
        renWin.AddRenderer(self.renderer)

        # Set Background and camera parameters
        self.renderer.SetBackground(1, 1, 1)
        self.renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
        self.renderer.GetActiveCamera().SetPosition(1, 0, 0)
        self.renderer.GetActiveCamera().SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Azimuth(20)
        self.renderer.GetActiveCamera().Elevation(30)
        self.renderer.GetActiveCamera().Dolly(1.2)
        self.renderer.ResetCameraClippingRange()
        
        iren.Initialize()
        renWin.Render()
        iren.Start()
        renWin.Render()
        
