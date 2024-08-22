import vtk
import numpy as np
import os
import sys
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
import pdb
from itertools import combinations
import networkx as nx
from numi_minimum_span import *
from get_attributes_to_cl import *
import vtk

def create_decreasing_spheres(coordinates, max_radius, min_radius):
    # Calculate the number of points
    num_points = len(coordinates)
    
    # Ensure there's more than one point to decrease the radius
    if num_points < 2:
        raise ValueError("There should be at least two points to create decreasing radii.")

    # Create a render window and interactor
    renderWindow = vtk.vtkRenderWindow()
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    renderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(renderer)
    
    for i, coord in enumerate(coordinates):
        # Calculate the radius linearly between max_radius and min_radius
        radius = max_radius - (max_radius - min_radius) * (i / (num_points - 1))
        
        # Create a sphere at the given coordinate
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(coord)
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(50)  # Resolution of the sphere
        sphere.SetPhiResolution(50)
        
        # Create a mapper and actor for the sphere
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Optionally, set color based on the sphere's index
        actor.GetProperty().SetColor(1.0 - i / num_points, 0.1, 0.5)  # Color gradient
        
        # Add the actor to the renderer
        renderer.AddActor(actor)
    
    # Set background color and size of the window
    renderer.SetBackground(0.1, 0.2, 0.4)  # Background color
    renderWindow.SetSize(800, 600)  # Window size
    
    # Render and start interaction
    renderWindow.Render()
    renderWindowInteractor.Start()

def get_coordinates_from_polyline(polyline):
    # Ensure the input is a vtkPolyLine
    if not isinstance(polyline, vtk.vtkPolyLine):
        raise TypeError("Input must be a vtkPolyLine object.")
    
    # Get the points associated with the polyline
    points = polyline.GetPoints()
    
    # Initialize an empty list to store the coordinates
    coordinates = []
    
    # Loop through each point in the polyline and get the coordinates
    for i in range(points.GetNumberOfPoints()):
        coord = points.GetPoint(i)  # Get the (x, y, z) tuple
        coordinates.append(coord)   # Add it to the list
    
    return coordinates


def create_decreasing_spheres_multiple_lists(lists_of_coordinates, max_radius, min_radius):
    # Create a render window and interactor
    renderWindow = vtk.vtkRenderWindow()
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    renderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(renderer)
    
    for coordinates in lists_of_coordinates:
        # Calculate the number of points in the current list
        num_points = len(coordinates)
        
        # Ensure there's more than one point to create decreasing radii
        if num_points < 2:
            raise ValueError("Each list should have at least two points to create decreasing radii.")
        
        for i, coord in enumerate(coordinates):
            # Calculate the radius linearly between max_radius and min_radius
            radius = max_radius - (max_radius - min_radius) * (i / (num_points - 1))
            
            # Create a sphere at the given coordinate
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(coord)
            sphere.SetRadius(radius)
            sphere.SetThetaResolution(50)  # Resolution of the sphere
            sphere.SetPhiResolution(50)
            
            # Create a mapper and actor for the sphere
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Optionally, set color based on the sphere's index
            color_value = 1.0 - (i / (num_points - 1))
            actor.GetProperty().SetColor(color_value, 0.1, 0.5)  # Color gradient
            
            # Add the actor to the renderer
            renderer.AddActor(actor)
    
    # Set background color and size of the window
    renderer.SetBackground(0.1, 0.2, 0.4)  # Background color
    renderWindow.SetSize(800, 600)  # Window size
    
    # Render and start interaction
    renderWindow.Render()
    renderWindowInteractor.Start()

def create_decreasing_spheres_with_ceneterline(coordinates, max_radius, min_radius,cent):
    # Calculate the number of points
    num_points = len(coordinates)
    
    # Ensure there's more than one point to decrease the radius
    if num_points < 2:
        raise ValueError("There should be at least two points to create decreasing radii.")

    # Create a render window and interactor
    renderWindow = vtk.vtkRenderWindow()
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    renderer = vtk.vtkRenderer()
    renderWindow.AddRenderer(renderer)
    
    for i, coord in enumerate(coordinates):
        # Calculate the radius linearly between max_radius and min_radius
        radius = max_radius - (max_radius - min_radius) * (i / (num_points - 1))
        
        # Create a sphere at the given coordinate
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(coord)
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(50)  # Resolution of the sphere
        sphere.SetPhiResolution(50)
        
        # Create a mapper and actor for the sphere
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Optionally, set color based on the sphere's index
        actor.GetProperty().SetColor(1.0 - i / num_points, 0.1, 0.5)  # Color gradient
        
        # Add the actor to the renderer
        renderer.AddActor(actor)
    
    # add the centerline to the renderer
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(cent)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    
    # Set background color and size of the window
    renderer.SetBackground(0.1, 0.2, 0.4)  # Background color
    renderWindow.SetSize(800, 600)  # Window size
    
    # Render and start interaction
    renderWindow.Render()
    renderWindowInteractor.Start()




class Bifurcation(object):
    """
    Simple class to track inlet and outlets of a bifurcation
    """
    def __init__(self):
        self.inflow = None
        self.outflow = []

    def add_inflow(self, i):
        """
        Add inflow branch i (can only be executed once)
        """
        if self.inflow is None:
            self.inflow = i
        elif self.inflow != i:
            raise ValueError('bifurcation already has inflow id ' + repr(self.inflow))

    def add_outflow(self, i):
        """
        Add outflow branch i
        """
        if i not in self.outflow:
            self.outflow += [i]


def get_connectivity(cent):
    """
    Extract the connectivity (which branches are connected to which bifurcation) from a centerline
    """
    # read arrays from centerline
    branch = v2n(cent.GetPointData().GetArray('BranchId'))
    bifurcation = v2n(cent.GetPointData().GetArray('BifurcationId'))
    bifurcation_list = np.unique(bifurcation).tolist()
    bifurcation_list.remove(-1)

    # get centerline connectivity: which branches are attached to which bifurcation?
    connectivity = {}
    for bf in bifurcation_list:
        connectivity[bf] = Bifurcation()

    # loop all cells
    for c in range(cent.GetNumberOfCells()):
        ele = cent.GetCell(c)
        point_ids = np.array([ele.GetPointIds().GetId(i) for i in range(ele.GetPointIds().GetNumberOfIds())])
        br_ids = branch[point_ids].tolist()
        print(br_ids)
        if br_ids[0] != br_ids[1]:
           print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # find cells that are at borders of bifurcations (two unique RegionIds)
        if np.unique(br_ids).shape[0] == 2:
            
            # should be one branch and one bifurcation
            assert -1 in br_ids, 'No bifurcation in cell'
            
            # local node ids of branch and bifurcation (0 or 1)
            i_bf_ele = br_ids.index(-1)
            i_br_ele = int(not i_bf_ele)

            # branch and bifurcation id
            bf = bifurcation[point_ids[i_bf_ele]] # retrieves the bifurcation ID associated with the bifurcation point in the current cell.
            br = branch[point_ids[i_br_ele]] # retrieves the branch ID associated with the branch point in the current cell.

            assert bf != -1, 'Multiple bifurcations in cell'
            assert br != -1, 'Multiple branches in cell'
            print("bifurcation ID that asso with this cell is ",bf)
            print("branch ID that asso with this cell is ", br)
            print(i_bf_ele, i_br_ele)
            coordinates_of_the_cell = [cent.GetPoint(point_ids[0]), cent.GetPoint(point_ids[1])]
            create_decreasing_spheres_with_ceneterline(coordinates_of_the_cell, 0.3, 0.2,cent)
            pdb.set_trace()
            # # branch node is upstream in cell?
            # if i_br_ele == 0:
            #     connectivity[bf].add_inflow(br)
            # else:
            #     connectivity[bf].add_outflow(br)

    # for bf, bifurcation in connectivity.items():
    #     assert len(bifurcation.outflow) >= 2, 'bifurcation ' + str(bf) + ' has less then two outlets'

    # return connectivity

def main():
    pd, gtclpd, dict_gt_attributes, dict_cell,surface_pd = setup()
    


    # create_decreasing_spheres(dict_cell[0][1],1,0.2)
    # create_decreasing_spheres(dict_cell[1][1],1,0.2)
    # create_decreasing_spheres(dict_cell[2][1],1,0.2)
    # create_decreasing_spheres(dict_cell[3][1],1,0.2)
    # create_decreasing_spheres(dict_cell[4][1],1,0.2)

    # line_coords = [dict_cell[i][1] for i in range(len(dict_cell))]
    # create_decreasing_spheres(dict_cell[1][1], 0.2, 0.05)
    # create_decreasing_spheres(dict_cell[2][1], 0.2, 0.05)
    # create_decreasing_spheres(dict_cell[3][1], 0.2, 0.05)
    # create_decreasing_spheres(dict_cell[4][1], 0.2, 0.05)

    #create_decreasing_spheres_multiple_lists(line_coords, 1,0.5)


    # get connectivity
    cent = read_polydata('c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\MIROS\\extracted_pred_centerlines.vtp')
    connectivity = get_connectivity(cent)
    # # renderer the first cell(2 points) in cent and the cent, render the two points as spheres, the first one has radius 0.5, the second one has radius 0.2
    # first_cell = cent.GetCell(0)
    # first_cell_points = [first_cell.GetPointId(0), first_cell.GetPointId(1)]
    # first_cell_coords = [cent.GetPoint(i) for i in first_cell_points]
    # # get coordinates of the entire centerline cell by cell
    # # get all cells
    # master_list = []
    # for i in range(cent.GetNumberOfCells()):
    #     cell = cent.GetCell(i)
    #     cell_points = [cell.GetPointId(0), cell.GetPointId(1)]
    #     cell_coords = [cent.GetPoint(i) for i in cell_points]
    #     master_list.append(cell_coords)
    # # remove all the brackets in the master_list
    # master_list = [item for sublist in master_list for item in sublist]
    # # in master_list, if nearby points are the same, remove one of them
    # master_list = list(dict.fromkeys(master_list))

    
    # create_decreasing_spheres(master_list, 1, 0.01)
    

main()