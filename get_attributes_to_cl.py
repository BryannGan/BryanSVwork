import vtk
import numpy as np
import os
import sys
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
import pdb
from itertools import combinations
import networkx as nx
###
# this code is to add attributes to a centerline polydata
###





def read_polydata(fname):
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()
    return reader.GetOutput()


def create_spheres_at_coordinates(coords, radius=1.0):
    """
    Create a vtkPolyData containing a sphere at each provided coordinate.

    Parameters:
    coords (list of tuples): List of coordinates where spheres should be placed.
    radius (float): Radius of each sphere.

    Returns:
    vtkPolyData: A single vtkPolyData containing all spheres.
    """
    # Create an object to append all sphere polydata objects
    append_filter = vtk.vtkAppendPolyData()

    # Iterate over all coordinates and create a sphere at each position
    for coord in coords:
        # Create a sphere source
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*coord)  # Unpack the tuple into x, y, z coordinates
        sphere.SetRadius(radius)
        sphere.SetPhiResolution(16)  # Set the resolution of the sphere
        sphere.SetThetaResolution(16)
        sphere.Update()  # Important to update each sphere

        # Append the current sphere's polydata to the append filter
        append_filter.AddInputData(sphere.GetOutput())

    # Combine all spheres into a single vtkPolyData object
    append_filter.Update()

    # Return the combined polydata
    return append_filter.GetOutput()

def setup():
    pd_path = 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\centerline_fm_0176_0000.vtp'
    pd = read_polydata(pd_path)
    ######## cross check ##############################
    gtcl_path = 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\pipeline_testing_master\\test_cl_attributes\\extracted_pred_centerlines.vtp'
    gtclpd = read_polydata(gtcl_path)
    # get BranchId” “BifurcationId” “CenterlineId” “CenterlineSectionArea" “CenterlineSectionNormal"“GlobalNodeId” “Path” and save in dict
    dict_gt_attributes = {}
    dict_gt_attributes['BranchId'] = v2n(gtclpd.GetPointData().GetArray('BranchId'))
    dict_gt_attributes['BifurcationId'] = v2n(gtclpd.GetPointData().GetArray('BifurcationId'))
    dict_gt_attributes['CenterlineId'] = v2n(gtclpd.GetPointData().GetArray('CenterlineId'))
    dict_gt_attributes['CenterlineSectionArea'] = v2n(gtclpd.GetPointData().GetArray('CenterlineSectionArea'))
    dict_gt_attributes['CenterlineSectionNormal'] = v2n(gtclpd.GetPointData().GetArray('CenterlineSectionNormal'))
    dict_gt_attributes['GlobalNodeId'] = v2n(gtclpd.GetPointData().GetArray('GlobalNodeId'))
    dict_gt_attributes['Path'] = v2n(gtclpd.GetPointData().GetArray('Path'))
    ##################################################
    # all length = 1554 - number of points in the cl



    # get lines and save each cell
    dict_cell = {}
    for i in range(pd.GetNumberOfCells()):
        # key = i; value = 
        cell = pd.GetCell(i)
        num_points = cell.GetNumberOfPoints()

        # Get the coordinates of each point in the cell
        points = pd.GetPoints()
        init_pt_array = [0]*num_points

        for j in range(num_points):
            point_id = cell.GetPointId(j)
            point = points.GetPoint(point_id)
            init_pt_array[j] = point

        # reverse the coordinate ordering due to Eikonal equation
        init_pt_array = init_pt_array[::-1]

        # save the cell [vtp cell, pt coordinates]
        dict_cell[i] = []
        dict_cell[i].append(cell)
        #dict_cell[i].append(points)
        dict_cell[i].append(init_pt_array)
        
    
    return pd, gtclpd, dict_gt_attributes, dict_cell
 
def create_polydata_from_coordinates(coords):
    # Create a vtkPoints object and insert the coordinates
    points = vtk.vtkPoints()
    for coord in coords:
        points.InsertNextPoint(coord)
    
    # Create a vtkPolyLine which holds the connectivity information
    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(len(coords))
    for i in range(len(coords)):
        polyLine.GetPointIds().SetId(i, i)
    
    # Create a vtkCellArray to store the lines in and add the polyLine to it
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyLine)
    
    # Create a vtkPolyData and add the points and lines to it
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(cells)
    
    return polyData

def generate_pairs(n):
    # Create a list of numbers from 0 to n-1
    numbers = list(range(0, n))
    # Generate all pairs
    pairs = combinations(numbers, 2)
    # Format and print the pairs
    result = [f"{a:d}{b:d}" for a, b in pairs]
    return result

def findclosestpoint(polydata, refpoint):
    #"""Returns coordinate of point on polydata closest to refpoint."""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()
    
    id = locator.FindClosestPoint(refpoint)
    coord = polydata.GetPoint(id)
    return coord

def get_bifurcation_pts(dict_cell):
    #distance between two stepping points is around 0.593

    # for i in dict_cell:
    #     print(dict_cell[i][2][0])
        #
        #print(np.linalg.norm(np.array(dict_cell[i][2][0]) - np.array(dict_cell[i][2][1])))   

    result = [] 
    cell_num = len(dict_cell)
    pairs = generate_pairs(cell_num)
    # hold the first and iterate the second index
    for i in pairs: #['01', '02', '03', '04', '05', '06', '12', '13', '14', ...]
        print(i)
        cell1_id = int(i[0])
        cell2_id = int(i[1])

        cell1_pd = dict_cell[cell1_id][0]  # vtkPolyLine
        cell1_pd = create_polydata_from_coordinates(dict_cell[cell1_id][1]) # vtkPolyData

        last_pt = 0
        for refpoint in dict_cell[cell2_id][1]:
            #print(f"Reference Point: {refpoint}")
            closest_point = findclosestpoint(cell1_pd, refpoint)
            #print(f"Closest Point: {closest_point}")
            distance = np.linalg.norm(np.array(refpoint) - np.array(closest_point))
            #print(f"Distance: {distance}")
            if distance > 0.1:
                result.append(last_pt)
                break
            last_pt = refpoint

    return result

import numpy as np
from scipy.interpolate import CubicSpline
import vtk

def subdivide_coordinates_with_cubic_spline(coords, subdivision_factor=10):
    # Convert list of tuples to a numpy array for easier manipulation
    coords_array = np.array(coords)
    
    # Extract x, y, z coordinates
    x = coords_array[:, 0]
    y = coords_array[:, 1]
    z = coords_array[:, 2]
    
    # Create parameter t along the curve
    t = np.linspace(0, 1, len(coords))
    
    # Create cubic spline interpolations for each dimension
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    cs_z = CubicSpline(t, z)
    
    # Generate a finer parameterization for the curve
    t_fine = np.linspace(0, 1, len(coords) * subdivision_factor)
    
    # Evaluate the splines on the finer parameterization
    x_fine = cs_x(t_fine)
    y_fine = cs_y(t_fine)
    z_fine = cs_z(t_fine)
    
    # Combine the results back into a list of tuples
    subdivided_coords = list(zip(x_fine, y_fine, z_fine))
    return subdivided_coords


import numpy as np
import vtk
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# def create_mst_polydata(coords):
#     # Convert coordinates to a numpy array if not already
#     points_array = np.array(coords)
    
#     # Calculate the distance matrix
#     dist_matrix = distance_matrix(points_array, points_array)
    
#     # Compute the Minimum Spanning Tree (MST)
#     mst = minimum_spanning_tree(dist_matrix)
    
#     # Extract the MST edges
#     mst_edges = mst.toarray().astype(float)
#     np.fill_diagonal(mst_edges, 0)
    
#     # Create vtkPoints and add all points
#     vtk_points = vtk.vtkPoints()
#     for coord in coords:
#         vtk_points.InsertNextPoint(coord)
    
#     # Create the lines based on the MST edges
#     vtk_lines = vtk.vtkCellArray()
#     for i in range(len(coords)):
#         for j in range(i + 1, len(coords)):
#             if mst_edges[i, j] > 0:
#                 line = vtk.vtkLine()
#                 line.GetPointIds().SetId(0, i)
#                 line.GetPointIds().SetId(1, j)
#                 vtk_lines.InsertNextCell(line)
    
#     # Create the PolyData
#     poly_data = vtk.vtkPolyData()
#     poly_data.SetPoints(vtk_points)
#     poly_data.SetLines(vtk_lines)
    
#     return poly_data

def create_mst_polydata(coords, connection_threshold=1000):
    points_array = np.array(coords)
    dist_matrix = distance_matrix(points_array, points_array)
    print(dist_matrix)
    # Apply a threshold to consider close enough points as directly connected
    dist_matrix[dist_matrix > connection_threshold] = np.inf

    mst = minimum_spanning_tree(dist_matrix)
    mst_edges = mst.toarray().astype(float)
    np.fill_diagonal(mst_edges, 0)

    vtk_points = vtk.vtkPoints()
    for coord in coords:
        vtk_points.InsertNextPoint(coord)

    vtk_lines = vtk.vtkCellArray()
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if mst_edges[i, j] > 0:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, j)
                vtk_lines.InsertNextCell(line)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)
    
    return poly_data

def process_pts(dict_cell, bfpt):
    #
    # this function gets all the points in the centerline and eliminates duplicates
    #
    all_pts = []
    for i in range(len(dict_cell)):
        pts = dict_cell[i][1][:]
        #pts = subdivide_coordinates_with_cubic_spline(pts)
        all_pts.extend(pts)

    # now get rid of duplicates
    all_pts = list(set(all_pts))
    return all_pts


def merge_coordinate_lists(coord_lists):
    """
    Merges lists of coordinates if they share at least one coordinate.

    Parameters:
    coord_lists (list of lists of tuples): List containing multiple lists of coordinates.

    Returns:
    list of lists: Merged lists of coordinates.
    """
    # Create a graph
    G = nx.Graph()

    # Add each list of coordinates as a separate node in the graph
    for index, coord_list in enumerate(coord_lists):
        G.add_node(index, coords=set(coord_list))

    # Compare each list with every other list to check for common coordinates
    for i in range(len(coord_lists)):
        for j in range(i + 1, len(coord_lists)):
            # If there's an intersection between the sets of coordinates, add an edge
            if set(coord_lists[i]) & set(coord_lists[j]):
                G.add_edge(i, j)

    # Find connected components of the graph
    merged_lists = []
    for component in nx.connected_components(G):
        combined_set = set()
        for index in component:
            combined_set.update(coord_lists[index])
        merged_lists.append(list(combined_set))

    return merged_lists

def get_bifurcation_junctions(all_pts, bfpt,radius=1):
    """
    this function gets the bifurcation junctions
    in other words, it gets the BifurcationId 
    """
    # determine number of junctions
    # first see if bfpts are close to each other, if so, they are the same junction
    junctions_master_lsts = []
    for bfp in bfpt:
        junction_local = []
        for pts in all_pts:
            if np.linalg.norm(np.array(bfp) - np.array(pts)) < radius:
                junction_local.append(pts)
        junctions_master_lsts.append(junction_local)
    # merge the junctions
    junctions = merge_coordinate_lists(junctions_master_lsts)
    # make sure each junction has no dupilcates
    junctions = [list(set(junction)) for junction in junctions]
    # get all coordinates in junctions
    junctions_pts = []
    for junction in junctions:
        junctions_pts.extend(junction)

    return junctions, junctions_pts

def write_polydata(polydata, filename):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

    



if __name__ == '__main__':
    pd, gtclpd, dict_gt_attributes, dict_cell = setup()
    bfpt = get_bifurcation_pts(dict_cell)
    sphere = create_spheres_at_coordinates(bfpt, radius=0.3)
    
    ### process points
    all_pts = process_pts(dict_cell, bfpt)
    junction, junction_pts = get_bifurcation_junctions(all_pts, bfpt)
    junction_pts_sphere = create_spheres_at_coordinates(junction_pts, radius=0.2)
    write_polydata(junction_pts_sphere, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\junction_pts.vtp')

    # mst_polydata = create_mst_polydata(all_pts)
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName('c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\mst.vtp')
    # writer.SetInputData(mst_polydata)
    # writer.Update()
    # writer.Write()
