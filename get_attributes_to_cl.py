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




###
# this code is to add attributes to a centerline polydata
# note: each point is a tuple! eg. [(x1,y1,z1), (x2,y2,z2), ...]
###

def numi_MST(input_polydata):
    # Compute pairwise distances between points
    distances = compute_distances(input_polydata)

    # Compute minimum spanning tree
    mst = compute_mst(distances)

    # Extract edges from minimum spanning tree
    edges = np.argwhere(mst.toarray())

    # Create polydata from edges
    output_polydata = create_polydata_from_edges(edges, input_polydata.GetPoints())

    return output_polydata


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
    surface_pd = read_polydata('c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\processed_0176_surface.vtp')
    surface_pd = bryan_fillholes(surface_pd)
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
    dict_gt_attributes['all_pts'] = v2n(gtclpd.GetPoints().GetData())
    ##################################################
    # all length = 1554 - number of points in the cl

    # get lines and save each cell
    dict_cell = {}  # key = cell number; value = [vtkPolyLine, [pt1, pt2, ...]]
    distance_lst = [[0, 0, 0] for _ in range(pd.GetNumberOfCells())] # [distance, vtkPolyLine, [pt1, pt2, ...]]
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
            # calculated cumulative distance for each line
            if j > 0:
                distance_lst[i][0] += np.linalg.norm(np.array(init_pt_array[j]) - np.array(init_pt_array[j-1]))

        # reverse the coordinate ordering due to Eikonal equation
        init_pt_array = init_pt_array[::-1]
        distance_lst[i][1] = cell
        distance_lst[i][2] = init_pt_array
    
    # sort the distance_lst based on the first index
    # re-order dict_cell content base on distance (from longest to shortest)
    distance_lst = sorted(distance_lst, key=lambda x: x[0], reverse=True)
    


    # dict_cell[0] has the longest distance
    for i in range(len(distance_lst)): 
        # save the cell [vtp cell, pt coordinates]
        dict_cell[i] = []
        dict_cell[i].append(distance_lst[i][1])
        #dict_cell[i].append(points)
        dict_cell[i].append(distance_lst[i][2])
    return pd, gtclpd, dict_gt_attributes, dict_cell,surface_pd
 
# def create_polydata_from_coordinates(coords):
#     # Create a vtkPoints object and insert the coordinates
#     points = vtk.vtkPoints()
#     for coord in coords:
#         points.InsertNextPoint(coord)
    
#     # Create a vtkPolyLine which holds the connectivity information
#     polyLine = vtk.vtkPolyLine()
#     polyLine.GetPointIds().SetNumberOfIds(len(coords))
#     for i in range(len(coords)):
#         polyLine.GetPointIds().SetId(i, i)
    
#     # Create a vtkCellArray to store the lines in and add the polyLine to it
#     cells = vtk.vtkCellArray()
#     cells.InsertNextCell(polyLine)
    
#     # Create a vtkPolyData and add the points and lines to it
#     polyData = vtk.vtkPolyData()
#     polyData.SetPoints(points)
#     polyData.SetLines(cells)
    
#     return polyData

def create_polydata_from_coordinates(coords):
    # Create a vtkPoints object and insert the coordinates
    points = vtk.vtkPoints()
    for coord in coords:
        points.InsertNextPoint(coord)  
    # Create a vtkPolyData and add the points and lines to it
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
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


def process_pts(dict_cell):
    #
    # this function gets all the points of the centerlines
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

def get_BifurcationId(all_pts,junction):
    """
    assume all_pts is in order of "global node id"
    """
    # create np array to write the BifurcationId
    BifurcationId = np.ones(len(all_pts),dtype=np.int64)*-1
    for i in range(len(junction)):
        for pt in all_pts:
            if pt in junction[i]:
                BifurcationId[all_pts.index(pt)] = i
    return BifurcationId

def get_CenterlineId(mst_pts,dict_cell,mst_polydata):
    length_of_centerlineId= len(mst_pts)
    length_of_each_index = len(dict_cell.keys())
    mapper = list(enumerate(mst_pts))
    CenterlineId = np.full((length_of_centerlineId,length_of_each_index),-1,dtype=np.int64)
   
    for i in range(length_of_each_index): # iterate through each point each cell(each centerlne) and find closet point in mst_polydata
        for j in range(len(dict_cell[i][1])):
            coord = findclosestpoint(mst_polydata,dict_cell[i][1][j])
            # find coord in mapper
            for k in range(len(mapper)):
                if coord == mapper[k][1]:
                    CenterlineId[k][i] = 1
        # print progress: done with cell i
        print(f"Done with centerline {i}")
    
    # make all -1 to 0
    CenterlineId[CenterlineId == -1] = 0
    return CenterlineId


        
def get_connectivity(mst_polydata):
    # get the connectivity of the mst_polydata
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mst_polydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    return connectivity.GetOutput()

def find_component_label(point, polydata):
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(polydata)
    cell_locator.BuildLocator()
    
    closest_point = [0.0, 0.0, 0.0]
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)
    
    cell_locator.FindClosestPoint(point, closest_point, cell_id, sub_id, dist2)
    
    return polydata.GetPointData().GetArray("RegionId").GetTuple1(cell_id)



def add_attributes(attribute_name,list_to_add,pd):
    # add the list to the polydata
    list_to_add = n2v(list_to_add)
    list_to_add.SetName(attribute_name)
    pd.GetPointData().AddArray(list_to_add)

def smooth_polydata(poly, iteration=25, boundary=False, feature=False, smoothingFactor=0.):
    """
    This function smooths a vtk polydata
    Args:
        poly: vtk polydata to smooth
        boundary: boundary smooth bool
    Returns:
        smoothed: smoothed vtk polydata
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetPassBand(pow(10., -4. * smoothingFactor))
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    smoothed = smoother.GetOutput()


    return smoothed

def get_connected_point_id(mst_polydata,mst_pts):
    connected_pointId_pair = []
    num_cell = mst_polydata.GetNumberOfCells()
    for i in range(num_cell):
        cell = mst_polydata.GetCell(i)
        num_points = cell.GetNumberOfPoints()
        pair = [0,0]
        for j in range(num_points):
            pair[j] = cell.GetPointId(j)
        connected_pointId_pair.append(pair)
    return connected_pointId_pair

def get_connectivity(connected_pointId_pair):
    # given a list like this [[0, 36], [1, 79], [1, 164], [3, 99], [3, 159], [4, 82], [6, 179], [7, 158], [8, 118], [8, 163], [9, 169], [11, 81], [12, 127], [12, 161], [13, 132], [15, 13], [16, 26], [17, 52], [19, 83], [19, 129], [20, 56], [20, 94], [21, 64], [22, 154], [23, 49], [23, 196], [24, 22], [25, 51], [27, 141], [28, 61], [28, 65], [29, 204], [31, 105], [32, 69], [32, 106], [33, 117], [34, 20], [34, 113], [35, 78], [37, 79], [38, 176], [39, 196], [42, 67], [42, 136], [43, 35], [45, 175]
    # return what number occured once, twice, etc
    # return a dictionary with keys being the number of repetition and values being the number
    dict = {}
    for pair in connected_pointId_pair: 
        for pt in pair:
            if pt in dict:
                dict[pt] += 1
            else:
                dict[pt] = 1
    # make the keys go from low to high
    dict = dict.items()
    dict = sorted(dict)
    # dict = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (12, 2), (13, 2), (14, 2), (15, 2), (16, 2), (17, 2), (18, 2), (19, 2), (20, 3), (21, 2), (22, 2), (23, 3), (24, 2), (25, 2), (26, 1), (27, 2), (28, 3), (29, 2), (30, 2), (31, 2), (32, 2), (33, 2), (34, 2), (35, 2), (36, 2), (37, 2), (38, 2), (39, 2), (40, 2), (41, 2), (42, 2), (43, 2), (44, 2), (45, 2), (46, 2), (47, 2), (48, 2), (49, 2), (50, 2), (51, 2), (52, 2), (53, 3), (54, 1), (55, 2), (56, 2), (57, 2), (58, 2), (59, 2), (60, 2), (61, 2), (62, 2), (63, 2), (64, 2), (65, 2), (66, 2), (67, 2), (68, 2), (69, 2), (70, 2), (71, 2), (72, 2), (73, 2), (74, 2), (75, 2), (76, 2), (77, 2), (78, 2), (79, 2), (80, 2), (81, 2), (82, 2), (83, 2), (84, 2), (85, 2), (86, 2), (87, 2), (88, 2), (89, 2), (90, 2), (91, 2), (92, 2), (93, 2), (94, 2), (95, 2), (96, 1), (97, 2), (98, 2), (99, 2), (100, 2), (101, 2), (102, 2), (103, 3), (104, 2), (105, 2), (106, 2), (107, 2), (108, 2), (109, 2), (110, 2), (111, 2), (112, 2), (113, 2), (114, 1), (115, 2), (116, 1), (117, 2), (118, 2), (119, 2), (120, 2), (121, 2), (122, 2), (123, 2), (124, 2), (125, 2), (126, 1), (127, 2), (128, 2), (129, 2), (130, 2), (131, 2), (132, 2), (133, 2), (134, 2), (135, 2), (136, 2), (137, 2), (138, 2), (139, 2), (140, 2), (141, 2), (142, 2), (143, 2), (144, 2), (145, 2), (146, 2), (147, 2), (148, 2), (149, 2), (150, 2), (151, 2), (152, 2), (153, 2), (154, 2), (155, 2), (156, 2), (157, 2), (158, 2), (159, 2), (160, 2), (161, 2), (162, 2), (163, 2), (164, 2), (165, 2), (166, 2), (167, 2), (168, 2), (169, 2), (170, 2), (171, 2), (172, 2), (173, 2), (174, 2), (175, 2), (176, 2), (177, 2), (178, 2), (179, 2), (180, 2), (181, 2), (182, 2), (183, 2), (184, 2), (185, 2), (186, 2), (187, 2), (188, 2), (189, 2), (190, 1), (191, 2), (192, 2), (193, 2), (194, 2), (195, 2), (196, 2), (197, 2), (198, 2), (199, 2), (200, 2), (201, 2), (202, 2), (203, 2), (204, 2), (205, 2)]
    # find largest number in the second element
    # make tuple in dict into list
    for i in range(len(dict)):
        dict[i] = list(dict[i])
    # find the largest number in the second element
    # pdb.set_trace()
    # max_repetition = max([pair[1] for pair in dict])
    # create a dictionary with keys being the number of repetition and values being the number
    # repetition = 1 means that it is an endpoint
    # start from an end point, use connected_pointId_pair to construct a "linked list of coordinates"
    # pt id: [1,2,3,[3,4,5,[5,8,9]],[3,6,7]] 


    return dict
    
def get_BranchId(mst_pts,junction_pts,junction,dict_cell,mst_polydata):
        BranchId = np.ones(len(mst_pts),dtype=np.int64)
        BranchId = BranchId*-2
        #start tracing from centerlineId 0->n
        # find cloest point via dict_cell
        length_of_each_index = len(dict_cell.keys())
        if dict_cell[0][1][0] in junction_pts:  # see if the starting point of all centerline is a junction
            index = -1 # becuase moving out of junction, index increases by 1 --> the first branch id is 0
        else: 
            index = 0 # when not moving out of a junction, BranchId index starts with 0
        junction_index = -1
        last_pt_is_in_junction = False
        last_pt_is_an_endpt = False
        for i in range(length_of_each_index): # iterate through each point each cell(each centerlne) and find closet point in mst_polydata
                for j in range(len(dict_cell[i][1])):
                    coord = findclosestpoint(mst_polydata,dict_cell[i][1][j])
                    # see how condition/ what to write
                    if coord in junction_pts: # if its in a bifurcation juntion
                        # find coord in mst_pts
                        for k in range(len(mst_pts)):
                            if coord == mst_pts[k] and BranchId[k]==-2:
                                print(junction_index)
                                BranchId[k] = junction_index
                        last_pt_is_in_junction = True
                    elif coord in endpts_coord: # if its  an endpt
                        for k in range(len(mst_pts)):
                            if coord == mst_pts[k] and BranchId[k]==-2:
                                if last_pt_is_in_junction == True:
                                    index += 1
                                print(index)
                                BranchId[k] = index
                        last_pt_is_an_endpt = True
                    else:
                        # write
                        for k in range(len(mst_pts)):
                            if coord == mst_pts[k] and BranchId[k]==-2:
                                if last_pt_is_in_junction == True or last_pt_is_an_endpt == True:
                                    index += 1
                                    last_pt_is_in_junction = False
                                    last_pt_is_an_endpt = False
                                print(index)
                                BranchId[k] = index
                        
        
        return BranchId

def get_BranchId_and_Path(mst_pts,junction_pts,junction,dict_cell,mst_polydata):
        """
        added get_path to function get_BranchId
        """
        ### variables to calculate Path ###
        Path = np.ones(len(mst_pts))
        Path = Path*-5
        last_pt_coord = 0
        Path_to_write = 0
        Path_at_junction = 0
        ##################################

        BranchId = np.ones(len(mst_pts),dtype=np.int64)
        BranchId = BranchId*-2
        #start tracing from centerlineId 0->n
        # find cloest point via dict_cell
        length_of_each_index = len(dict_cell.keys())
        if dict_cell[0][1][0] in junction_pts:  # see if the starting point of all centerline is a junction
            index = -1 # becuase moving out of junction, index increases by 1 --> the first branch id is 0
        else: 
            index = 0 # when not moving out of a junction, BranchId index starts with 0
        junction_index = -1
        last_pt_is_in_junction = False
        last_pt_is_an_endpt = False
        for i in range(length_of_each_index): # iterate over each each cell(each centerlne) and find closet point in mst_polydata
                for j in range(len(dict_cell[i][1])): # iterate over each point of the centerline
                    
                    
                    coord = findclosestpoint(mst_polydata,dict_cell[i][1][j])
                    if j == 0: 
                        last_pt_coord = coord #for calculating path at the first point of the centerline

                    ### see how condition/ what to write
                    # if its in a bifurcation juntion
                    if coord in junction_pts:                         
                        # get branch id
                        # find coord in mst_pts
                        for k in range(len(mst_pts)):
                            if coord == mst_pts[k] and BranchId[k]==-2:
                                print(junction_index)
                                BranchId[k] = junction_index # -1
                        last_pt_is_in_junction = True

                        # get Path
                        for k in range(len(mst_pts)):
                            if coord == mst_pts[k] and Path[k]==-5:
                                Path_to_write = 0                                
                                Path[k] = Path_to_write
                                last_pt_coord = coord

                    ### if its an endpt
                    elif coord in endpts_coord: 
                        # get BranchId                        
                        for k in range(len(mst_pts)):
                            if coord == mst_pts[k] and BranchId[k]==-2:
                                if last_pt_is_in_junction == True:
                                    index += 1
                                print(index)
                                BranchId[k] = index
                        last_pt_is_an_endpt = True

                        # get Path
                        for k in range(len(mst_pts)):
                            if coord == mst_pts[k] and Path[k]==-5:
                                Path_to_write += np.linalg.norm(np.array(coord) - np.array(last_pt_coord))
                                # accumulate path
                                Path[k] = Path_to_write
                                last_pt_coord = coord
                                Path_to_write = 0 # reset Path_to_write bc will start from 0 at the next branch


                    else:
                        # write branchId and path the same time b/c find branchid would edit last_pt_is_in_junction and last_pt_is_an_endpt
                        for k in range(len(mst_pts)):
                            if coord == mst_pts[k] and BranchId[k]==-2:
                                if last_pt_is_in_junction == True or last_pt_is_an_endpt == True:
                                    index += 1
                                    last_pt_is_in_junction = False
                                    last_pt_is_an_endpt = False                                    
                                print(index)
                                BranchId[k] = index
                                
                        # get Path
                        for k in range(len(mst_pts)):
                            if coord == mst_pts[k] and Path[k]==-5:                        
                                Path_to_write += np.linalg.norm(np.array(coord) - np.array(last_pt_coord))
                                # accumulate path
                                Path[k] = Path_to_write
                                last_pt_coord = coord       
        
        return BranchId, Path


def get_CenterlineSectionNormal(dict_cell,mst_pts):
    # get the normal of each point
    CenterlineSectionNormal = np.zeros((len(mst_pts),3))
    for i in range(len(dict_cell)):
        pts = dict_cell[i][1]   
        for j in range(len(pts)):
                if j == len(pts)-1:
                    CenterlineSectionNormal[mst_pts.index(pts[j])] = CenterlineSectionNormal[mst_pts.index(pts[j-1])]
                else: 
                    CenterlineSectionNormal[mst_pts.index(pts[j])] = np.array(pts[j+1]) - np.array(pts[j])
    for i in range(len(CenterlineSectionNormal)):
        CenterlineSectionNormal[i] = CenterlineSectionNormal[i]/ np.linalg.norm(CenterlineSectionNormal[i])
    
    return CenterlineSectionNormal
    
def bryan_fillholes(pd):
    # Fill the holes
    fill = vtk.vtkFillHolesFilter()
    fill.SetInputData(pd)
    fill.SetHoleSize(100000.0)
    fill.Update()
    return fill.GetOutput()

if __name__ == '__main__':
    pd, gtclpd, dict_gt_attributes, dict_cell,surface_pd = setup()
    
    def subdivide_update_dict_cell(dict_cell):
        
        # subdivide
        for i in range(len(dict_cell)):
            pts = dict_cell[i][1]
            #subdivide
            pts = subdivide_coordinates_with_cubic_spline(pts,2)
            dict_cell[i][1] = pts
            # create polydata using new pts
            polydata = create_polydata_from_coordinates(pts)
            # update dict_cell
            dict_cell[i][0] = polydata

        return dict_cell
    
    def delete_close_pts_update_dict_cell(dict_cell,cluster_radius=0.15):
        # delete close points
        all_pts = process_pts(dict_cell)
        for i in range(len(all_pts)):  
            for j in range(len(all_pts)):
                if np.linalg.norm(np.array(all_pts[i]) - np.array(all_pts[j])) < cluster_radius and i != j:
                    # make them 'X' so that they can be deleted
                    all_pts[j] = 0
        # delete 'X'
        all_pts = [pt for pt in all_pts if pt != 0]

        # update dict_cell coords
        for i in range(len(dict_cell)):
            for j in range(len(dict_cell[i][1])):
                if dict_cell[i][1][j] not in all_pts:
                    dict_cell[i][1][j] = 0
            dict_cell[i][1] = [pt for pt in dict_cell[i][1] if pt != 0]
        
        # update dict_cell polydata
        for i in range(len(dict_cell)):
            dict_cell[i][0] = create_polydata_from_coordinates(dict_cell[i][1])

        return dict_cell
    
    
    #dict_cell = subdivide_update_dict_cell(dict_cell)
    #dict_cell = delete_close_pts_update_dict_cell(dict_cell,0.15)

    bfpt = get_bifurcation_pts(dict_cell)
    sphere = create_spheres_at_coordinates(bfpt, radius=0.3)
    #write_polydata(sphere, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\bifurcation_pts.vtp')
    
    ### process points
    all_pts = process_pts(dict_cell)
    junction, junction_pts = get_bifurcation_junctions(all_pts, bfpt)

    # get all_pts into a polydata
    all_pts_pd = create_polydata_from_coordinates(all_pts)
    # all_pts_sphere = create_spheres_at_coordinates(all_pts, radius=0.2)
    # write_polydata(all_pts_sphere, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\all_pts.vtp')
    
    # MST
    mst_polydata = numi_MST(all_pts_pd)
    # get points based on id
    mst_pts = v2n(mst_polydata.GetPoints().GetData())
    mst_pts = mst_pts.tolist()
    mst_pts = [tuple(pt) for pt in mst_pts]
    #write_polydata(mst_polydata, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\mst.vtp')

    # #MST not working at the moment, so lets use gt points
    # # make np array to list then tuple
    # gtpts = dict_gt_attributes['all_pts'].tolist()
    # gtpts = [tuple(pt) for pt in gtpts]
    junction, junction_pts = get_bifurcation_junctions(mst_pts, bfpt, radius=1)
    #junction_pts_sphere = create_spheres_at_coordinates(junction_pts, radius=0.2)
    #write_polydata(junction_pts_sphere, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\junction_pts.vtp')

    # get BifurcationId
    BifurcationId = get_BifurcationId(mst_pts, junction)
    # add BifurcationId to mst_polydata
    add_attributes('BifurcationId',BifurcationId,mst_polydata)
    #write_polydata(mst_polydata, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\mst_added_bifurcationId.vtp')

    # get GlobalNodeId
    GlobalNodeId = np.arange(len(mst_pts))
    # add GlobalNodeId to mst_polydata
    add_attributes('GlobalNodeId',GlobalNodeId,mst_polydata)
    #write_polydata(mst_polydata, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\mst_added_bifurcationId_GlobalNodeId.vtp')

    # get CenterlineId
    CenterlineId = get_CenterlineId(mst_pts,dict_cell,mst_polydata)
    add_attributes('CenterlineId',CenterlineId,mst_polydata)
    #write_polydata(mst_polydata, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\mst_added_bifurcationId_GlobalNodeId_CenterlineId_ver2.vtp')
    
    # get BranchId ------

        # get end pts
    pairs = get_connected_point_id(mst_polydata,mst_pts) #[[0, 36], [1, 79], [1, 164], [3, 99], [3, 159]] paris of global nodel ID
    count_pt_appearance = get_connectivity(pairs) #[[0, 2], [1, 2], [2, 2], [3, 2]] -> [global nodel id, time appeared (1 means its endpt, >2 means its a junction)]
    endpts_id = [count_pt_appearance[i][0] for i in range(len(count_pt_appearance)) if count_pt_appearance[i][1] == 1]
    endpts_coord = [mst_pts[i] for i in endpts_id]
    endpts_sphere = create_spheres_at_coordinates(endpts_coord, radius=0.2)
    write_polydata(endpts_sphere, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\endpts.vtp')

    
    BranchId = get_BranchId(mst_pts,junction_pts,junction,dict_cell,mst_polydata)
    add_attributes('BranchId',BranchId,mst_polydata) # FREAKING WORKED!
    #write_polydata(mst_polydata, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\mst_added_bifurcationId_GlobalNodeId_CenterlineId_BranchId.vtp')
    
    BranchId2,Path = get_BranchId_and_Path(mst_pts,junction_pts,junction,dict_cell,mst_polydata)
    add_attributes('Path',Path,mst_polydata) 
   

    CenterlineSectionNormal = get_CenterlineSectionNormal(dict_cell,mst_pts)
    add_attributes('CenterlineSectionNormal',CenterlineSectionNormal,mst_polydata)
    write_polydata(mst_polydata, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\mst_added_bifurcationId_GlobalNodeId_CenterlineId_BranchId_Path_CenterlineSectionNormal.vtp')



    pdb.set_trace()
    def get_CenterlineSectionArea(mst_pts,CenterlineSectionNormal,surface_pd):
        """
        create a plane/slice at each point base on CenterlineSectionNormal
        use the surface clipped by the plane to get enclosed area(s)
        only include the area that has the orgin of the plane
        there should be a value at each point
        """
        def calculate_polygon_area(points):
            """
            Calculate the area of a polygon using the shoelace formula.
            Points should be a 2D array of shape (N, 2), where N is the number of points.
            """
            n = len(points)
            area = 0.0
            for i in range(n):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % n]
                area += x1 * y2 - x2 * y1
            return abs(area) / 2.0

        # get the cross sectional area of the polydata at each point
        CenterlineSectionArea = np.zeros(len(mst_pts))
        for i in range(len(mst_pts)):
            # create a plane at each point
            plane = vtk.vtkPlane()
            plane.SetOrigin(mst_pts[i])
            plane.SetNormal(CenterlineSectionNormal[i])

            # use vtkcutter to get the clipped surface
            cutter = vtk.vtkCutter()
            cutter.SetCutFunction(plane)
            cutter.SetInputData(surface_pd)
            cutter.Update()
            slicePolyData = cutter.GetOutput()

            # Extract points from the slice
            points = slicePolyData.GetPoints()
            if points is None:
                continue
            num_points = points.GetNumberOfPoints()
            polygon_points = []

            for j in range(num_points):
                p = points.GetPoint(j)
                polygon_points.append((p[1], p[2]))  # Projecting onto the YZ plane

            # Convert to numpy array for easy manipulation
            polygon_points = np.array(polygon_points)

            # Calculate the area using the shoelace formula
            area = calculate_polygon_area(polygon_points)

            # Store the area
            CenterlineSectionArea[i] = area 
        
        return CenterlineSectionArea
        


    CenterlineSectionArea = get_CenterlineSectionArea(mst_pts,CenterlineSectionNormal,surface_pd)
    add_attributes('CenterlineSectionArea',CenterlineSectionArea,mst_polydata)
    write_polydata(mst_polydata, 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\mst_added_bifurcationId_GlobalNodeId_CenterlineId_BranchId_Path_CenterlineSectionNormal_CenterlineSectionArea.vtp')


    pdb.set_trace()
  