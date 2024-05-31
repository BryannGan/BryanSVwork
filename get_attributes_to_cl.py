import vtk
import numpy as np
import os
import sys
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
import pdb
from itertools import combinations

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


# def identify_bifurcation_points(polydata):
#     # This function assumes that 'polydata' contains line segments forming the centerlines
    
#     # Initialize the count of connections for each point
#     connectivity_count = {}

#     # Iterate over all cells (lines) in the polydata
#     for i in range(polydata.GetNumberOfPoints()):
#         cell = polydata.GetCell(i)
#         points = cell.GetPointIds()  # Get the point ids that make up the line cell
        
#         for j in range(points.GetNumberOfIds()):
#             point_id = points.GetId(j)
#             if point_id not in connectivity_count:
#                 connectivity_count[point_id] = 0
#             connectivity_count[point_id] += 1

#     # Identify points with more than 2 connections
#     bifurcation_points = [point for point, count in connectivity_count.items() if count > 2]
    
#     print('Bifurcation points:', bifurcation_points)
#     return bifurcation_points


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

        # save the cell [vtp cell, vtp points, pt coordinates]
        dict_cell[i] = []
        dict_cell[i].append(cell)
        dict_cell[i].append(points)
        dict_cell[i].append(init_pt_array)
    
    return pd, gtclpd, dict_gt_attributes, dict_cell

def findclosestpoint(polydata, refpoint):
    #"""Returns pointid and coordinate of point on polydata closest to refpoint."""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()
    
    id = locator.FindClosestPoint(refpoint)
    coord = polydata.GetPoint(id)
    return id, coord
 

def generate_pairs(n):
    # Create a list of numbers from 0 to n-1
    numbers = list(range(0, n))
    # Generate all pairs
    pairs = combinations(numbers, 2)
    # Format and print the pairs
    result = [f"{a:d}{b:d}" for a, b in pairs]
    return result

def get_BranchId(dict_cell):
    #distance between two stepping points is around 0.593

    # for i in dict_cell:
    #     print(dict_cell[i][2][0])
        #
        #print(np.linalg.norm(np.array(dict_cell[i][2][0]) - np.array(dict_cell[i][2][1])))    
    cell_num = len(dict_cell)
    pairs = generate_pairs(cell_num)
    for i in pairs: #['01', '02', '03', '04', '05', '06', '12', '13', '14', ...]
        cell_id1 = int(pairs[0][0])
        cell_id2 = int(pairs[0][1])



    pdb.set_trace()
    return




if __name__ == '__main__':
    pd, gtclpd, dict_gt_attributes, dict_cell = setup()
    get_BranchId(dict_cell)