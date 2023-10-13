import vtk
import os
from vtk.util import numpy_support
import numpy as np
import pandas as pd
import pdb

def read_geo(fname):
    """
    Read geometry from file, chose corresponding vtk reader
    Args:
        fname: vtp surface or vtu volume mesh

    Returns:
        vtk reader, point data, cell data
    """
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

def get_max_branch_id(polydata):
    branchid = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray('BranchId'))
    max_branch_id = np.max(branchid)
    return max_branch_id

def get_max_bifurcation_id(polydata):
    bifurcationid = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray('BifurcationId'))
    max_bifurcation_id = np.max(bifurcationid)
    return max_bifurcation_id

def get_max_centerline_id(polydata):
    centerlineid = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray('CenterlineId'))
    max_centerline_id = len(centerlineid[0])
    return max_centerline_id

def get_average_maxinscribedsphere(polydata):
    maxinscribedsphere = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray('MaximumInscribedSphereRadius'))
    average_maxinscribedsphere = np.mean(maxinscribedsphere)
    return average_maxinscribedsphere

def get_all_data(polydata):
    try:
        max_branch_id = get_max_branch_id(polydata)
    except:
        max_branch_id = 'ERROR'
    try:
        max_bifurcation_id = get_max_bifurcation_id(polydata)
    except:
        max_bifurcation_id = 'ERROR'
    try:
        max_centerline_id = get_max_centerline_id(polydata)
    except: 
        max_centerline_id = 'ERROR'
    try:
        average_maxinscribedsphere = get_average_maxinscribedsphere(polydata)
    except:
        average_maxinscribedsphere = 'ERROR'
    return max_branch_id, max_bifurcation_id, max_centerline_id, average_maxinscribedsphere
    

def get_filename(folder_name):
    if folder_name[0:16] == 'final_assembly_o':
            filename = folder_name[24:33]+'.vtp' #remove Numi_ from the filename
    elif folder_name[0:16] == 'final_assembly_u':
        filename = folder_name[25:34]+'.vtp'
    return filename

def compute_gt_data(folder_name,gt_cl_path):
    filename = get_filename(folder_name)
    gt_cl_path = os.path.join(gt_cl_path,filename)
    gt_cl = read_geo(gt_cl_path)
    max_branch_id, max_bifurcation_id, max_centerline_id, average_maxinscribedsphere = get_all_data(gt_cl)
    return max_branch_id, max_bifurcation_id, max_centerline_id, average_maxinscribedsphere

def compute_pred_data(folder_name,result_dir):
    
    extracted_cl_path = os.path.join(result_dir,folder_name, 'extracted_centerlines.vtp')
    if os.path.exists(extracted_cl_path) == False:
        max_branch_id, max_bifurcation_id, max_centerline_id, average_maxinscribedsphere = 'DOES NOT EXIST', 'DOES NOT EXIST', 'DOES NOT EXIST', 'DOES NOT EXIST'
        return max_branch_id, max_bifurcation_id, max_centerline_id, average_maxinscribedsphere
    
    extracted_clpd = read_geo(extracted_cl_path)
    
    max_branch_id, max_bifurcation_id, max_centerline_id, average_maxinscribedsphere = get_all_data(extracted_clpd)
    return max_branch_id, max_bifurcation_id, max_centerline_id, average_maxinscribedsphere

def create_excel(result_dir, gt_cl_path,output_excel):
    # List all the folders in directory A
    folders = [filename for filename in os.listdir(result_dir)]

    # Create an empty DataFrame with the desired columns
    df = pd.DataFrame(columns=['FolderName', 'GT_BranchID', 'pred_BranchID', 'GT_BifurcationID', 'pred_BifurcationID', 'GT_CenterlineID', 'pred_CenterlineID', 'GT_AverageMaxInscribedSphere', 'pred_AverageMaxInscribedSphere'])

    # Iterate through the folders and populate the DataFrame
    for folder_name in folders:
        folder_path = os.path.join(result_dir, folder_name)

        gt_branch_id, gt_bifurcation_id, gt_centerline_id, gt_average_maxinscribedsphere = compute_gt_data(folder_name,gt_cl_path)
        pred_branch_id, pred_bifurcation_id, pred_centerline_id, pred_average_maxinscribedsphere = compute_pred_data(folder_name,result_dir)

        df = df.append({
            'FolderName': folder_name,
            'GT_BranchID': gt_branch_id,
            'pred_BranchID': pred_branch_id,
            'GT_BifurcationID' : gt_bifurcation_id,
            'pred_BifurcationID' : pred_bifurcation_id,
            'GT_CenterlineID' : gt_centerline_id,
            'pred_CenterlineID' : pred_centerline_id,
            'GT_AverageMaxInscribedSphere' : gt_average_maxinscribedsphere,
            'pred_AverageMaxInscribedSphere' : pred_average_maxinscribedsphere
            
        }, ignore_index=True)

    # Write the DataFrame to an Excel file
    df.to_excel(output_excel, index=False, engine='openpyxl')

if __name__=='__main__':

    result_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Numi_surface_testing_result'
    gt_cl_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\centerlines'
    # pdb.set_trace()
    create_excel(result_path,gt_cl_path,'sep19_seqseg_result.xlsx')
    