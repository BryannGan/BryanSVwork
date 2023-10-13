import sys
# import ur sv_master python folder
sys.path.append('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\SimVascular-master\\Python\\site-packages')
from sv import *
from sv_rom_simulation import *
from sv_rom_simulation import parameters as res_params
import sv
import vtk
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
import pdb 
import os
import numpy as np
import shutil
import argparse
import glob
import os
import csv
import re
from collections import defaultdict, OrderedDict
import scipy
from scipy.interpolate import interp1d

def collect_arrays(output):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        res[name] = v2n(data)
    return res

def get_dict(fpath):
    """
    Read .npy dictionary saved with numpy.save if path is defined and exists
    Args:
        fpath: path to .npy file

    Returns:
        dictionary
    """
    if fpath is not None and os.path.exists(fpath):
        return np.load(fpath, allow_pickle=True).item()
    else:
        return {}
    
def get_all_arrays(geo):
    # collect all arrays
    cell_data = collect_arrays(geo.GetCellData())
    point_data = collect_arrays(geo.GetPointData())

    return point_data, cell_data    

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


def read_results_1d(res_dir, params_file=None):
    """
    Read results from oneDSolver and store in dictionary
    Args:
        res_dir: directory containing 1D results
        params_file: optional, path to dictionary of oneDSolver input parameters

    Returns:
    Dictionary sorted as [result field][segment id][time step]
    """
    # requested output fields
    fields_res_1d = ['flow', 'pressure', 'area', 'wss', 'Re']

    # read 1D simulation results
    results_1d = {}


    for field in fields_res_1d:
        # list all output files for field
        result_list_1d = glob.glob(os.path.join(res_dir, '*branch*seg*_' + field + '.dat'))

        # loop segments
        results_1d[field] = defaultdict(dict)
        for f_res in result_list_1d:
            with open(f_res) as f:
                reader = csv.reader(f, delimiter=' ')

                # loop nodes
                results_1d_f = []
                for line in reader:
                    results_1d_f.append([float(l) for l in line if l][1:])

            # store results and GroupId
            seg = int(re.findall(r'\d+', f_res)[-1])
            branch = int(re.findall(r'\d+', f_res)[-2])
            results_1d[field][branch][seg] = np.array(results_1d_f)

    # read simulation parameters and add to result dict
    results_1d['params'] = get_dict(params_file)

    return results_1d

def save_dict_to_csv(res, output_file):
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(["Field", "Branch", "Segment", "Timestep", "Value"])

        # Loop through fields
        for field, branches in res.items():
            if not isinstance(branches, dict):
                continue  # We only handle dictionaries for this particular format
            for branch, segments in branches.items():
                for segment, values in segments.items():
                    for timestep, value in enumerate(values):
                        writer.writerow([field, branch, segment, timestep, value[0]])


def map_rom_to_centerline(rom, geo_cent, res, time, only_last=True):
    """
    Map 0d or 1d results to centerline
    """
    # assemble output dict
    rec_dd = lambda: defaultdict(rec_dd)
    arrays = rec_dd()

    # get centerline arrays
    arrays_cent, _ = get_all_arrays(geo_cent)

    # centerline points
    points = v2n(geo_cent.GetPoints().GetData())

    # pick results
    if only_last:
        name = rom + '_int_last'
        t_vec = time[rom]
    else:
        name = rom + '_int'
        t_vec = time[rom + '_all']

    # loop all result fields
    for f in res[0].keys():
        if 'path' in f:
            continue
        array_f = np.zeros((arrays_cent['Path'].shape[0], len(t_vec)))
        n_outlet = np.zeros(arrays_cent['Path'].shape[0])
        for br in res.keys():
            # get centerline path
            path_cent = arrays_cent['Path'][arrays_cent['BranchId'] == br]
            path_cent /= path_cent[-1]

            # get 0d path
            path_0d = res[br][rom + '_path']
            path_0d /= path_0d[-1]

            # linearly interpolate results along centerline
            f_cent = interp1d(path_0d, res[br][f][name].T)(path_cent).T

            # store in global array
            array_f[arrays_cent['BranchId'] == br] = f_cent

            # add upstream part of branch within junction
            if br == 0:
                continue

            # first point of branch
            ip = np.where(arrays_cent['BranchId'] == br)[0][0]

            # centerline that passes through branch (first occurence)
            cid = np.where(arrays_cent['CenterlineId'][ip])[0][0]

            # id of upstream junction
            jc = arrays_cent['BifurcationId'][ip - 1]

            # centerline within junction
            jc_cent = np.where(np.logical_and(arrays_cent['BifurcationId'] == jc,
                                              arrays_cent['CenterlineId'][:, cid]))[0]

            # length of centerline within junction
            jc_path = np.append(0, np.cumsum(np.linalg.norm(np.diff(points[jc_cent], axis=0), axis=1)))
            jc_path /= jc_path[-1]

            # results at upstream branch
            res_br_u = res[arrays_cent['BranchId'][jc_cent[0] - 1]][f][name]

            # results at beginning and end of centerline within junction
            f0 = res_br_u[-1]
            f1 = res[br][f][name][0]

            # map 1d results to centerline using paths
            array_f[jc_cent] += interp1d([0, 1], np.vstack((f0, f1)).T, fill_value='extrapolate')(jc_path).T

            # count number of outlets of this junction
            n_outlet[jc_cent] += 1

            # normalize by number of outlets
        array_f[n_outlet > 0] = (array_f[n_outlet > 0].T / n_outlet[n_outlet > 0]).T

        # assemble time steps
        if only_last:
            arrays[0]['point'][f] = array_f[:, -1]
        else:
            for i, t in enumerate(t_vec):
                arrays[str(t)]['point'][f] = array_f[:, i]

    return arrays

def set_res_params():
    res_Params = res_params.Parameters()
    res_Params.data_names = None
    res_Params.output_directory = None
    res_Params.results_directory = None
    res_Params.plot_results = None
    ## Solver parameters.
    res_Params.solver_file_name = None
    res_Params.model_name = None
    res_Params.model_order = None
    res_Params.num_steps = None
    res_Params.time_step = None
    res_Params.time_range = None
    ## Segment names and booleans for selecting segements.
    res_Params.segment_names = None
    res_Params.all_segments = False
    res_Params.outlet_segments = False
    res_Params.select_segment_names = False
    res_Params.output_file_name = None
    res_Params.output_format = "csv"
    # model input
    res_Params.oned_model = '1d_model.vtp'
    res_Params.centerlines_file = None
    res_Params.volume_mesh_file = None
    res_Params.walls_mesh_file = None
    res_Params.display_geometry = False
    res_Params.node_sphere_radius = 0.1

def set_path_name():
    "path to ur result folder"
    result_master_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing'

    "path to ur svproject(gt model) folder"
    svproject_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\sv_projects'

    'path to ur Numi model folder' 'all file starts with Numi_ then the model name'
    Numi_model_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Numi_New_nnUNet_results-20230927'

    'path to ur gt centerline folder'
    gt_cl_path =   'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\centerlines'

    martin_1d_input_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\Martin_result\\input_1d'

    return result_master_folder, svproject_path, Numi_model_path, gt_cl_path, martin_1d_input_path

def main():
    result_master_folder, svproject_path, Numi_model_path, gt_cl_path, martin_1d_input_path = set_path_name()
    path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\1d_results'
    our_res = read_results_1d(path)
    gt_res_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\Martin_result\\results_1d\\0176_0000.npy'
    #save_dict_to_csv(our_res, 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\1d_results\\1d_results.csv')
    # save_dict_to_csv(gt_dic, 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\1d_results\\1d_gt_results.csv')
    gt_dic = np.load(gt_res_path, allow_pickle=True).item()

    # our_res.keys()
    # >>>dict_keys(['pressure', 'wss', 'Re', 'area', 'params', 'flow'])
    our_pressure = our_res['pressure']
    our_wss = our_res['wss']
    our_re = our_res['Re']
    our_area = our_res['area']
    our_params = our_res['params']
    our_flow = our_res['flow']

    gt_pressure = gt_dic['pressure']
    gt_wss = gt_dic['wss']
    gt_re = gt_dic['Re']
    gt_area = gt_dic['area']
    gt_params = gt_dic['params']
    gt_flow = gt_dic['flow']
    
    geo_cent  = read_polydata('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\extracted_centerlines.vtp')
    hi = map_rom_to_centerline('0176_0000.vtp', geo_cent, gt_dic, 0.7, only_last=True)



    pdb.set_trace()
    # visualize results results_1d into 


main()