import sys
# import ur sv_master python folder
sys.path.append('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\SimVascular-master\\Python\\site-packages')
from sv import *
from sv_rom_simulation import *
#from sv_rom_simulation import parameters as res_params
import sv
import vtk
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
from sv_auto_lv_modeling.modeling.src import meshing as svmeshtool
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

def write_geo(fname, input):
    """
    Write geometry to file
    Args:
        fname: file name
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == '.vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    writer.SetFileName(fname)
    writer.SetInputData(input)
    writer.Update()
    writer.Write()

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


def get_time(model, res, time, dt_3d=0, nt_3d=0, ns_3d=0, t_in=0):
    if '3d_rerun' in model:
        time[model + '_all'] = res['time'] * dt_3d
    elif '3d' in model:
        time[model] = np.array([0] + res['time'].tolist())
        time[model + '_all'] = time[model]
    elif '1d' in model:
        dt = 1e-3
        time[model + '_all'] = np.arange(0, res['pressure'][0][0].shape[1] + 1)[1:] * dt
        time[model + '_all'] = np.append(0, time[model + '_all'])
    elif '0d' in model:
        time[model + '_all'] = res['time']
    else:
        raise RuntimeError('Unknown model ' + model)

    # time steps for last cycle
    if not model == '3d':
        print(time)
        # how many full cycles where completed?
        n_cycle = max(1, int(time[model + '_all'][-1] // t_in))
        time[model + '_n_cycle'] = n_cycle

        # first and last time step in cycle
        t_end = t_in
        t_first = t_end * (n_cycle - 1)
        t_last = t_end * n_cycle

        # tolerance (<< time step * numstep) to prevent errors due to time step round-off
        eps = 1.0e-3

        # select last cycle and shift time to start from zero
        try:
            time[model + '_last_cycle_i'] = np.logical_and(time[model + '_all'] >= t_first - eps, time[model + '_all'] <= t_last + eps)
            time[model] = time[model + '_all'][time[model + '_last_cycle_i']] - t_first
        except:
            pdb.set_trace()
        cycle_range = []
        for i in np.arange(1, n_cycle + 1):
            t_first = t_end * (i - 1)
            t_last = t_end * i
            bound0 = time[model + '_all'] >= t_first - eps
            bound1 = time[model + '_all'] <= t_last + eps
            time[model + '_i_cycle_' + str(i)] = np.logical_and(bound0, bound1)
            time[model + '_cycle_' + str(i)] = time[model + '_all'][time[model + '_i_cycle_' + str(i)]] - t_first
            cycle_range += [np.where(time[model + '_i_cycle_' + str(i)])[0]]
        time[model + '_cycles'] = np.array(cycle_range, dtype=object)
        return time

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
    
# def set_res_params():
#     res_Params = res_params.Parameters()
#     res_Params.data_names = None
#     res_Params.output_directory = None
#     res_Params.results_directory = None
#     res_Params.plot_results = None
#     ## Solver parameters.
#     res_Params.solver_file_name = None
#     res_Params.model_name = None
#     res_Params.model_order = None
#     res_Params.num_steps = None
#     res_Params.time_step = None
#     res_Params.time_range = None
#     ## Segment names and booleans for selecting segements.
#     res_Params.segment_names = None
#     res_Params.all_segments = False
#     res_Params.outlet_segments = False
#     res_Params.select_segment_names = False
#     res_Params.output_file_name = None
#     res_Params.output_format = "csv"
#     # model input
#     res_Params.oned_model = '1d_model.vtp'
#     res_Params.centerlines_file = None
#     res_Params.volume_mesh_file = None
#     res_Params.walls_mesh_file = None
#     res_Params.display_geometry = False
#     res_Params.node_sphere_radius = 0.1

def read_results_0d(fpath):
    """
    Read 0d simulation results from dictionary
    """
    return get_dict(fpath)


def get_branches(arrays):
    """
    Get list of branch IDs from point arrays
    """
    branches = np.unique(arrays['BranchId']).astype(int).tolist()
    if -1 in branches:
        branches.remove(-1)
    return branches

def res_1d_to_path(path, res):
    path_1d = []
    int_1d = []
    for seg, res_1d_seg in sorted(res.items()):
        # 1d results are duplicate at FE-nodes at corners of segments
        if seg == 0:
            # start with first FE-node
            i_start = 0
        else:
            # skip first FE-node (equal to last FE-node of previous segment)
            i_start = 1

        # generate path for segment FEs, assuming equidistant spacing
        p0 = path[seg]
        p1 = path[seg + 1]
        path_1d += np.linspace(p0, p1, res_1d_seg.shape[0])[i_start:].tolist()
        int_1d += res_1d_seg[i_start:].tolist()

    return np.array(path_1d), np.array(int_1d)


def collect_results(model, res, time, f_res, centerline=None, dt_3d=0, nt_3d=0, ns_3d=0, t_in=0, caps=None):
    # read results
    # todo: store 1d results in vtp as well
    if '0d' in model:
        res_in = read_results_0d(f_res)
        f_geo = centerline
        if res_in['time'][0] > 0:
            print('truncating results')
            i_start = np.argmin(np.abs(res_in['time'] - t_in))

            # truncate time
            for f in res_in.keys():
                if f == 'time':
                    res_in[f] = res_in[f][i_start:] - res_in[f][i_start]
                else:
                    for br in res_in[f].keys():
                        for n in res_in[f][br].keys():
                            res_in[f][br][n] = res_in[f][br][n][i_start:]
    elif '1d' in model:
        res_in = f_res # the dictionary directly
        f_geo = centerline
    
    else:
        raise ValueError('Model ' + model + ' not recognized')

    # # read geometry
    # geo = read_geo(f_geo)

    # extract point and cell arrays from geometry
    arrays, _ = get_all_arrays(centerline)

    # get branches
    branches = get_branches(arrays)

    # simulation time steps
    
    get_time(model, res_in, time, dt_3d=dt_3d, nt_3d=nt_3d, ns_3d=ns_3d, t_in=t_in)

    # loop outlets
    for br in branches:
        # 1d-path along branch (real length units)
        branch_path = arrays['Path'][arrays['BranchId'] == br]

        # loop result fields
        for f in ['flow', 'pressure', 'area']:
            if '0d' in model:
                if f == 'area':
                    res[br][f]['0d_int'] = np.zeros(res_in['flow'][br].shape)
                else:
                    res[br][f]['0d_int'] = res_in[f][br]
                    res[br]['0d_path'] = res_in['distance'][br]
            elif '1d' in model:
                res[br]['1d_path'], res[br][f]['1d_int'] = res_1d_to_path(branch_path, res_in[f][br])
                if res[br][f]['1d_int'].shape[1] + 1 == time['1d_all'].shape[0]:
                    res[br][f]['1d_int'] = np.hstack((np.zeros((res[br][f]['1d_int'].shape[0], 1)), res[br][f]['1d_int']))
            
            if br == 0:
                # inlet
                i_cap = 0
            else:
                # outlet
                i_cap = -1

            # extract cap results
            res[br][f][model + '_cap'] = res[br][f][model + '_int'][i_cap, :]

    # get last cycle
    for br in res.keys():
        for f in res[br].keys():
            if 'path' not in f:
                res[br][f][model + '_all'] = res[br][f][model + '_cap']

                if model + '_last_cycle_i' in time and len(time[model + '_last_cycle_i']) > 1:
                    res[br][f][model + '_int_last'] = res[br][f][model + '_int'][:, time[model + '_last_cycle_i']]
                    res[br][f][model + '_cap_last'] = res[br][f][model + '_cap'][time[model + '_last_cycle_i']]
                elif model == '3d':
                    res[br][f][model + '_int_last'] = res[br][f][model + '_int']
                    res[br][f][model + '_cap_last'] = res[br][f][model + '_cap']

def write_results(f_out, cent, arrays, only_last=True):
    """
    Write results to vtp file
    """
    # export last time step (= initial conditions)
    if only_last:
        for f, a in arrays[0]['point'].items():
            out_array = n2v(a)
            out_array.SetName(f)
            cent.GetPointData().AddArray(out_array)
    # export all time steps
    else:
        for t in arrays.keys():
            for f in arrays[t]['point'].keys():
                out_array = n2v(arrays[t]['point'][f])
                out_array.SetName(f + '_' + t)
                cent.GetPointData().AddArray(out_array)

    # write to file
    write_geo(f_out, cent)

def get_inflow_smooth(inflow):
        f = inflow
        if os.path.exists(f):
            m = np.loadtxt(f)
            return m[:, 0], m[:, 1]
        else:
            return None, None

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
    
    # sys.path.append('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall')
    # from rom_master_to_analyze_result_ import *


    result_master_folder, svproject_path, Numi_model_path, gt_cl_path, martin_1d_input_path = set_path_name()
    path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\1d_results'
    our_res = read_results_1d(path)
    gt_res_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\Martin_result\\results_1d\\0176_0000.npy'
    #save_dict_to_csv(our_res, 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\1d_results\\1d_results.csv')
    # save_dict_to_csv(gt_dic, 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\1d_results\\1d_gt_results.csv')
    gt_dic = np.load(gt_res_path, allow_pickle=True).item()

    # our_res.keys()
    # >>>dict_keys(['pressure', 'wss', 'Re', 'area', 'params', 'flow'])
    # (Pdb) our_res.keys()
    # dict_keys(['Re', 'flow', 'wss', 'area', 'pressure', 'params'])
    # (Pdb) our_res['Re'].keys()
    # dict_keys([0, 1, 2, 3, 4, 5, 6])
    # (Pdb) our_res['Re'][0].keys()
    # dict_keys([0])
    # (Pdb) our_res['Re'][2].keys()
    # dict_keys([0, 1, 2])
    
    
    our_cent  = read_polydata('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\extracted_centerlines.vtp')
    gt_cent = read_polydata('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\centerlines\\0176_0000.vtp')
    inflow = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\inflow_files\\inflow_1d.flow'
    
    
    
    # time_inflow, _ = get_inflow_smooth(inflow)

    # result_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\result_analysis'

    
    
    

    # time = {}
    # res = defaultdict(lambda: defaultdict(dict))
    # f_res_1d = gt_dic
    # f_oned = gt_cent
    
    # collect_results('1d', res, time, f_res_1d, f_oned, t_in=time_inflow[-1])
    # # getT = get_time('1d', gt_dic, time,t_in=0.8)

    
    # arrays = map_rom_to_centerline('1d', gt_cent, res, time, only_last=True)
    # f_out  = os.path.join(result_folder, '0176_0000_gt_res_mapped.vtp')

    # write_results(f_out, gt_cent, arrays, only_last=True)

    # # our res
    # time = {}
    # res = defaultdict(lambda: defaultdict(dict))
    # f_res_1d = our_res
    # f_oned = read_polydata('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\extracted_centerlines.vtp')
    # collect_results('1d', res, time, f_res_1d, f_oned, t_in=time_inflow[-1])
    # arrays = map_rom_to_centerline('1d', f_oned, res, time, only_last=True)
    # f_out  = os.path.join(result_folder, '0176_0000_OUR_res_mapped.vtp')
    # write_results(f_out, our_cent, arrays, only_last=True)

    
    def pipeline_mapping_last_t_step_1d_res_to_cl(res_folder,cl_path,inflow_path,output_path):
        
        # initiate parameters
        
        time = {}
        res = defaultdict(lambda: defaultdict(dict))
        clpd = read_polydata(cl_path)
        res_dic = read_results_1d(res_folder)
        time_inflow, _ = get_inflow_smooth(inflow)

        collect_results('1d', res, time, res_dic, clpd, t_in=time_inflow[-1])

        arrays = map_rom_to_centerline('1d', clpd, res, time, only_last=True)
        output_path = os.path.join(output_path, '0176_0000_OUR_GUI_ID_mapped.vtp')
        write_results(output_path, clpd, arrays, only_last=True)
        return res_dic


    out_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\result_analysis'
    our_cent_path  = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\extracted_centerlines.vtp'
    res_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\pipeline_till_simulation_testing\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\1d_result_swapped_GUI_and_id'
    res_dic = pipeline_mapping_last_t_step_1d_res_to_cl(res_folder,our_cent_path,inflow,out_folder)
        

    #--------------------- map 1d to 3d --------------#
    # see Oned_to_3d_projection.py




    pdb.set_trace()
    # visualize results results_1d into 


main()



# (Pdb) arrays_cent, _ = get_all_arrays(geo_cent)
# (Pdb) arrays_cent
# {'Path': array([ 0.        ,  0.52026405,  1.04052893, ..., 20.91086214,
#        21.30466263, 21.69846121]), 'CenterlineSectionNormal': array([[ 0.02004773, -0.20664808,  0.97820991],
#        [-0.0342225 , -0.20406785,  0.97835839],
#        [-0.0883917 , -0.20088523,  0.97561879],
#        ...,
#        [ 0.09778827,  0.1008527 , -0.99008393],
#        [ 0.09520385,  0.05995264, -0.9936508 ],
#        [ 0.09245833,  0.01895114, -0.99553621]]), 'BranchIdTmp': array([0, 0, 0, ..., 8, 8, 8]), 'CenterlineSectionShape': array([0.93048108, 0.98468026, 0.9721108 , ..., 0.98652437, 0.97228775,
#        0.93092067]), 'CenterlineSectionArea': array([4.54488055, 4.48729457, 4.22246703, ..., 3.44803698, 3.520767  ,
#        3.5627084 ]), 'BifurcationId': array([-1, -1, -1, ..., -1, -1, -1], dtype=int64), 'CenterlineSectionMinSize': array([2.3428125 , 2.38708161, 2.33043105, ..., 2.12572651, 2.11818438,
#        2.0400351 ]), 'BifurcationIdTmp': array([-1, -1, -1, ..., -1, -1, -1]), 'CenterlineSectionMaxSize': array([2.51785078, 2.42422004, 2.39728954, ..., 2.15476331, 2.1785571 ,
#        2.1914167 ]), 'MaximumInscribedSphereRadius': array([1.06212353, 1.06212353, 1.06212353, ..., 0.97062236, 0.97062236,
#        0.97062236]), 'CenterlineId': array([[1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        ...,
#        [0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 1]]), 'GlobalNodeId': array([   0,    1,    2, ..., 1493, 1494, 1495]), 'BranchId': array([0, 0, 0, ..., 6, 6, 6], dtype=int64), 'CenterlineSectionBifurcation': array([0, 0, 0, ..., 0, 0, 0]), 'CenterlineSectionClosed': array([1, 1, 1, ..., 1, 1, 1])}
# (Pdb) array_f = np.zeros((arrays_cent['Path'].shape[0], len(t_vec)))
# *** NameError: name 't_vec' is not defined
# (Pdb) t_vec = time['1d']
# (Pdb) array_f = np.zeros((arrays_cent['Path'].shape[0], len(t_vec)))
# (Pdb) n_outlet = np.zeros(arrays_cent['Path'].shape[0])



