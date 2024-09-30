import sys
# import ur sv_master python folder
sys.path.append('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\SimVascular-master\\Python\\site-packages')
#from sv_rom_simulation import *
#from sv_rom_simulation import parameters as res_params
#import sv
import vtk
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
#from sv_auto_lv_modeling.modeling.src import meshing as svmeshtool
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
import matplotlib.pyplot as plt

import seaborn as sns

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





def get_data_at_caps(rom_mapped_cl_path):
    clpd = read_geo(rom_mapped_cl_path)
    points  = v2n(clpd.GetPoints().GetData())
    branchid_at_pts = v2n(clpd.GetPointData().GetArray('BranchId'))
    pressrue_at_pts = v2n(clpd.GetPointData().GetArray('pressure'))
    flow_at_pts = v2n(clpd.GetPointData().GetArray('flow'))
    cap_data = {}
    for i in range(len(branchid_at_pts)-1):
        if branchid_at_pts[i] != branchid_at_pts[i+1]:
            key = 'branchID ' + str(branchid_at_pts[i])
            cap_data[key] = ['pressure', pressrue_at_pts[i], 'flow', flow_at_pts[i]]
    # add last point
    key = 'branchID ' + str(branchid_at_pts[-1])
    cap_data[key] = ['pressure', pressrue_at_pts[-1], 'flow', flow_at_pts[-1]]

    # modify dic: get rid of key that says -1
    cap_data.pop('branchID -1', None)
    # organize dictionary so that keys goes from brachid 0 to bigger
    cap_data = OrderedDict(sorted(cap_data.items(), key=lambda t: int(t[0].split(' ')[1])))
    return cap_data
            

        
def pipeline_mapping_last_t_step_1d_res_to_cl(res_folder,cl_path,inflow_path,output_path,filename):
        
    # initiate parameters
        
    time = {}
    res = defaultdict(lambda: defaultdict(dict))
    clpd = read_geo(cl_path)
    res_dic = read_results_1d(res_folder)
    time_inflow, _ = get_inflow_smooth(inflow_path)

    collect_results('1d', res, time, res_dic, clpd, t_in=time_inflow[-1])

    arrays = map_rom_to_centerline('1d', clpd, res, time, only_last=True)
    output_path = os.path.join(output_path, filename)
    write_results(output_path, clpd, arrays, only_last=True)
    return res_dic, output_path

def pipeline_mapping_1d_res_to_cl(res_folder,cl_path,inflow_path,output_path,filename):
        
    # initiate parameters
        
    time = {}
    res = defaultdict(lambda: defaultdict(dict))
    clpd = read_geo(cl_path)
    res_dic = read_results_1d(res_folder)
    time_inflow, _ = get_inflow_smooth(inflow_path)

    collect_results('1d', res, time, res_dic, clpd, t_in=time_inflow[-1])

    arrays = map_rom_to_centerline('1d', clpd, res, time, only_last=False)
    output_path = os.path.join(output_path, filename)
    write_results(output_path, clpd, arrays, only_last=False)
    return res_dic, output_path

def get_avg_flow_on_branch_over_time(our_res,cycle):
    num_branch = len(our_res['flow'])
    last_segment_flow_over_time_for_each_branch = {i:[] for i in range(num_branch)}
    #calc average at every time step for every branch starting from 1
    for i in range(num_branch):
            # get the last segment: cooresponding to the outlet segment
            last_segment_flow_over_time_for_each_branch[i].append(our_res['flow'][i][-1][:])
    last_segment_flow_over_time_for_each_branch = {i:np.array(last_segment_flow_over_time_for_each_branch[i][0]) for i in range(num_branch)}

    # flow_master.keys()
    # dict_keys([0, 1, 2, 3, 4])
    # len(flow_master[0])
    # 151 -- number of points over branch
    # len(flow_master[0][0])
    # 6889 -- number of time steps
    # sum at each time step for each branch
    
    

    avg_flow_last_cycle = {i:np.mean(last_segment_flow_over_time_for_each_branch[i][-int(len(last_segment_flow_over_time_for_each_branch[i])/cycle):], axis=0) for i in range(num_branch)}
    

    last_segment_flow_of_last_cycle = {i:last_segment_flow_over_time_for_each_branch[i][-int(len(last_segment_flow_over_time_for_each_branch[i])/cycle):] for i in range(num_branch)}
    
    return last_segment_flow_over_time_for_each_branch, avg_flow_last_cycle, last_segment_flow_of_last_cycle
    

def get_avg_pressure_on_branch_over_time(our_res,cycle):
    num_branch = len(our_res['flow'])
    last_segment_pressure_over_time_for_each_branch = {i:[] for i in range(num_branch)}
    for i in range(num_branch):
            # get the last segment: cooresponding to the outlet segment
            last_segment_pressure_over_time_for_each_branch[i].append(our_res['pressure'][i][-1][:])
    last_segment_pressure_over_time_for_each_branch = {i:np.array(last_segment_pressure_over_time_for_each_branch[i][0]) for i in range(num_branch)}
    avg_pressure_last_cycle = {i:np.mean(last_segment_pressure_over_time_for_each_branch[i][-int(len(last_segment_pressure_over_time_for_each_branch[i])/cycle):], axis=0) for i in range(num_branch)}
    last_segment_pressure_of_last_cycle = {i:last_segment_pressure_over_time_for_each_branch[i][-int(len(last_segment_pressure_over_time_for_each_branch[i])/cycle):] for i in range(num_branch)}
    return last_segment_pressure_over_time_for_each_branch, avg_pressure_last_cycle, last_segment_pressure_of_last_cycle





def plot_flow_in_every_branch_of_last_cycle(pred_avg_flow_last_cycle,filename,gt_avg_flow_last_cycle=None):
    # the number of graph = number of branches
    if gt_avg_flow_last_cycle is None:
        for i in range(1,len(pred_avg_flow_last_cycle)):
            plt.plot(pred_avg_flow_last_cycle[i], label = 'branch ' + str(i),linewidth = 6)
            plt.legend()
            plt.title(str(filename) + ' Flow vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('Flow (cm3/s)')
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_Flow_vs_Time.png')
            plt.show()
            
    else:
        for i in range(1,len(pred_avg_flow_last_cycle)):
            plt.plot(pred_avg_flow_last_cycle[i], label = 'MIROS branch ' + str(i),linewidth = 6)
            plt.plot(gt_avg_flow_last_cycle[i], label = 'VMR branch ' + str(i),linewidth = 6)
            plt.legend(fontsize = 20)
            plt.title(str(filename) + ' Flow vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('Flow (cm3/s)')
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_Flow_vs_Time.png')
            plt.show()
            

def plot_pressure_in_every_branch_of_last_cycle(pred_avg_pressure_last_cycle, filename, gt_avg_pressure_last_cycle=None):
    # the number of graph = number of branches
    if gt_avg_pressure_last_cycle is None:
        for i in range(1,len(pred_avg_pressure_last_cycle)):
            plt.plot(pred_avg_pressure_last_cycle[i], label = 'branch ' + str(i),linewidth = 6)
            plt.legend()
            plt.title(str(filename) +' Pressure vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('Pressure (dynes/cm2)')
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_Pressure_vs_Time.png')
            plt.show()
            
    else:
        for i in range(1,len(pred_avg_pressure_last_cycle)):
            plt.plot(pred_avg_pressure_last_cycle[i], label = 'MIROS branch ' + str(i),linewidth = 6)
            plt.plot(gt_avg_pressure_last_cycle[i], label = 'VMR branch ' + str(i),linewidth = 6)
            plt.legend(fontsize = 20)
            plt.title(str(filename)+ ' Pressure vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('Pressure (dynes/cm2)')
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_Pressure_vs_Time.png')
            plt.show()
            
    
def plot_wss_in_every_branch_of_last_cycle(pred_avg_wss_last_cycle,filename, gt_avg_wss_last_cycle=None):
    # the number of graph = number of branches
    if gt_avg_wss_last_cycle is None:
        for i in range(1,len(pred_avg_wss_last_cycle)):
            plt.plot(pred_avg_wss_last_cycle[i], label = 'branch ' + str(i))
            plt.legend()
            plt.title(str(filename)+' WSS vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('WSS (Pa)')
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_WSS_vs_Time.png')
            plt.show()
            

    else:
        for i in range(1,len(pred_avg_wss_last_cycle)):
            plt.plot(pred_avg_wss_last_cycle[i], label = 'pred branch ' + str(i))
            plt.plot(gt_avg_wss_last_cycle[i], label = 'gt branch ' + str(i))
            plt.legend()
            plt.title(str(filename)+ ' WSS vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('WSS (Pa)')
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_WSS_vs_Time.png')
            plt.show()
            

def plot_Re_in_every_branch_of_last_cycle(pred_avg_Re_last_cycle,filename, gt_avg_Re_last_cycle=None):
    # the number of graph = number of branches
    if gt_avg_Re_last_cycle is None:
        for i in range(1,len(pred_avg_Re_last_cycle)):
            plt.plot(pred_avg_Re_last_cycle[i], label = 'branch ' + str(i))
            plt.legend()
            plt.title(str(filename)+' Re vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('Re')
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_Re_vs_Time.png')
            plt.show()
            
    else:
        for i in range(1,len(pred_avg_Re_last_cycle)):
            plt.plot(pred_avg_Re_last_cycle[i], label = 'pred branch ' + str(i))
            plt.plot(gt_avg_Re_last_cycle[i], label = 'gt branch ' + str(i))
            plt.legend()
            plt.title(str(filename)+ ' Re vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('Re')
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_Re_vs_Time.png')

            plt.show()
            
def plot_area_in_every_branch_of_last_cycle(pred_avg_area_last_cycle,filename, gt_avg_area_last_cycle=None):
    # the number of graph = number of branches
    if gt_avg_area_last_cycle is None:
        for i in range(1,len(pred_avg_area_last_cycle)):
            plt.plot(pred_avg_area_last_cycle[i], label = 'branch ' + str(i))
            plt.legend()
            plt.title(str(filename)+' Area vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('Area (cm2)')
            plt.ylim(bottom=0)
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_Area_vs_Time.png')
            plt.show()
            
    else:
        for i in range(1,len(pred_avg_area_last_cycle)):
            plt.plot(pred_avg_area_last_cycle[i], label = 'pred branch ' + str(i))
            plt.plot(gt_avg_area_last_cycle[i], label = 'gt branch ' + str(i))
            plt.legend()
            plt.title(str(filename)+' Area vs Time')
            plt.xlabel('Time (ms)')
            plt.ylabel('Area (cm2)')
            plt.ylim(bottom=0)
            plt.savefig(str(filename)+'_branch_'+ str(i)+'_Area_vs_Time.png')
            plt.show()
            
def create_flow_box_plot(lst_of_data):
    # Prepare the data for plotting
     # Prepare the data for plotting
    plot_data = []
    labels = []
    counts = []  # To store the counts of errors per model
    for model, errors_dict in lst_of_data:
        errors = list(errors_dict.values())  # Extract the error percentages from the dictionary
        plot_data.extend(errors)
        labels.extend([model] * len(errors))
        counts.append(len(errors))  # Keep track of the number of data points for each model

    # Create the boxplot
    ax = sns.boxplot(x=labels, y=plot_data)
    plt.xlabel('Model Name')
    plt.ylabel('Relative Error %')
    plt.title('Box Plot of 0D Model Flow Errors')
    plt.xticks(rotation=45)  # Rotating the x-axis labels for better readability
    for i, count in enumerate(counts):
        # Place the text at the mean y-position of the last 'count' points
        y_pos = sum(plot_data[i*count:(i+1)*count]) / count
        ax.text(i, y_pos, f'n={count}', horizontalalignment='center', size='x-small', color='black', weight='semibold')

    plt.show()

def create_pressure_box_plot(lst_of_data):
    # Prepare the data for plotting
    plot_data = []
    labels = []
    counts = []  # To store the counts of errors per model
    for model, errors_dict in lst_of_data:
        errors = list(errors_dict.values())  # Extract the error percentages from the dictionary
        plot_data.extend(errors)
        labels.extend([model] * len(errors))
        counts.append(len(errors))  # Keep track of the number of data points for each model

    # Create the boxplot
    ax = sns.boxplot(x=labels, y=plot_data)
    plt.xlabel('Model Name')
    plt.ylabel('Relative Error %')
    plt.title('Box Plot of 0D Model Pressure Errors')
    plt.xticks(rotation=45)  # Rotating the x-axis labels for better readability
    for i, count in enumerate(counts):
        # Place the text at the mean y-position of the last 'count' points
        y_pos = sum(plot_data[i*count:(i+1)*count]) / count
        ax.text(i, y_pos, f'n={count}', horizontalalignment='center', size='x-small', color='black', weight='semibold')

    plt.show()

def set_path_name():
    "path to ur result folder"
    result_master_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\pipeline_testing_master\\for_sb3c'

    "path to ur svproject(gt model) folder"
    svproject_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\rom_sv_projects'

    'path to ur Numi model folder' 'all file starts with Numi_ then the model name'
    Numi_model_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\rom_Numi_surf'

    'path to ur gt centerline folder'
    gt_cl_path =   'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\centerlines'

    martin_1d_input_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\Martin_result\\input_1d'

    return result_master_folder, svproject_path, Numi_model_path, gt_cl_path, martin_1d_input_path
   

def main():
    
    def read_0d_res(fpath):
        res = np.load(fpath, allow_pickle=True)
        res = np.ndarray.tolist(res)
        return res

    # MLres = np.load('XXXXXXX.npy', allow_pickle=True)
    # MLres = np.ndarray.tolist(MLres)
        
    # result_master_folder, svproject_path, Numi_model_path, gt_cl_path, martin_1d_input_path = set_path_name()
    box_plot_flow_data = []
    box_plot_pressure_data = []
    extraction_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\pipeline_testing_master\\for_sb3c_0d\\ready_for_result_extraction'
    for filename in os.listdir(extraction_folder):
        model_master_folder = os.path.join(extraction_folder,filename)
        gt_res = os.path.join(model_master_folder,'GT_0d_solver_output_branch_results.npy')
        pred_res = os.path.join(model_master_folder,'PRED_0d_solver_output_branch_results.npy')
        
        if os.path.exists(gt_res) and os.path.exists(pred_res):
            pred_res = read_0d_res(pred_res)
            gt_res = read_0d_res(gt_res)
            
            ### add function to parse # of cycle
            if filename == '0063_1001_clipped_away_branch4':
                cycle = 10
            elif filename == '0090_0001':
                cycle = 8
            elif filename == '0131_0000':
                cycle = 15
            elif filename == '0146_1001':
                cycle = 9
            elif filename == '0176_0000':
                cycle = 9
            elif filename == '0174_0000_new':
                cycle = 17
            else:
                cycle = 6
           

            last_segment_flow_over_time_for_each_branch, avg_flow_last_cycle, last_segment_flow_of_last_cycle = get_avg_flow_on_branch_over_time(pred_res,cycle)
            
            last_segment_pressure_over_time_for_each_branch, avg_pressure_last_cycle, last_segment_pressure_of_last_cycle = get_avg_pressure_on_branch_over_time(pred_res,cycle)
            
            last_segment_flow_over_time_for_each_branch_gt, avg_flow_last_cycle_gt, last_segment_flow_of_last_cycle_gt = get_avg_flow_on_branch_over_time(gt_res,cycle)
            last_segment_pressure_over_time_for_each_branch_gt, avg_pressure_last_cycle_gt, last_segment_pressure_of_last_cycle_gt = get_avg_pressure_on_branch_over_time(gt_res,cycle)
            
            if filename == '0176_0000':
                # swap branches: match gt to pred
                last_segment_flow_of_last_cycle_gt[1], last_segment_flow_of_last_cycle_gt[2],last_segment_flow_of_last_cycle_gt[4], last_segment_flow_of_last_cycle_gt[5],last_segment_flow_of_last_cycle_gt[6] = last_segment_flow_of_last_cycle_gt[2], last_segment_flow_of_last_cycle_gt[6],last_segment_flow_of_last_cycle_gt[5], last_segment_flow_of_last_cycle_gt[4],last_segment_flow_of_last_cycle_gt[1]
                last_segment_pressure_of_last_cycle_gt[1], last_segment_pressure_of_last_cycle_gt[2],last_segment_pressure_of_last_cycle_gt[4], last_segment_pressure_of_last_cycle_gt[5],last_segment_pressure_of_last_cycle_gt[6] = last_segment_pressure_of_last_cycle_gt[2], last_segment_pressure_of_last_cycle_gt[6],last_segment_pressure_of_last_cycle_gt[5], last_segment_pressure_of_last_cycle_gt[4],last_segment_pressure_of_last_cycle_gt[1]
                

            if filename == '0131_0000':
                # swap branches: match gt to pred
                last_segment_flow_of_last_cycle_gt[3],last_segment_flow_of_last_cycle_gt[4],last_segment_flow_of_last_cycle_gt[6] = last_segment_flow_of_last_cycle_gt[6],last_segment_flow_of_last_cycle_gt[3],last_segment_flow_of_last_cycle_gt[4]
                last_segment_pressure_of_last_cycle_gt[3],last_segment_pressure_of_last_cycle_gt[4],last_segment_pressure_of_last_cycle_gt[6] = last_segment_pressure_of_last_cycle_gt[6],last_segment_pressure_of_last_cycle_gt[3],last_segment_pressure_of_last_cycle_gt[4]
           
            if filename == "0146_1001":
                last_segment_flow_of_last_cycle_gt[1], last_segment_flow_of_last_cycle_gt[2],last_segment_flow_of_last_cycle_gt[3],last_segment_flow_of_last_cycle_gt[4], last_segment_flow_of_last_cycle_gt[5],last_segment_flow_of_last_cycle_gt[6], last_segment_flow_of_last_cycle_gt[7], last_segment_flow_of_last_cycle_gt[8],last_segment_flow_of_last_cycle_gt[9] = last_segment_flow_of_last_cycle_gt[3],last_segment_flow_of_last_cycle_gt[4],last_segment_flow_of_last_cycle_gt[5],last_segment_flow_of_last_cycle_gt[6],last_segment_flow_of_last_cycle_gt[7],last_segment_flow_of_last_cycle_gt[8],last_segment_flow_of_last_cycle_gt[9],last_segment_flow_of_last_cycle_gt[1],last_segment_flow_of_last_cycle_gt[2]
                last_segment_pressure_of_last_cycle_gt[1], last_segment_pressure_of_last_cycle_gt[2],last_segment_pressure_of_last_cycle_gt[3],last_segment_pressure_of_last_cycle_gt[4], last_segment_pressure_of_last_cycle_gt[5],last_segment_pressure_of_last_cycle_gt[6], last_segment_pressure_of_last_cycle_gt[7], last_segment_pressure_of_last_cycle_gt[8],last_segment_pressure_of_last_cycle_gt[9] = last_segment_pressure_of_last_cycle_gt[3],last_segment_pressure_of_last_cycle_gt[4],last_segment_pressure_of_last_cycle_gt[5],last_segment_pressure_of_last_cycle_gt[6],last_segment_pressure_of_last_cycle_gt[7],last_segment_pressure_of_last_cycle_gt[8],last_segment_pressure_of_last_cycle_gt[9],last_segment_pressure_of_last_cycle_gt[1],last_segment_pressure_of_last_cycle_gt[2]
             

            if filename == '0174_0000_new':
                last_segment_flow_of_last_cycle_gt[1], last_segment_flow_of_last_cycle_gt[2],last_segment_flow_of_last_cycle_gt[3],last_segment_flow_of_last_cycle_gt[4],last_segment_flow_of_last_cycle_gt[5] = last_segment_flow_of_last_cycle_gt[2], last_segment_flow_of_last_cycle_gt[3], last_segment_flow_of_last_cycle_gt[4], last_segment_flow_of_last_cycle_gt[5],last_segment_flow_of_last_cycle_gt[1]
                last_segment_pressure_of_last_cycle_gt[1], last_segment_pressure_of_last_cycle_gt[2],last_segment_pressure_of_last_cycle_gt[3],last_segment_pressure_of_last_cycle_gt[4],last_segment_pressure_of_last_cycle_gt[5] = last_segment_pressure_of_last_cycle_gt[2], last_segment_pressure_of_last_cycle_gt[3], last_segment_pressure_of_last_cycle_gt[4], last_segment_pressure_of_last_cycle_gt[5],last_segment_pressure_of_last_cycle_gt[1]   
            # calculate relative error across time
            
            
            #write into function
            def calc_relative_error_across_time(pred_avg, gt_avg,type):
                mean_value_pred = {i:np.mean(pred_avg[i]) for i in range(1,len(pred_avg))}
                mean_value_gt = {i:np.mean(gt_avg[i]) for i in range(1,len(gt_avg))}
                rel_error_branch = {i:np.abs((mean_value_pred[i]- mean_value_gt[i])/mean_value_gt[i]) for i in range(1,len(pred_avg))}
                
                rel_error_of_surface = sum(rel_error_branch.values()) / len(rel_error_branch)
                
                print('relative error of ', type, ' across time is', rel_error_of_surface)
                return rel_error_branch, rel_error_of_surface


            
            # out_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\pipeline_testing_master\\for_sb3c\\0090_0001\\extracted_pred_centerlines\\result_from_program'
            # our_cent_path  = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\pipeline_testing_master\\for_sb3c\\0090_0001\\extracted_pred_centerlines.vtp'
            # res_folder = path
            # pdb.set_trace()
            # last_res_dic, last_rom_cl_path = pipeline_mapping_last_t_step_1d_res_to_cl(res_folder,our_cent_path,inflow,out_folder,'last_t_step_mapped.vtp')
    
            
            #flow
            rel_error_mean_flow_branch, rel_error_avg_flow_surf, = calc_relative_error_across_time(avg_flow_last_cycle, last_segment_flow_of_last_cycle_gt, 'flow')
            #pressure
            rel_error_mean_pressure_branch, rel_error_avg_pressure_surf = calc_relative_error_across_time(avg_pressure_last_cycle, last_segment_pressure_of_last_cycle_gt, 'pressure')
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(filename, "avg flow error", rel_error_avg_flow_surf, "avg pressure error", rel_error_avg_pressure_surf)
            print(filename,'flow',rel_error_mean_flow_branch,'pressure', rel_error_mean_pressure_branch)
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            for key in rel_error_mean_flow_branch:
                rel_error_mean_flow_branch[key]*=100
            for key in rel_error_mean_pressure_branch:
                rel_error_mean_pressure_branch[key]*=100


            box_plot_flow_data.append([filename, rel_error_mean_flow_branch])
            box_plot_pressure_data.append([filename, rel_error_mean_pressure_branch])
                
            # plot_flow_in_every_branch_of_last_cycle(last_segment_flow_of_last_cycle,filename, last_segment_flow_of_last_cycle_gt)
            # plot_pressure_in_every_branch_of_last_cycle(last_segment_pressure_of_last_cycle,filename,last_segment_pressure_of_last_cycle_gt)
         

        else:
            continue

    create_flow_box_plot(box_plot_flow_data)
    create_pressure_box_plot(box_plot_pressure_data)








main()



