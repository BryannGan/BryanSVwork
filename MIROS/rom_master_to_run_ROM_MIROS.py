import sys
# import ur sv_master python folder
#sys.path.append('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\SimVascular-master\\Python\\site-packages')
from sv import *
from sv_rom_simulation import *
import sv
import vtk
from vtk.util import numpy_support
from sv_auto_lv_modeling.modeling.src import meshing as svmeshtool
import pdb 
import os
import numpy as np
import shutil
import subprocess
import csv
from scipy.optimize import linear_sum_assignment
#import vmtk

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

def edit_log(path, input, txt_file="process_log.txt"):
    txt_file_path = os.path.join(path, txt_file)
    with open(txt_file_path, "a") as f:
        f.write(input + "\n")
        f.close()
    return


def mkdir(path, name):
    new_folder_path = os.path.join(path, name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path, exist_ok=True)
    return new_folder_path
        
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

def bryan_get_clipping_parameters(clpd):
    points = numpy_support.vtk_to_numpy(clpd.GetPoints().GetData())
    CenterlineID_for_each_point = numpy_support.vtk_to_numpy(clpd.GetPointData().GetArray('CenterlineId'))
    radii = numpy_support.vtk_to_numpy(clpd.GetPointData().GetArray('MaximumInscribedSphereRadius'))
    n_keys = len(CenterlineID_for_each_point[0])
    line_dict = {}

    #create dict with keys line0, line1, line2, etc
    for i in range(n_keys):
        key = "line{}".format(i)
        line_dict[key] = []

    for i in range(len(points)):
        for j in range(n_keys):
            if CenterlineID_for_each_point[i][j] == 1:
                key = "line{}".format(j)
                line_dict[key].append(points[i])
    
    for i in range(n_keys):
        key = "line{}".format(i)
        line_dict[key] = np.array(line_dict[key])
    # Done with spliting centerliens into dictioanry

    # find the end points of each line
    lst_of_end_pts = []
    # append the very first point
    lst_of_end_pts.append(line_dict["line0"][0])
    # append the rest of the end points
    for i in range(n_keys):
        key = "line{}".format(i)
        lst_of_end_pts.append(line_dict[key][-1])
    nplst_of_endpts = np.array(lst_of_end_pts) #convert to numpy array

    # find the radii at the end points
    radii_at_caps = []
    for i in lst_of_end_pts:
            for j in range(len(points)):
                if np.array_equal(i,points[j]):
                    radii_at_caps.append(radii[j])
    nplst_radii_at_caps = np.array(radii_at_caps) #convert to numpy array

    # find the unit tangent vectors at the end points
    unit_tangent_vectors = []
    #compute the unit tangent vector of the first point of the first line
    key = "line0"
    line = line_dict[key]
    tangent_vector = line[0] - line[1]
    unit_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
    unit_tangent_vectors.append(unit_tangent_vector)
    # compute the unit tangent vector of the last point of each line
    for i in range(len(line_dict)):
        key = "line{}".format(i)
        line = line_dict[key]
        tangent_vector = line[-1] - line[-2]
        unit_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
        unit_tangent_vectors.append(unit_tangent_vector)


    return nplst_of_endpts, nplst_radii_at_caps, unit_tangent_vectors



def bryan_generate_oriented_boxes(endpts,unit_tan_vectors,radius,output_file,box_scale=3):
    box_surfaces = vtk.vtkAppendPolyData()
    # Convert the input center_points to a list, in case it is a NumPy array
    endpts = np.array(endpts).tolist()
    centerpts = []
    pd_lst = []
    for i in range(len(endpts)):
        
        compute_x = endpts[i][0]+0.5*box_scale*radius[i]*unit_tan_vectors[i][0]
        compute_y = endpts[i][1]+0.5*box_scale*radius[i]*unit_tan_vectors[i][1]
        compute_z = endpts[i][2]+0.5*box_scale*radius[i]*unit_tan_vectors[i][2] 
        centerpts.append([compute_x,compute_y,compute_z])
    
    box_surfaces = vtk.vtkAppendPolyData()

    for i in range(len(centerpts)):
        # Create an initial vtkCubeSource for the box
        box = vtk.vtkCubeSource()
        box.SetXLength(box_scale*radius[i])
        box.SetYLength(box_scale*radius[i])
        box.SetZLength(box_scale*radius[i])
        box.Update()

         # Compute the rotation axis by taking the cross product of the unit_vector and the z-axis
        rotation_axis = np.cross(np.array([0, 0, 1]),unit_tan_vectors[i])

        # Compute the rotation angle in degrees between the unit_vector and the z-axis
        rotation_angle = np.degrees(np.arccos(np.dot(unit_tan_vectors[i], np.array([0, 0, 1]))))
        transform = vtk.vtkTransform()
        transform.Translate(centerpts[i])
        transform.RotateWXYZ(rotation_angle, rotation_axis)

        # Apply the transform to the box
        box_transform = vtk.vtkTransformPolyDataFilter()
        box_transform.SetInputConnection(box.GetOutputPort())
        box_transform.SetTransform(transform)
        box_transform.Update()

        pd_lst.append(box_transform.GetOutput())
        box_surfaces.AddInputData(box_transform.GetOutput())
    
    box_surfaces.Update()

    # Write the oriented box to a .vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(box_surfaces.GetOutput())
    writer.Write()
    return box_surfaces.GetOutput(),pd_lst

def bryan_clip_surface(surf1, surf2):
    # Create an implicit function from surf2
    implicit_function = vtk.vtkImplicitPolyDataDistance()
    implicit_function.SetInput(surf2)

    # Create a vtkClipPolyData filter and set the input and implicit function
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(surf1)
    clipper.SetClipFunction(implicit_function)
    clipper.InsideOutOff()  # keep the part of surf1 outside of surf2
    clipper.Update()

    # Get the output polyData with the part enclosed by surf2 clipped away
    clipped_surf1 = clipper.GetOutput()

    return clipped_surf1

def bryan_fillholes(pd):
    # Fill the holes
    fill = vtk.vtkFillHolesFilter()
    fill.SetInputData(pd)
    fill.SetHoleSize(100000.0)
    fill.Update()
    return fill.GetOutput()


def keep_largest_surface(polyData):
    # Create a connectivity filter to label the regions
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(polyData)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()

    # Get the output of the connectivity filter
    connectedPolyData = connectivity.GetOutput()

    # Get the region labels
    regionLabels = connectedPolyData.GetPointData().GetArray('RegionId')

    # Convert region labels to numpy array
    regionLabels_np = vtk.util.numpy_support.vtk_to_numpy(regionLabels)

    # Find the unique region labels and their counts
    uniqueLabels, counts = np.unique(regionLabels_np, return_counts=True)

    # Find the label of the largest region
    largestRegionLabel = uniqueLabels[np.argmax(counts)]

    # Create a threshold filter to extract the largest region
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(connectedPolyData)
    threshold.ThresholdBetween(largestRegionLabel, largestRegionLabel)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'RegionId')
    threshold.Update()

    # Convert the output of the threshold filter to vtkPolyData
    largestRegionPolyData = vtk.vtkGeometryFilter()
    largestRegionPolyData.SetInputData(threshold.GetOutput())
    largestRegionPolyData.Update()

    return largestRegionPolyData.GetOutput()

def bryan_boxcut(centerline_file,pred_surface_file,file_name,scale, gt_cl_file=None):
    if gt_cl_file is None:
        # adjust cap according to its own centerline
        clpd = read_geo(centerline_file)
        predpd = read_geo(pred_surface_file)
        endpts, radii, unit_vecs = bryan_get_clipping_parameters(clpd)
        boxpd, boxpdlst = bryan_generate_oriented_boxes(endpts,unit_vecs,radii, file_name +'clippingbox.vtp',scale)
        clippedpd = bryan_clip_surface(predpd,boxpd)
        largest = keep_largest_surface(clippedpd)
        return largest
    else: # clip predicted surface according to gt centerline
        clpd = read_geo(gt_cl_file)
        predpd = read_geo(pred_surface_file)
        # print(predpd)
        endpts, radii, unit_vecs = bryan_get_clipping_parameters(clpd)
        # print('debug:')
        # print(endpts)
        boxpd, boxpdlst = bryan_generate_oriented_boxes(endpts,unit_vecs,radii,'146_clippingbox.vtp',scale)
        clippedpd = bryan_clip_surface(predpd,boxpd)
        largest = keep_largest_surface(clippedpd)
        return largest,boxpd

def zerod_sim(rom_ready_vtp_path):
    model_vtp = read_surface(rom_ready_vtp_path,'vtp',None)
    extract = sv.modeling.PolyData(model_vtp)
    extract.compute_boundary_faces(90.0) # 90 degree angle --> form 1 face (the wall)   // under modeling 
    extracted_pd = extract.get_polydata() # extract pd from modeling class        #under modeling
    filled = sv.vmtk.cap(extracted_pd)   #filled holes --> form inlets and outlets   #under vmtk
    filled_md =sv.modeling.PolyData(filled) # create new object (polydata with face id attribute)

def setup_PRED_ROM_parameters(path,model_name,order):
    if order == 1:
        Params = Parameters() # put everything is params
        Params.density = 1.06
        Params.output_directory = path
        Params.boundary_surfaces_dir = os.path.join(path,'pred_boundary_faces')
        Params.inlet_face_input_file = 'inlet.vtp'
        Params.centerlines_output_file = os.path.join(path,'extracted_pred_centerlines.vtp')
        Params.surface_model = os.path.join(path,'export_PRED_ROM_ready_surface.vtp')
        Params.inflow_input_file = os.path.join(path,'pred_inflow_files','inflow_1d.flow')
        Params.model_order = 1
        Params.solver_output_file = 'PRED_1d_solver_output.in' # need this to write out the solver file
        Params.model_name = model_name
        Params.outflow_bc_type = 'rcr' #rcr, resistance or coronary vmr 3d sim uses this and it calls 
        Params.outflow_bc_file =  os.path.join(path,'pred_inflow_files\\')
        # create own .dat file? 
        Params.uniform_bc = False
        Params.seg_min_num = 8
        Params.outlet_face_names_file = os.path.join(path,'centerlines_outlets.dat')
        Params.CENTERLINES_OUTLET_FILE_NAME = 'centerlines_outlets.dat'
        return Params
    elif order == 0:
        Params = Parameters() # put everything is params
        Params.density = 1.06
        Params.output_directory = path
        Params.boundary_surfaces_dir = os.path.join(path,'pred_boundary_faces')
        Params.inlet_face_input_file = 'inlet.vtp'
        Params.centerlines_output_file = os.path.join(path,'extracted_pred_centerlines.vtp')
        Params.surface_model = os.path.join(path,'export_PRED_ROM_ready_surface.vtp')
        Params.inflow_input_file = os.path.join(path,'pred_inflow_files','inflow_1d.flow')
        Params.model_order = 0
        Params.solver_output_file = 'PRED_0d_solver_output.json' # need this to write out the solver file
        Params.model_name = model_name
        Params.outflow_bc_type = 'rcr' #rcr, resistance or coronary vmr 3d sim uses this and it calls 
        Params.outflow_bc_file =  os.path.join(path,'pred_inflow_files\\')
        # create own .dat file? 
        Params.uniform_bc = False
        #watch out!
        Params.model_order= 0
        Params.seg_min_num = 3
        Params.outlet_face_names_file = os.path.join(path,'centerlines_outlets.dat')
        Params.CENTERLINES_OUTLET_FILE_NAME = 'centerlines_outlets.dat'
        
        return Params

def setup_GT_ROM_parameters(path,model_name,order):
    if order == 1:
        Params = Parameters() # put everything is params
        Params.density = 1.06
        Params.output_directory = path
        Params.boundary_surfaces_dir = os.path.join(path,'GT_sim_boundary_faces')
        Params.inlet_face_input_file = 'inlet.vtp'
        Params.centerlines_output_file = os.path.join(path,'extracted_GT_sim_centerlines.vtp')
        Params.surface_model = os.path.join(path,'modified_gt_surf_for_sim.vtp')
        Params.inflow_input_file = os.path.join(path,'gt_inflow_files','inflow_1d.flow')
        Params.model_order = 1
        Params.solver_output_file = 'GT_1d_solver_output.in' # need this to write out the solver file
        Params.model_name = model_name
        Params.outflow_bc_type = 'rcr' #rcr, resistance or coronary vmr 3d sim uses this and it calls 
        Params.outflow_bc_file =  os.path.join(path,'gt_inflow_files\\')
        # create own .dat file? 
        Params.uniform_bc = False
        Params.seg_min_num = 8
        Params.outlet_face_names_file = os.path.join(path,'VMR_centerlines_outlets.dat')
        Params.CENTERLINES_OUTLET_FILE_NAME = 'VMR_centerlines_outlets.dat'
        return Params
    elif order == 0:
        Params = Parameters() # put everything is params
        Params.density = 1.06
        Params.output_directory = path
        Params.boundary_surfaces_dir = os.path.join(path,'GT_sim_boundary_faces')
        Params.inlet_face_input_file = 'inlet.vtp'
        Params.centerlines_output_file = os.path.join(path,'extracted_GT_sim_centerlines.vtp')
        Params.surface_model = os.path.join(path,'modified_gt_surf_for_sim.vtp')
        Params.inflow_input_file = os.path.join(path,'gt_inflow_files','inflow_1d.flow')
        Params.model_order = 0
        Params.solver_output_file = 'GT_0d_solver_output.json' # need this to write out the solver file
        Params.model_name = model_name
        Params.outflow_bc_type = 'rcr' #rcr, resistance or coronary vmr 3d sim uses this and it calls 
        Params.outflow_bc_file =  os.path.join(path,'gt_inflow_files\\')
        # create own .dat file? 
        Params.uniform_bc = False
        #watch out!
        Params.model_order= 0
        Params.seg_min_num = 3
        Params.outlet_face_names_file = os.path.join(path,'VMR_centerlines_outlets.dat')
        Params.CENTERLINES_OUTLET_FILE_NAME = 'VMR_centerlines_outlets.dat' # for writing out file when extracting cl 
        return Params




def parse_mdl_to_list(file_path):
    """ 
    Parses an .mdl file (in sv_proj\modles folder) and returns a list of faces.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()

    face_list = []

    for line in lines:
        if "<face id=" in line:
            # Remove XML tag brackets and split by spaces
            parts = line.replace("<", "").replace(">", "").split()

            # Get face id, name, and type
            face_id = int(parts[1].split('"')[1])
            face_name = parts[2].split('"')[1]
            face_type = parts[3].split('"')[1]

            face_list.append([face_id, face_name, face_type])


    return face_list
#lst = parse_mdl_to_list('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\sv projects\\0146_1001\\Models\\0146_1001.mdl')

def sort_and_reorder(lst):
    """
    Sorts the list of faces by face_id and reorders the list so that the inflow face is first and remove wall face from list. 
    """
    # First, remove "wall" entries
    lst = [x for x in lst if x[1] != 'wall']

    # Then, sort the list
    lst = sorted(lst, key=lambda x: x[0])

    # find the inflow
    for i in range(len(lst)):
        if lst[i][1] == 'inflow':
            inflow_id = i

    # if inflow is not the first item, reorder the list
    if inflow_id != 0:
        lst = lst[inflow_id:] + lst[:inflow_id]

    face_name_lst = [item[1] for item in lst]
    return lst, face_name_lst

def insert_outlet_names_to_rcrt(file_path, outlet_names):
    """
    Inserts outlet names into the rcrt.dat file.
    """
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    
    # outlet name index
    outlet_index = 0
    new_lines = []
    
    # Add first two lines without modifying them
    new_lines.append(lines[0])


    # Start from third line
    for line in lines[1:]:
        if line.strip() == "2":  # If we encounter a border line
            new_lines.append(line)  # add the border line
            new_lines.append(outlet_names[outlet_index])  # add the outlet name
            outlet_index += 1  # move to the next outlet name
        else:
            new_lines.append(line)  # add the line as is

    return new_lines

def write_to_file(file_path, lines):
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(str(line) + "\n")
# path C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\sv projects\\0146_1001\\Simulations\\0146_1001\\rcrt.dat
# outlet_names = face_name_lst

# lst = parse_mdl_to_list('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\sv projects\\0146_1001\\Models\\0146_1001.mdl')
# >>> lst
# [[1, 'wall', 'wall'], [10, 'renal_left', 'cap'], [11, 'renal_right', 'cap'], [12, 'SMA', 'cap'], 
# [2, 'inflow', 'cap'], [3, 'celiac_hepatic', 'cap'], [4, 'celiac_splenic', 'cap'],
# [5, 'ext_iliac_left', 'cap'], [6, 'ext_iliac_right', 'cap'], [7, 'IMA', 'cap'],
# [8, 'int_iliac_left', 'cap'], [9, 'int_iliac_right', 'cap']]
# lst, face_name_lst = sort_and_reorder(lst)
# >>> lst
# [[2, 'inflow', 'cap'], [3, 'celiac_hepatic', 'cap'], [4, 'celiac_splenic', 'cap'], 
#  [5, 'ext_iliac_left', 'cap'], [6, 'ext_iliac_right', 'cap'], [7, 'IMA', 'cap'], [8, 'int_iliac_left', 'cap'],
#  [9, 'int_iliac_right', 'cap'], [10, 'renal_left', 'cap'], [11, 'renal_right', 'cap'], [12, 'SMA', 'cap']]
# >>> face_name_lst
# ['celiac_hepatic', 'celiac_splenic', 'ext_iliac_left', 'ext_iliac_right', 'IMA', 'int_iliac_left', 'int_iliac_right', 'renal_left', 'renal_right', 'SMA']
# lines = insert_outlet_names_to_rcrt('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\sv projects\\0146_1001\\Simulations\\0146_1001\\rcrt.dat',face_name_lst)
# write_to_file('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\ROM results\\inflow_files\\rcrt_from_parsing.dat',lines)



def compute_polydata_centroid(polydata):
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    x, y, z = 0, 0, 0
    for i in range(num_points):
        px, py, pz = points.GetPoint(i)
        x += px
        y += py
        z += pz
    centroid = (x / num_points, y / num_points, z / num_points)
    return centroid

def compute_distance(centroid1, centroid2):
        return sum([(c1 - c2) ** 2 for c1, c2 in zip(centroid1, centroid2)]) ** 0.5

def match_files(dir_a, dir_b, tolerance=0.1):
    """
    Match files from two directories based on their centroids within a certain tolerance.

    :param dir_a: path to the ground truth caps!
    :param dir_b: path to the pred caps!
    :param tolerance: tolerance for matching centroids
    :return: list of matched polydata pairs and list of unmatched file names in dir_b
    """

    def get_centroids(directory):
        centroids = {}
        for filename in os.listdir(directory):
            if filename.endswith('.vtp'):
                file_path = os.path.join(directory, filename)
                polydata = read_geo(file_path)
                centroid = compute_polydata_centroid(polydata)
                centroids[file_path] = centroid
        return centroids

    def compute_distance(centroid1, centroid2):
        return sum([(c1 - c2) ** 2 for c1, c2 in zip(centroid1, centroid2)]) ** 0.5

    def extract_id_from_filename(filename):
        match = re.search(r'gt_cap_(\d+).vtp', filename)
        return int(match.group(1)) if match else None

    centroids_a = get_centroids(dir_a)
    centroids_b = get_centroids(dir_b)

    # Sort paths from dir_a based on the extracted ID
    ordered_paths_a = sorted(centroids_a.keys(), key=lambda path: extract_id_from_filename(os.path.basename(path)))

    matched_files = []
    unmatched_files_b = set(centroids_b.keys())

    for path_a in ordered_paths_a:
        centroid_a = centroids_a[path_a]
        best_match = None
        min_distance = sys.float_info.max  # set to max float value as a starting point
        
        for path_b, centroid_b in centroids_b.items():
            if path_b in unmatched_files_b:
                distance = compute_distance(centroid_a, centroid_b)
                if distance <= tolerance and distance < min_distance:
                    best_match = path_b
                    min_distance = distance
        
        # If there's a best match, add to matched files and remove from potential matches.
        if best_match:
            matched_files.append([read_geo(path_a), read_geo(best_match)])
            unmatched_files_b.remove(best_match)
        else:
            matched_files.append([read_geo(path_a), None])
    
    return matched_files, list(unmatched_files_b)

def write_gt_caps(specific_folder,gt_model_path):
        gt_cap_path = mkdir(specific_folder,'groundtruth_caps')
        gtpd = read_geo(gt_model_path)
        gtmd = sv.modeling.PolyData(gtpd)
        gt_face_id = gtmd.get_face_ids()
        gt_cap_ids = gtmd.identify_caps()
        keep_gt_true_cap = [i+1 for i in range(len(gt_cap_ids)) if gt_cap_ids[i] == True]
        for i , id in enumerate(keep_gt_true_cap):
            write_polydata(os.path.join(gt_cap_path,'gt_cap_'+str(id)+'.vtp'),gtmd.get_face_polydata(id))
        return gt_cap_path
    
def delete_everything(path):
    '''
    delete everything in a folder
    '''
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("error to delete")

def delete_and_rename_inlet_files(pred_caps, gt_caps, input_vtp, tolerance=0.1):
    """
    Delete and rename matching files in the specified directories based on the centroid of an input polydata.

    :param pred_caps: path to the predicted caps directory
    :param gt_caps: path to the ground truth caps directory
    :param input_vtp: input polydata file
    :param tolerance: tolerance for centroid matching
    """

    def get_centroids(directory):
        centroids = {}
        for filename in os.listdir(directory):
            if filename.endswith('.vtp'):
                file_path = os.path.join(directory, filename)
                polydata = read_geo(file_path)
                centroid = compute_polydata_centroid(polydata)
                centroids[file_path] = centroid
        return centroids

    # Calculate the centroid of the input polydata
    input_polydata = read_geo(input_vtp)
    input_centroid = compute_polydata_centroid(input_polydata)

    # Calculate centroids of all .vtp files in both directories
    centroids_pred = get_centroids(pred_caps)
    centroids_gt = get_centroids(gt_caps)

    # Find and operate on matching files
    for path_pred, centroid_pred in centroids_pred.items():
        for path_gt, centroid_gt in centroids_gt.items():
            distance_pred = ((centroid_pred[0] - input_centroid[0]) ** 2 +
                             (centroid_pred[1] - input_centroid[1]) ** 2 +
                             (centroid_pred[2] - input_centroid[2]) ** 2) ** 0.5
            distance_gt = ((centroid_gt[0] - input_centroid[0]) ** 2 +
                           (centroid_gt[1] - input_centroid[1]) ** 2 +
                           (centroid_gt[2] - input_centroid[2]) ** 2) ** 0.5

            if distance_pred <= tolerance and distance_gt <= tolerance:
                # Delete the matching file in the gt_caps directory
                os.remove(path_gt)
                
                # Rename the matching file in the pred_caps directory to "inlet.vtp"
                new_path_pred = os.path.join(pred_caps, "inlet.vtp")
                os.rename(path_pred, new_path_pred)
                break

def rename_highest_points_vtp(directory):
    """
    Rename the .vtp file with the highest number of points in the specified directory to 'wall.vtp'.

    :param directory: path to the directory containing .vtp files
    """

    def get_number_of_points(polydata):
        """Return the number of points in the given polydata."""
        return polydata.GetNumberOfPoints()

    max_points = -1
    highest_points_vtp_path = ""

    # Iterate through all .vtp files in the directory and find the one with the highest number of points
    for filename in os.listdir(directory):
        if filename.endswith('.vtp'):
            file_path = os.path.join(directory, filename)
            polydata = read_geo(file_path)
            num_points = get_number_of_points(polydata)
            if num_points > max_points:
                max_points = num_points
                highest_points_vtp_path = file_path

    # Rename the .vtp file with the highest number of points to 'wall.vtp'
    if highest_points_vtp_path:
        new_path = os.path.join(directory, "wall.vtp")
        os.rename(highest_points_vtp_path, new_path)

def write_pred_caps(specific_folder,sv_modeler):
     '''
     write all faces of pred surface including wall and inflow into a folder
     '''
     unmatched_pred_caps= mkdir(specific_folder,'pred_boundary_faces_unmatched')
     pred_face_ids = sv_modeler.get_face_ids()
     for i , id in enumerate(pred_face_ids):
         write_polydata(os.path.join(unmatched_pred_caps,'pred_cap_'+str(id)+'.vtp'),sv_modeler.get_face_polydata(id))
     return unmatched_pred_caps

def write_final_caps(specific_folder,sv_modeler):
    '''
     write all faces of surface including wall and inflow into a folder
     '''
    final_pred_caps= mkdir(specific_folder,'pred_boundary_faces_for_final_geo_not_for_simulation') 
    pred_face_ids = sv_modeler.get_face_ids()
    for i , id in enumerate(pred_face_ids):
        write_polydata(os.path.join(final_pred_caps,'pred_cap_'+str(id)+'.vtp'),sv_modeler.get_face_polydata(id))
    return final_pred_caps

def move_inflow_and_wall(src_folder, dest_folder):
    """
    Move 'inlet.vtp' and 'wall.vtp' from source folder to destination folder.

    :param src_folder: Path to the source folder.
    :param dest_folder: Path to the destination folder.
    """

    # List of special files to move
    special_files = ["inlet.vtp", "wall.vtp"]

    for file in special_files:
        src_path = os.path.join(src_folder, file)
    
        # Check if file exists in the source folder
        if os.path.exists(src_path):
            dest_path = os.path.join(dest_folder, file)
            shutil.move(src_path, dest_path)
        else:
            print("The files do not exist in the source folder.")

def copy_file(src_folder, filename, dest_folder):
    """
    Copies a file from the source folder to the destination folder.

    :param src_folder: Path to the source directory.
    :param filename: Name of the file to copy.
    :param dest_folder: Path to the destination directory.
    """
    try:
        # Constructing the full path to the file in the source folder
        src_file_path = os.path.join(src_folder, filename)
        
        # If destination folder does not exist, create it
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        # Constructing the full path in the destination folder
        dest_file_path = os.path.join(dest_folder, filename)
        
        # Copying the file to destination folder
        shutil.copy2(src_file_path, dest_file_path)
    except:
        print("Error copying file")

# def read_results_0d(fpath):
#     """
#     Read 0d simulation results from dictionary
#     """
#     return get_dict(fpath)
def move_all_except(src_dir, dest_dir, excluded_file="OneDSolver.exe"):
            
            # List all files and directories in the source directory
            for item in os.listdir(src_dir):
                src_item_path = os.path.join(src_dir, item)

                # If it's a file and is not the excluded file, move it
                if os.path.isfile(src_item_path) and item != excluded_file:
                    shutil.move(src_item_path, dest_dir)

def write_ordered_cap_for_pred_simulation(path,lst):
    for i in range(len(lst)):
        write_polydata(os.path.join(path,'outlet'+str(i)+'.vtp'),lst[i][1])

############################################################################################
######################## functions below parses rcrt.dat ##################################
def read_gt_solver_get_number_of_outlets(file_path):
    """
    Find the highest number that follows the pattern 'RCR_' in a file without using regex.

    :param file_path: Path to the file to be scanned
    :return: The highest number found following 'RCR_'. Returns -1 if none found.
    """
    highest_number = -1

    with open(file_path, 'r') as file:
        for line in file:
            if 'RCR_' in line:
                parts = line.split('RCR_')
                for part in parts[1:]:
                    number_str = ''
                    for char in part:
                        if char.isdigit():
                            number_str += char
                        else:
                            break
                    if number_str:
                        number = int(number_str)
                        if number > highest_number:
                            highest_number = number

    return highest_number

def find_rcr_lines_form_dictionary(file_path, max_number):
    """
    Find lines containing 'RCR_0' up to 'RCR_#' and store them in a dictionary.

    :param file_path: Path to the file to be scanned
    :param max_number: The highest number (#) to be considered
    :return: A dictionary with keys 'RCR_#' and values as lists of lines containing 'RCR_#'
    """
    rcr_dict = {'RCR_{}'.format(i): [] for i in range(max_number + 1)}

    with open(file_path, 'r') as file:
        for line in file:
            for i in range(max_number + 1):
                if 'RCR_{}'.format(i) in line:
                    rcr_dict['RCR_{}'.format(i)].append(line.strip())
    return rcr_dict

def pair_rcr_with_branch_id(rcr_dict):
    def extract_number(text):
        """
        Extracts the first sequence of digits in a given string.

        :param text: The string from which to extract the number
        :return: The extracted number as an integer. Returns None if no number is found
        """
        number_str = ''
        for char in text:
            if char.isdigit():
                number_str += char
            elif number_str:
                break

        return int(number_str) if number_str else None
    lst = []
    for i in rcr_dict:
        text = rcr_dict[i][0]
        branchid = extract_number(text)
        lst.append([branchid,i])
    return lst

def extract_rcr_from_solver(path_to_file, target_phrase):

    """
    Extracts the next three lines after the second occurrence of the target phrase in the file.

    :param path_to_file: Path to the text file
    :param target_phrase: The phrase to search for
    :return: A list containing the next three lines after the second occurrence of the target phrase
    """
    with open(path_to_file, 'r') as file:
        count = 0
        lines_to_return = []

        for line in file:
            if target_phrase in line:
                count += 1
                if count == 2:
                    for _ in range(3):
                        next_line = next(file, None)
                        if next_line is not None:
                            lines_to_return.append(next_line.strip()[4:])
                    break
    return lines_to_return
def pair_gt_bc_with_branch_id(file_path):

    """
    Master function to pair boundary conditions with their branch IDs.

    :param file_path: Path to the file to be scanned
    :return: A list of lists, each containing a branch ID and the boundary condition associated with it
    """
    max_number = read_gt_solver_get_number_of_outlets(file_path)
    rcr_dict = find_rcr_lines_form_dictionary(file_path, max_number)
    pair = pair_rcr_with_branch_id(rcr_dict)
    for i in range(len(pair)):
        pair[i].append(extract_rcr_from_solver(file_path, pair[i][1]))
    return pair

def pair_endpts_with_branchid(clpd):
    points = numpy_support.vtk_to_numpy(clpd.GetPoints().GetData())
    CenterlineID_for_each_point = numpy_support.vtk_to_numpy(clpd.GetPointData().GetArray('CenterlineId'))
    # if extracted centerline only has one line, there is no centerline attribute
    n_keys = len(CenterlineID_for_each_point[0])
    line_dict = {}

    #create dict with keys line0, line1, line2, etc
    for i in range(n_keys):
        key = "line{}".format(i)
        line_dict[key] = []

    for i in range(len(points)):
        for j in range(n_keys):
            if CenterlineID_for_each_point[i][j] == 1:
                key = "line{}".format(j)
                line_dict[key].append(points[i])
    
    for i in range(n_keys):
        key = "line{}".format(i)
        line_dict[key] = np.array(line_dict[key])
    # Done with spliting centerlines into dictioanry

    # find the end points of each line
    lst_of_end_pts = []
    # # append the very first point #### this is the inlet point
    # lst_of_end_pts.append(line_dict["line0"][0])
    # append the rest of the end points
    for i in range(n_keys):
        key = "line{}".format(i)
        lst_of_end_pts.append(line_dict[key][-1])
    nplst_of_endpts = np.array(lst_of_end_pts) #convert to numpy array

    branchid = numpy_support.vtk_to_numpy(clpd.GetPointData().GetArray('BranchId'))
    max_brach = max(branchid)
    # make dictionary points of each branch, key is branchid, value is list of points
    branch_dict = {}
    for i in range(max_brach+1):
        key = "branch{}".format(i)
        branch_dict[key] = []
    for i in range(len(points)):
        for j in range(max_brach+1):
            if branchid[i] == j:
                key = "branch{}".format(j)
                branch_dict[key].append(points[i])
    for i in range(max_brach+1):
        key = "branch{}".format(i)
        branch_dict[key] = np.array(branch_dict[key])
    # Done with spliting branches into dictioanry
    
    # contruct list to pair end points with branch id
    paired_endpts_branchid = []
    for i in nplst_of_endpts:
        for j in range(max_brach+1):
            key = "branch{}".format(j)
            if i in branch_dict[key]:
                paired_endpts_branchid.append([j,i.tolist()])

    return paired_endpts_branchid
def master_bc_list(gt_bc_pair,gt_endpt_branchid_pair):
    # create a copy 
    master_gt_bc_lst = gt_bc_pair.copy()
    for i in master_gt_bc_lst:
        for j in gt_endpt_branchid_pair:
            if i[0] == j[0]:
                i.append(j[1])
    return master_gt_bc_lst
############################################################################################
############################################################################################

####################### functions below writes rcrt.dat ##################################
def attach_rcr_to_simulation_enpts(master_gt_bc_lst,endpts_for_our_simu):
    for j in endpts_for_our_simu:
        for i in master_gt_bc_lst:
            if compute_distance(j[1],i[3]) < 0.5:
                
                j.append(i[2])
    return endpts_for_our_simu
# >>> for j in endpts_for_pred_simu:
# ...     print(j[2])
# ...     for i in master_gt_bc_lst:
# ...             print('distance w/ ' +str(i[0]) +' is')
# ...             compute_distance(j[1],i[3])



def reparse_solver_input_use_martins_0d(gt_file, pred_file):
    # Read the content from gt_file
    with open(gt_file, 'r') as f_gt:
        gt_lines = f_gt.readlines()
    
    for i in range(len(gt_lines)):
        if gt_lines[i] ==  '    "simulation_parameters": {\n':
            gt_to_paste = gt_lines[i:i+6]
    

    # Read the content from pred_file
    with open(pred_file, 'r') as f_pred:
        pred_lines = f_pred.readlines()

    for i in range(len(pred_lines)):
        if pred_lines[i] ==  '    "simulation_parameters": {\n':
            before = pred_lines[:i]
            after = pred_lines[i+6:]

    os.remove(pred_file)
    
    combined_content = before + gt_to_paste + after
    with open(pred_file, 'w') as f_pred:
        f_pred.writelines(combined_content)



def write_rcrt_dat_file(gt_inflow_directory,bc_matching_endts_lst):
    
    def write_file(path_to_file,text):
        with open(path_to_file, "a") as f:
            f.write(text + "\n")
            f.close()

    rcrt_file_path = os.path.join(gt_inflow_directory, 'rcrt.dat')
    # initiate the file 
    with open(rcrt_file_path, "a") as f:
        f.write('2' + "\n")
        f.write('2' + "\n")
        f.close()
    # write the rest of the file
    for i in bc_matching_endts_lst:
        if len(i) == 4: 
            write_file(rcrt_file_path,i[2])
            write_file(rcrt_file_path,i[3][0])
            write_file(rcrt_file_path,i[3][1])
            write_file(rcrt_file_path,i[3][2])
            write_file(rcrt_file_path,'0.0 0.0')
            write_file(rcrt_file_path,'1.0 0.0')
            write_file(rcrt_file_path,'2')

    #delete the last line
    with open(rcrt_file_path, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for i in lines[:-1]:
            f.write(i)
        f.truncate()
        f.close()
def add_face_name_to_our_sim(endpts_for_our_simu,gt_sim_boundary_face_path):
    for i in endpts_for_our_simu:
        for j in os.listdir(gt_sim_boundary_face_path):
            if compute_distance(i[1],compute_polydata_centroid(read_geo(os.path.join(gt_sim_boundary_face_path,j)))) < 0.3:
                i.append(j[:-4])
    return endpts_for_our_simu



def add_face_name_to_our_sim(endpts_for_our_simu, gt_sim_boundary_face_path):
    # Step 1: Prepare the data
    # Compute all centroids for the boundary faces
    gt_centroids = []
    face_names = []
    for j in os.listdir(gt_sim_boundary_face_path):
        if j[:-4] == 'inlet' or j[:-4] == 'wall':
            continue
        centroid = compute_polydata_centroid(read_geo(os.path.join(gt_sim_boundary_face_path, j)))
        gt_centroids.append(centroid)
        face_names.append(j[:-4])  # Strip the extension to get the face name
    # Step 2: Create a distance matrix
    num_endpts = len(endpts_for_our_simu)
    num_faces = len(gt_centroids)
    distance_matrix = np.zeros((num_endpts, num_faces))
    for i, endpt in enumerate(endpts_for_our_simu):
        for j, centroid in enumerate(gt_centroids):
            distance_matrix[i, j] = compute_distance(endpt[1], centroid)
    # Step 3: Use the Hungarian algorithm to find the optimal 1-to-1 matching
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    # Step 4: Assign the face names to the corresponding endpoints
    for i, j in zip(row_ind, col_ind):
        endpts_for_our_simu[i].append(face_names[j])
    return endpts_for_our_simu











############################################################################################

def reparse_solver_input_use_martins(gt_file, pred_file):
    # Read the content from gt_file
    with open(gt_file, 'r') as f_gt:
        gt_lines = f_gt.readlines()
        
    for i in range(len(gt_lines)):
        if gt_lines[i] ==  '# SOLVEROPTIONS CARD\n':
            gt_to_paste = gt_lines[i:]
    

    # Read the content from pred_file
    with open(pred_file, 'r') as f_pred:
        pred_lines = f_pred.readlines()

    for i in range(len(pred_lines)):
        if pred_lines[i] ==  '# SOLVEROPTIONS CARD\n':
            pred_to_keep = pred_lines[:i]

    os.remove(pred_file)

    combined_content = pred_to_keep + gt_to_paste
    with open(pred_file, 'w') as f_pred:
        f_pred.writelines(combined_content)

def extract_connected_loops(slicePolyData):
    """
    Extract all connected loops (regions) from the given slicePolyData.
    
    :param slicePolyData: vtkPolyData containing sliced loops.
    :return: List of vtkPolyData, each representing a connected loop.
    """
    # List to store each connected loop as a separate vtkPolyData
    connected_loops = []

    # Set up the connectivity filter to extract individual regions (loops)
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(slicePolyData)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()

    # Get the number of regions (loops) in the slice
    num_regions = connectivity.GetNumberOfExtractedRegions()
    
    for region_id in range(num_regions):
        # Extract the region corresponding to this loop
        connectivity.InitializeSpecifiedRegionList()
        connectivity.AddSpecifiedRegion(region_id)
        connectivity.SetExtractionModeToSpecifiedRegions()
        connectivity.Update()

        # Get the extracted loop as vtkPolyData and add it to the list
        loop_polydata = vtk.vtkPolyData()
        loop_polydata.DeepCopy(connectivity.GetOutput())  # Make a deep copy to avoid overwriting
        connected_loops.append(loop_polydata)

    return connected_loops


def project_points_to_2d_plane(points, normal_vector, d=0):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Define basis vectors u and v on the plane
    if np.allclose(normal_vector, [1, 0, 0]):
        u = np.array([0, 1, 0])
    else:
        u = np.array([1, 0, 0])
    
    v = np.cross(normal_vector, u)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    
    projected_points_2d = []
    
    for point in points:
        # Project the point onto the plane in 3D space
        projection = point - np.dot(point, normal_vector + d) * normal_vector
        
        # Convert the 3D projection to 2D coordinates on the plane
        x_2d = np.dot(projection, u)
        y_2d = np.dot(projection, v)
        
        projected_points_2d.append([x_2d, y_2d])
    
    return np.array(projected_points_2d)

def pre_shoelace(loop_pd, points_2d, loop_coords_3D):
    """
    process the points and order them based on connectivity
    
    get the area of the loop
    """
    # loop_coords_3D and projected_points_2d are the same length (index matches)
    # thus finding connectivity of loop_coords_3D is the same as projected_points_2d
    # get the connectivity of the loop_coords_3D
    # connected_pointId_pair = get_connected_point_id(loop_pd,loop_coords_3D)
    # pdb.set_trace()

    # get center of the loop
    #pdb.set_trace()
    center = np.mean(points_2d, axis=0)
    # sort the points based on angle
    angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    points_2d = points_2d[sorted_indices]

    #point_2d = subdivide_2D_coordinates_with_cubic_spline(points_2d)
    return points_2d


def shoelace(points_2d):
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))



def get_CenterlineSectionArea_ourway(vmtkclpd,surface_pd):
    # get all points in the centerline
    mst_pts = numpy_support.vtk_to_numpy(vmtkclpd.GetPoints().GetData())
    CenterlineSectionArea = np.zeros(len(mst_pts))
    # initialize variable
    shoelace_area = np.zeros(len(mst_pts))
    shoelace_n_radii_if_too_small = np.zeros(len(mst_pts))  
    # get CenterlineSectionNormal
    CenterlineSectionNormal = numpy_support.vtk_to_numpy(vmtkclpd.GetPointData().GetArray('CenterlineSectionNormal'))
    
    MaximumInscribedSphereRadius = numpy_support.vtk_to_numpy(vmtkclpd.GetPointData().GetArray('MaximumInscribedSphereRadius'))
    twod_coord_dict = {}

    # get junction_pts
    junction_pts = []
    bifurcationid = numpy_support.vtk_to_numpy(vmtkclpd.GetPointData().GetArray('BifurcationId'))
    for i in range(len(bifurcationid)):
        if bifurcationid[i] != -1:
            junction_pts = mst_pts[i]

    for i in range(len(mst_pts)):
        coord = mst_pts[i]
        # if pts is in bifurcation, use the maximum inscribed sphere radius to calculate the area
        if coord in junction_pts:
        # CORRECTION get the area using shoelace instead of maximum inscribed sphere radius
            projected_points_2d = []
            # create a plane at each point
            
            plane = vtk.vtkPlane()
            plane.SetOrigin(coord)
            plane.SetNormal(CenterlineSectionNormal[i])

            # use vtkcutter to get the sliced loop of the surface
            cutter = vtk.vtkCutter()
            cutter.SetCutFunction(plane)
            cutter.SetInputData(surface_pd)
            cutter.Update()
            slicePolyData = cutter.GetOutput()
            polydata = slicePolyData    
            #write_polydata(slicePolyData, f'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\total_slice_{i}.vtp')
            
            
            loop_pd_lst = extract_connected_loops(slicePolyData)
            # write all loops
            # for j in range(len(loop_pd_lst)):
            #     write_polydata(loop_pd_lst[j], f'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\ref_pt_{i}_loop{j}.vtp')
            #pdb.set_trace()
            # clean the polydata
            cleaned_pd_lst = []
            for loop_pd in loop_pd_lst:
                cleaner = vtk.vtkCleanPolyData()
                cleaner.SetInputData(loop_pd)
                cleaner.Update()
                cleaned_loop_pd = cleaner.GetOutput()
                cleaned_pd_lst.append(cleaned_loop_pd)
            
            # get coordinates of each loop
            loop_coords = [numpy_support.vtk_to_numpy(loop_pd.GetPoints().GetData()) for loop_pd in cleaned_pd_lst]
            #print(loop_coords[0][0] == loop_coords[1][0])
            # find out which loop has point closest to the reference point
            min_distance = float('inf')
            closest_loop_id = -1
            for j in range(len(loop_coords)):
                distance = np.min([np.linalg.norm(np.array(coord) - np.array(loop_coord)) for loop_coord in loop_coords[j]])
                if distance < min_distance:
                    min_distance = distance
                    closest_loop_id = j
            #print("closest loop id is {}".format(closest_loop_id))
            loop_pd = cleaned_pd_lst[closest_loop_id]
            #write_polydata(cleaned_pd_lst[closest_loop_id], f'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\single_slice\\single_slice_{i}.vtp')
            
            # use enclosed loop to get area#################
            loop_coords_3d = loop_coords[closest_loop_id]

            # project to 2D
            points_2d = project_points_to_2d_plane(loop_coords_3d,CenterlineSectionNormal[i])
            
            #visualize_2d_coordinates(projected_points_2d)
            
            points_2d = pre_shoelace(loop_pd, points_2d, loop_coords_3d)
           
            #method 1: shoelace
            
            CenterlineSectionArea[i] = shoelace(points_2d)
            shoelace_n_radii_if_too_small[i]= shoelace(points_2d)
            

            # method 2: use maximum inscribed sphere radius for small area
            if shoelace_n_radii_if_too_small[i] < 0.1: # if too small, use maximum inscribed sphere radius
                shoelace_n_radii_if_too_small[i] = np.pi*MaximumInscribedSphereRadius[i]*MaximumInscribedSphereRadius[i]
            
            
        else:
            projected_points_2d = []
            # create a plane at each point
            
            plane = vtk.vtkPlane()
            plane.SetOrigin(coord)
            plane.SetNormal(CenterlineSectionNormal[i])

            # use vtkcutter to get the sliced loop of the surface
            cutter = vtk.vtkCutter()
            cutter.SetCutFunction(plane)
            cutter.SetInputData(surface_pd)
            cutter.Update()
            slicePolyData = cutter.GetOutput()
            polydata = slicePolyData    
            #write_polydata(slicePolyData, f'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\total_slice_{i}.vtp')
            
            
            loop_pd_lst = extract_connected_loops(slicePolyData)
            # write all loops
            # for j in range(len(loop_pd_lst)):
            #     write_polydata(loop_pd_lst[j], f'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\ref_pt_{i}_loop{j}.vtp')
            #pdb.set_trace()
            # clean the polydata
            cleaned_pd_lst = []
            for loop_pd in loop_pd_lst:
                cleaner = vtk.vtkCleanPolyData()
                cleaner.SetInputData(loop_pd)
                cleaner.Update()
                cleaned_loop_pd = cleaner.GetOutput()
                cleaned_pd_lst.append(cleaned_loop_pd)
            
            # get coordinates of each loop
            loop_coords = [numpy_support.vtk_to_numpy(loop_pd.GetPoints().GetData()) for loop_pd in cleaned_pd_lst]
            #print(loop_coords[0][0] == loop_coords[1][0])
            # find out which loop has point closest to the reference point
            min_distance = float('inf')
            closest_loop_id = -1
            for j in range(len(loop_coords)):
                distance = np.min([np.linalg.norm(np.array(coord) - np.array(loop_coord)) for loop_coord in loop_coords[j]])
                if distance < min_distance:
                    min_distance = distance
                    closest_loop_id = j
            #print("closest loop id is {}".format(closest_loop_id))
            loop_pd = cleaned_pd_lst[closest_loop_id]
            #write_polydata(cleaned_pd_lst[closest_loop_id], f'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\single_slice\\single_slice_{i}.vtp')
            
            # use enclosed loop to get area#################
            loop_coords_3d = loop_coords[closest_loop_id]

            # project to 2D
            points_2d = project_points_to_2d_plane(loop_coords_3d,CenterlineSectionNormal[i])
            
            #visualize_2d_coordinates(projected_points_2d)
            
            points_2d = pre_shoelace(loop_pd, points_2d, loop_coords_3d)
            twod_coord_dict[i] = points_2d
            #method 1: shoelace
            
            CenterlineSectionArea[i] = shoelace(points_2d)
            shoelace_n_radii_if_too_small[i]= shoelace(points_2d)
            

            # method 2: use maximum inscribed sphere radius for small area
            if shoelace_n_radii_if_too_small[i] < 0.1: # if too small, use maximum inscribed sphere radius
                shoelace_n_radii_if_too_small[i] = np.pi*MaximumInscribedSphereRadius[i]*MaximumInscribedSphereRadius[i]
            
            
    master = {}
    master['shoelace_radii_at_junction'] = CenterlineSectionArea
    master['shoelace_radii_if_too_small'] = shoelace_n_radii_if_too_small
    lst_too_small = []
    
    for a in range(len(CenterlineSectionArea)):
        if CenterlineSectionArea[a] < 0.09:
            lst_too_small.append(a)
    #pdb.set_trace()

    CenterlineSectionArea[0] = CenterlineSectionArea[1]
    return CenterlineSectionArea,lst_too_small,twod_coord_dict,master


def add_attributes(attribute_name,list_to_add,pd):
    # add the list to the polydata
    list_to_add = numpy_support.numpy_to_vtk(list_to_add)
    list_to_add.SetName(attribute_name)
    pd.GetPointData().AddArray(list_to_add)

def delete_attributes(attribute_name,pd):
    pd.GetPointData().RemoveArray(attribute_name)
    

def run_1d_simulation(input_file_path):
        OneDSolver_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\svOneDSolver\\svOneDSolver_build\\bin\\Release\\OneDSolver.exe'
        
        # Use subprocess.run and set stdout and stderr to None to inherit the output to the console
        subprocess.run([OneDSolver_path, input_file_path], 
                    cwd='C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\svOneDSolver\\svOneDSolver_build\\bin\\Release',
                    stdout=None,  # Display stdout in the terminal
                    stderr=None,  # Display stderr in the terminal
                    shell=False)  # shell=False is usually safer
        
        OneDSolver_folder_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\svOneDSolver\\svOneDSolver_build\\bin\\Release'
        return OneDSolver_folder_path

def set_path_name():
    "path to ur result folder"
    result_master_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\MIROS_paper\\rom_res\\0d_res_seg3_worked'

    "path to ur svproject(gt model) folder"
    svproject_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\MIROS_paper\\rom_sv_projects'

    'path to ur Numi model folder' 'all file starts with Numi_ then the model name'
    Numi_model_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\MIROS_paper\\rom_Numi_surf'

    'path to ur gt centerline folder'
    gt_cl_path =   'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\centerlines'

    martin_1d_input_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\Martin_result\\input_1d'

    return result_master_folder, svproject_path, Numi_model_path, gt_cl_path, martin_1d_input_path


def main():
    result_master_folder, svproject_path, Numi_model_path, gt_cl_path, martin_1d_input_path = set_path_name()
    
    
    for filename in os.listdir(Numi_model_path):  #iterate through all the models in the folder
        result_master_folder, svproject_path, Numi_model_path, gt_cl_path, martin_1d_input_path = set_path_name()
        
        
        # print('debug: filename ' + filename)
        pred_surf_path = os.path.join(Numi_model_path, filename)
        # print('debug: pred_surf_path' + pred_surf_path)
        specific_folder = mkdir(result_master_folder,filename[:-4])
        print(filename)
        # print('debug: specific_folder' + specific_folder)
        
        if filename[0:16] == 'final_assembly_o':  # original
            filename = filename[24:33]+filename[-4:] #remove Numi_ from the filename
        elif filename[0:16] == 'final_assembly_u':
            filename = filename[25:34]+filename[-4:] # upsampled
        else:
            filename = filename[0:9]+'.vtp'
        # specific_folder = specific_folder[:-4]
        print(filename)
        
        # print('debug: filename' + filename)

        # create log file for each model
        
        edit_log(specific_folder, "boxcutting model: "+filename)
        
        
        gt_cl_path = os.path.join(gt_cl_path,filename)
        gt_model_path = os.path.join(svproject_path, os.path.splitext(filename)[0], 'Models', filename)
        gt_inflow_pd_path = os.path.join(svproject_path, os.path.splitext(filename)[0],'Simulations',os.path.splitext(filename)[0],'mesh-complete','mesh-surfaces','inflow.vtp')
        path_to_write= os.path.join(specific_folder, 'clipped_')+filename
        mdl_path = os.path.join(svproject_path, os.path.splitext(filename)[0], 'Models', filename[:-4]+'.mdl')
        bc_file_path = os.path.join(svproject_path, os.path.splitext(filename)[0],'Simulations',os.path.splitext(filename)[0],'rcrt.dat')
        gt_inflow_file_path = os.path.join(svproject_path, os.path.splitext(filename)[0],'flow-files')
        martin_1d_input = os.path.join(martin_1d_input_path, filename[:-4]+'_1d.in')
        # print('debug: pred_surf_path ' + pred_surf_path)
        # print('debug: specific_folder ' + specific_folder)
        # print('debug: filename ' + filename)
        # print('debug: gt_cl_path ' + gt_cl_path)
        # print('debug: gt_model_path ' + gt_model_path)
        # print('debug: path_to_write ' + path_to_write)
        # print('debug: gt_inflow_pd_path ' + gt_inflow_pd_path)

        # boxcut the model
        box_scale = 4
        try:
            boxcut_model,clippingbox = bryan_boxcut(gt_cl_path,pred_surf_path,filename,box_scale, gt_cl_path)
            
        except:
            print('boxcutting failed')
            edit_log(specific_folder, "boxcutting failed at Numi surf pre-processing")
            continue

        edit_log(specific_folder, "boxcutting finished")
        
        # save the boxcut model
        write_polydata(os.path.join(specific_folder, 'clipped_')+filename, boxcut_model)
        write_polydata(os.path.join(specific_folder, 'box_for_')+filename, clippingbox)
        edit_log(specific_folder, "boxcutted model saved, still need validation")
        
        rom_ready_vtp_path = os.path.join(specific_folder, 'clipped_'+filename)
        
        print('debug: rom_ready_vtp_path ' + rom_ready_vtp_path)
        
        model_vtp = read_surface(rom_ready_vtp_path,'vtp',None)
        extract = sv.modeling.PolyData(model_vtp)
        extract.compute_boundary_faces(90.0) # 90 degree angle --> form 1 face (the wall)   // under modeling 
        extracted_pd = extract.get_polydata() # extract pd from modeling class    #under modeling
        filled = sv.vmtk.cap(extracted_pd)   #filled holes --> form inlets and outlets   #under vmtk
        filled_md =sv.modeling.PolyData(filled) # create new object (polydata with face id attribute)
        
        # save the filled model
        write_polydata(os.path.join(specific_folder, 'filled_'+ filename), filled_md.get_polydata())
        edit_log(specific_folder, "filled model saved")
        full_pd = filled_md.get_polydata() #get polydata from new object
        edit_log(specific_folder, "remeshing to get finer model for mesh generation")
        first_time_remeshed = svmeshtool.remesh_polydata(full_pd,0.1,0.1)
        write_polydata(os.path.join(specific_folder, 'first_time_remeshed.vtp'), first_time_remeshed)
        edit_log(specific_folder, "saved remeshed model as first_time_remeshed.vtp")
        
        # if ROM_ready_pd is None:
        #     edit_log(specific_folder, "remeshing failed")
        #     continue
        # else:
        #     edit_log(specific_folder, "remeshing finished,start exportiing caps")

        ready_md = sv.modeling.PolyData(first_time_remeshed) # create new object (finer polydata with face id attribute)
        # export caps 
        boundary_face_path= mkdir(specific_folder,'pred_boundary_faces')

        # first write all the faces of ground truth model  into a folder in order of faceID. face1.vtp, face2.vtp ....
        # then write all the faces of the predicted model into another folder 
        # match the faces of the predicted model with the ground truth model, in order of the ground truth faces
        # write the matched faces into a new folder with naming convention

        # writing the ground truth model faces into a folder (make into function) WORKED!!
        try:
            gt_cap_path = write_gt_caps(specific_folder,gt_model_path)
        except:
            edit_log(specific_folder, "something is wrong with the groundtruth model! Can't get face data")
            print('something is wrong with the groundtruth model! Can\'t get face data')
            continue
        edit_log(specific_folder, "groundtruth caps exported")
        unmatched_pred_cap_path = write_pred_caps(specific_folder,ready_md)

        # rename wall
        rename_highest_points_vtp(unmatched_pred_cap_path) # rename the largest vtp file to wall.vtp
        

        # match the faces of the predicted model with the ground truth model, in order of the ground truth faces
        
    

        # renamed inlet and deleted inlet vtp from gt folder
        delete_and_rename_inlet_files(unmatched_pred_cap_path, gt_cap_path, gt_inflow_pd_path, tolerance=0.5)
        move_inflow_and_wall(unmatched_pred_cap_path, boundary_face_path)
        
        # we have dir_a and dir_b containing matching caps and unmatched caps
        matched_caps_lst,unmatched_lst = match_files(gt_cap_path,unmatched_pred_cap_path, tolerance=0.5)
        
        # if number of complete pairs in matched_caps_lst is less than 2, then we can't do anything
        def captured_more_than_2_outlets(matched_caps_lst):
            count = 0
            for i in matched_caps_lst:
                if i[1] != None:
                    count += 1
            if count >= 1: # used to be 2 but that would mean there are three outlets, 1 for inlet, 2 for outlets
                #fix this later
                return True
            else:
                return False 
        if captured_more_than_2_outlets(matched_caps_lst) == False:
            edit_log(specific_folder, "pred surf does not capture more than 2 outlets, can't run simulation")
            print('pred surf does not capture more than 2 outlets, can\'t run simulation')
            continue


        ###################### determine scenario for simulation: does pred surf capture all outlets?
        ###################### matched_caps_lst has [[gt_cap,pred_cap],[gt_cap,None]...]
        ###################### determine if need to rerun gt sim with selective outlets. 
        ###################### Lets get the centerline class and mesh class for gt sim first! 
        # evaluate if pred has captured all gt caps 
        def evaluate_matched_caps(matched_caps_lst):
            lst_of_uncaptured_gt_caps = []
            for i in matched_caps_lst:
                if i[1] == None:
                    lst_of_uncaptured_gt_caps.append(i[0])
            return lst_of_uncaptured_gt_caps
        
        def match_uncaptured_gt_caps_with_faceID(uncaptured_gt_caps,gt_cap_path,tol=0.01):
            lst_gt_cap_centroids = []
            lst_uncaptured_gt_cap_centroids = []
            for i in uncaptured_gt_caps:
                centroid = compute_polydata_centroid(i)
                lst_uncaptured_gt_cap_centroids.append(centroid)
            
            for i in os.listdir(gt_cap_path):
                file_path = os.path.join(gt_cap_path, i)
                pd = read_geo(file_path)
                centroid = compute_polydata_centroid(pd)
                lst_gt_cap_centroids.append([i,centroid])
            
            # now we have 2 lists of centroids, match them by returning a lst of faceID
            # dic,indices = match_coord(lst_uncaptured_gt_cap_centroids,lst_gt_cap_centroids)
            return  lst_uncaptured_gt_cap_centroids, lst_gt_cap_centroids
        
        def find_matching_numbers(uncap_lst, lst_centroid):
            # Create a dictionary to map centroid points to their corresponding number in the filename
            centroid_to_number = {centroid: int(filename.split('_')[2].split('.')[0]) for filename, centroid in lst_centroid}

            # Create a list to hold the matching numbers
            matching_numbers = []

            # Compare each point in uncap_lst with the centroids
            for point in uncap_lst:
                if point in centroid_to_number:
                    # If a match is found, append the corresponding number to the matching_numbers list
                    matching_numbers.append(centroid_to_number[point])

            return matching_numbers


        hi = evaluate_matched_caps(matched_caps_lst)
        if hi == []: # means pred surf captures all gt caps! result is given, run martin's 1d solver file
            # # run martin's 1d solver file
            # print('pred surf captures all gt caps! result is given, run martin\'s 1d solver file')
            print('pred surf captures all gt caps! change discretization and re run gt sim') 
            martin_1d_input = os.path.join(martin_1d_input_path, filename[:-4]+'_1d.in')
            gt_bc_pair = pair_gt_bc_with_branch_id(martin_1d_input)
            gtclpd = read_geo(gt_cl_path)
            ##########
            ########## re-setup gt sim with higher num of segments per branch
            ##########
            
            modified_gt_md = sv.modeling.PolyData(read_geo(gt_model_path))
            modified_gt_pd = modified_gt_md.get_polydata()
            wall_id_to_combine = modified_gt_md.identify_caps() # this never fails becuase we are doing it on vmr surface (all branch captured)
            wall_id_to_combine = [i+1 for i in range(len(wall_id_to_combine)) if wall_id_to_combine[i] == False]
            smallest = min(wall_id_to_combine)
            wall_id_to_combine.remove(smallest)
            for i in wall_id_to_combine:
                modified_gt_md.combine_faces(smallest,[i])
            modified_gt_pd = modified_gt_md.get_polydata()
            write_polydata(os.path.join(specific_folder, 'modified_gt_surf_for_sim.vtp'), modified_gt_pd)
            gt_sim_surf_path = os.path.join(specific_folder, 'modified_gt_surf_for_sim.vtp')
            gt_sim_boundary_face_path= mkdir(specific_folder,'GT_sim_boundary_faces_all_captured_not_used_for_sim')
            faceids_to_write = modified_gt_md.get_face_ids()
            faceids_to_write.remove(smallest)
            write_polydata(os.path.join(gt_sim_boundary_face_path, 'wall.vtp'), modified_gt_md.get_face_polydata(smallest))
            for i in faceids_to_write:
                write_polydata(os.path.join(gt_sim_boundary_face_path,'outlet'+str(i)+'.vtp'),modified_gt_md.get_face_polydata(i))
            # now we have a modified gt model with selective outlets, write out the polydata
            modified_gt_pd = modified_gt_md.get_polydata()
            write_polydata(os.path.join(specific_folder, 'modified_gt_surf_for_sim.vtp'), modified_gt_pd)
            gt_sim_surf_path = os.path.join(specific_folder, 'modified_gt_surf_for_sim.vtp')
            # done with GT surf, get parameter for simulation!!

            # set up gt sim
            gt_sim_boundary_face_path= mkdir(specific_folder,'GT_sim_boundary_faces')
            faceids_to_write = modified_gt_md.get_face_ids()
            faceids_to_write.remove(smallest)
            write_polydata(os.path.join(gt_sim_boundary_face_path, 'wall.vtp'), modified_gt_md.get_face_polydata(smallest))
            for i in faceids_to_write:
                write_polydata(os.path.join(gt_sim_boundary_face_path,'outlet'+str(i)+'.vtp'),modified_gt_md.get_face_polydata(i))
            # pdb.set_trace()


            # evaluate inlet
            gt_inlet_centroid = compute_polydata_centroid(read_geo(gt_inflow_pd_path))
            # calculate centroid of everything in gt_sim_boundary_face_path and find the closest one to gt_inlet_centroid and rename the file to inlet.vtp
            gt_inlet_centroid_lst = []
            for i in os.listdir(gt_sim_boundary_face_path):
                file_path = os.path.join(gt_sim_boundary_face_path, i)
                pd = read_geo(file_path)
                centroid = compute_polydata_centroid(pd)
                gt_inlet_centroid_lst.append([i,centroid])
            # now we have a list of all the centroids of the faces in gt_sim_boundary_face_path, find the closest one to gt_inlet_centroid
            for i in gt_inlet_centroid_lst:
                distance = compute_distance(i[1],gt_inlet_centroid)
                print('distance between GT/VMR inlet and seqseg inlet is ' + str(distance))
                if distance < 0.3:
                    new_path = os.path.join(gt_sim_boundary_face_path, 'inlet.vtp')
                    os.rename(os.path.join(gt_sim_boundary_face_path,i[0]), new_path)
                    break

            gt_inflow_directory = mkdir(specific_folder,'gt_inflow_files')
            # copy inflow files
            copy_file(gt_inflow_file_path, 'inflow_1d.flow', gt_inflow_directory)
            copy_file(gt_inflow_file_path, 'inflow_3d.flow', gt_inflow_directory)
            
            # form rcrt.dat
            gt_bc_pair = pair_gt_bc_with_branch_id(martin_1d_input)
            GT_Params = setup_GT_ROM_parameters(specific_folder,filename,1)

            Cl = Centerlines()
            try:
                Cl.extract_center_lines(GT_Params)
            except:
                edit_log(specific_folder, "centerline extraction failed!!!")
                continue
            # centerline extraction worked!! construct rcrt.dat file!
            gtclpd = read_geo(gt_cl_path) # GET ORIGINAL VMR surf to process bcs
            #pdb.set_trace()
            
            gt_endpt_branchid_pair = pair_endpts_with_branchid(gtclpd)
            gt_bc_pair = pair_gt_bc_with_branch_id(martin_1d_input)
            master_gt_bc_lst = master_bc_list(gt_bc_pair,gt_endpt_branchid_pair) #most important!!
            
            #>>> master_gt_bc_lst[i][0] is branchid
            #>>> master_gt_bc_lst[i][1] is RCR_# in martin's solver
            #>>> master_gt_bc_lst[i][2] is the BOUNDARY CONDITION!!  
            #>>> master_gt_bc_lst[i][3] is the coordinates of the end points in that branch
            extracted_gt_cl_pd = read_geo(os.path.join(specific_folder,'extracted_GT_sim_centerlines.vtp'))
            
            endpts_for_our_simu = pair_endpts_with_branchid(extracted_gt_cl_pd)
            #>>> endpts_for_our_simu[i][0] is branchid
            #>>> endpts_for_our_simu[i][1] is the coordinates of the end points in that branch
            add_face_name_to_our_sim(endpts_for_our_simu,gt_sim_boundary_face_path)
            gt_sim_bc_matching_endpts_lst = attach_rcr_to_simulation_enpts(master_gt_bc_lst,endpts_for_our_simu)
            # pdb.set_trace() 
            
            write_rcrt_dat_file(gt_inflow_directory,gt_sim_bc_matching_endpts_lst)
            
            gtmsh = mesh.Mesh()
            
            gtmsh.generate(GT_Params,Cl) 
            reparse_solver_input_use_martins(martin_1d_input, os.path.join(specific_folder,'GT_1d_solver_output.in'))
            gt_input_file_path = os.path.join(specific_folder,'GT_1d_solver_output.in')

            # uncomment the following
            # try:
            #     OneDSolver_folder_path = run_1d_simulation(gt_input_file_path)
            #     OneD_result_path = mkdir(specific_folder,'gt_1d_results')
            #     move_all_except(OneDSolver_folder_path, OneD_result_path)
            #     print('Done with GT 1d simulation')
            #     copy_file(specific_folder,'GT_1d_solver_output.in', OneD_result_path)
            # except:

            #     edit_log(specific_folder, "GT 1d simulation failed!!!")

            ### now lets do 0d simulation
            Params_0D = setup_GT_ROM_parameters(specific_folder,filename,0)
            Cl = Centerlines()
            extracted_gt_cl_pd_path = os.path.join(specific_folder,'extracted_GT_sim_centerlines.vtp')
            Cl.read(Params_0D,extracted_gt_cl_pd_path)
            gtmsh_0D = mesh.Mesh()
            gtmsh_0D.set_outlet_face_names(Params_0D)
            gtmsh_0D.outlet_face_names_file = os.path.join(specific_folder,'VMR_centerlines_outlets.dat')
            gtmsh_0D.generate(Params_0D,Cl)
            martin_0d_input_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\Martin_result\\input_0d'
            martin_0d_input = os.path.join(martin_0d_input_path, filename[:-4]+'_0d.in')
            reparse_solver_input_use_martins_0d(martin_0d_input, os.path.join(specific_folder,'GT_0d_solver_output.json'))
            

            ##################
            ##################
            ##################
            
            
        elif hi != []: # means pred surf does not capture all gt caps!
            uncap_lst, lst_centorid = match_uncaptured_gt_caps_with_faceID(hi,gt_cap_path,tol=0.01)
            ############
            ############ HERE, we figured out what gt caps are not captured by pred surf and the faceID of them in GT surface
            
            face_id_to_merge_to_wall_for_gt_sim = find_matching_numbers(uncap_lst, lst_centorid)
            
            ############ start setting up gt sim with selective outlets
            modified_gt_md = sv.modeling.PolyData(read_geo(gt_model_path))
            wall_id_to_combine = modified_gt_md.identify_caps()
            print(wall_id_to_combine)
            wall_id_to_combine = [i+1 for i in range(len(wall_id_to_combine)) if wall_id_to_combine[i] == False]
            print(wall_id_to_combine)
            wall_id_to_combine = wall_id_to_combine + face_id_to_merge_to_wall_for_gt_sim
            print(wall_id_to_combine)
            smallest = min(wall_id_to_combine)
            wall_id_to_combine.remove(smallest)
            for i in wall_id_to_combine:
                modified_gt_md.combine_faces(smallest,[i])    

            # now we have a modified gt model with selective outlets, write out the polydata
            modified_gt_pd = modified_gt_md.get_polydata()
            write_polydata(os.path.join(specific_folder, 'modified_gt_surf_for_sim.vtp'), modified_gt_pd)
            gt_sim_surf_path = os.path.join(specific_folder, 'modified_gt_surf_for_sim.vtp')
            # done with GT surf, get parameter for simulation!!

            # set up gt sim
            gt_sim_boundary_face_path= mkdir(specific_folder,'GT_sim_boundary_faces')
            faceids_to_write = modified_gt_md.get_face_ids()
            faceids_to_write.remove(smallest)
            write_polydata(os.path.join(gt_sim_boundary_face_path, 'wall.vtp'), modified_gt_md.get_face_polydata(smallest))
            for i in faceids_to_write:
                write_polydata(os.path.join(gt_sim_boundary_face_path,'outlet'+str(i)+'.vtp'),modified_gt_md.get_face_polydata(i))
            # pdb.set_trace()


            # evaluate inlet
            gt_inlet_centroid = compute_polydata_centroid(read_geo(gt_inflow_pd_path))
            # calculate centroid of everything in gt_sim_boundary_face_path and find the closest one to gt_inlet_centroid and rename the file to inlet.vtp
            gt_inlet_centroid_lst = []
            for i in os.listdir(gt_sim_boundary_face_path):
                file_path = os.path.join(gt_sim_boundary_face_path, i)
                pd = read_geo(file_path)
                centroid = compute_polydata_centroid(pd)
                gt_inlet_centroid_lst.append([i,centroid])
            # now we have a list of all the centroids of the faces in gt_sim_boundary_face_path, find the closest one to gt_inlet_centroid
            for i in gt_inlet_centroid_lst:
                distance = compute_distance(i[1],gt_inlet_centroid)
                print(distance)
                if distance < 0.3:
                    new_path = os.path.join(gt_sim_boundary_face_path, 'inlet.vtp')
                    os.rename(os.path.join(gt_sim_boundary_face_path,i[0]), new_path)
                    break

            gt_inflow_directory = mkdir(specific_folder,'gt_inflow_files')
            # copy inflow files
            copy_file(gt_inflow_file_path, 'inflow_1d.flow', gt_inflow_directory)
            copy_file(gt_inflow_file_path, 'inflow_3d.flow', gt_inflow_directory)

            # form rcrt.dat
            gt_bc_pair = pair_gt_bc_with_branch_id(martin_1d_input)

            
            GT_Params = setup_GT_ROM_parameters(specific_folder,filename,1)

            
            Cl = Centerlines()
            try:
                if not os.path.exists(os.path.join(specific_folder,'extracted_GT_sim_centerlines.vtp')):
                    Cl.extract_center_lines(GT_Params)
                else:
                    Cl.read(GT_Params,os.path.join(specific_folder,'extracted_GT_sim_centerlines.vtp'))

            except:
                edit_log(specific_folder, "gt centerline extraction failed!!!")
                print('gt centerline extraction failed!!!')
                continue
            
            # see if centerline file exiset
            if os.path.join(specific_folder,'extracted_GT_sim_centerlines.vtp') is None:
                edit_log(specific_folder, "gt centerline extraction failed!!! Centerlinefile does not exist")
                print('gt centerline extraction failed!!! Centerlinefile does not exist')
                continue
            extracted_gt_cl_pd = read_geo(os.path.join(specific_folder,'extracted_GT_sim_centerlines.vtp'))
             #detect if centerline extraction is successful for simulation: see if it has BranchId
            if extracted_gt_cl_pd.GetPointData().GetArray('BranchId') is None:
                edit_log(specific_folder, "centerline extraction failed!!!  No BranchId in the centerline file")
                print('gt centerline extraction failed!!! No BranchId in the centerline file')
                continue
            else:
                edit_log(specific_folder, "centerline extraction finished, start mesh generation")

            #pdb.set_trace()
            # centerline extraction worked!! construct rcrt.dat file!
            gtclpd = read_geo(gt_cl_path)
            gt_endpt_branchid_pair = pair_endpts_with_branchid(gtclpd)
            
            gt_bc_pair = pair_gt_bc_with_branch_id(martin_1d_input)
            master_gt_bc_lst = master_bc_list(gt_bc_pair,gt_endpt_branchid_pair) #most important!!

            #>>> master_gt_bc_lst[i][0] is branchid
            #>>> master_gt_bc_lst[i][1] is RCR_# in martin's solver
            #>>> master_gt_bc_lst[i][2] is the BOUNDARY CONDITION!!  
            #>>> master_gt_bc_lst[i][3] is the coordinates of the end points in that branch
            endpts_for_our_simu = pair_endpts_with_branchid(extracted_gt_cl_pd)
            #>>> endpts_for_our_simu[i][0] is branchid
            #>>> endpts_for_our_simu[i][1] is the coordinates of the end points in that branch
            add_face_name_to_our_sim(endpts_for_our_simu,gt_sim_boundary_face_path)
            gt_sim_bc_matching_endpts_lst = attach_rcr_to_simulation_enpts(master_gt_bc_lst,endpts_for_our_simu)
            # pdb.set_trace() 
            
            write_rcrt_dat_file(gt_inflow_directory,gt_sim_bc_matching_endpts_lst)
            
            gtmsh = mesh.Mesh()
            gtmsh.outlet_face_names_file = os.path.join(specific_folder,'centerline_outlets.dat')
            
            gtmsh.generate(GT_Params,Cl) 
            reparse_solver_input_use_martins(martin_1d_input, os.path.join(specific_folder,'GT_1d_solver_output.in'))
            gt_input_file_path = os.path.join(specific_folder,'GT_1d_solver_output.in')
            
            # uncomment the following
            # try:
            #     OneDSolver_folder_path = run_1d_simulation(gt_input_file_path)
            #     OneD_result_path = mkdir(specific_folder,'gt_1d_results')
            #     move_all_except(OneDSolver_folder_path, OneD_result_path)
            #     print('Done with GT 1d simulation')
            #     copy_file(specific_folder,'GT_1d_solver_output.in', OneD_result_path)
            # except:

            #     edit_log(specific_folder, "GT 1d simulation failed!!!")

            Params_0D = setup_GT_ROM_parameters(specific_folder,filename,0)
            Cl = Centerlines()
            extracted_gt_cl_pd_path = os.path.join(specific_folder,'extracted_GT_sim_centerlines.vtp')
            Cl.read(Params_0D,extracted_gt_cl_pd_path)
            gtmsh_0D = mesh.Mesh()
            gtmsh_0D.set_outlet_face_names(Params_0D)
            gtmsh_0D.outlet_face_names_file = os.path.join(specific_folder,'VMR_centerlines_outlets.dat')
            gtmsh_0D.generate(Params_0D,Cl)
            martin_0d_input_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\Martin_result\\input_0d'
            martin_0d_input = os.path.join(martin_0d_input_path, filename[:-4]+'_0d.in')
            reparse_solver_input_use_martins_0d(martin_0d_input, os.path.join(specific_folder,'GT_0d_solver_output.json'))
            
                
            
           
        ############ done with modified gt sim!##########################
        #################################################################
        #################################################################
        # #test
        # for i in range(len(matched_caps_lst)):
        #     write_polydata(os.path.join(boundary_face_path,'outlet'+str(i)+'.vtp'),matched_caps_lst[i][1])
        # for i in range(len(matched_caps_lst)):
        #     write_polydata(os.path.join(boundary_face_path,'outlet'+str(i+100)+'.vtp'),matched_caps_lst[i][0])
        # pdb.set_trace()

        #test if predicted surface has all the gt_caps by see if each element in matched_caps_lst has 2 elements (matched pair)
        
        for i in matched_caps_lst:
            if len(i) != 2:
                edit_log(specific_folder, "predicted surface does not have all the ground truth caps, consider to adjust tolerance and boxcut scale")
                print("predicted surface does not have all the ground truth caps")
                print('running simulation with partial VMR outletes')
                print("consider to adjust tolerance and boxcut scale")

        number_of_files = len([f for f in os.listdir(gt_cap_path) if os.path.isfile(os.path.join(gt_cap_path, f))])
        if len(matched_caps_lst) != number_of_files:
            edit_log(specific_folder, "predicted surface does not have all the ground truth caps, consider to adjust tolerance and boxcut scale")
            print("predicted surface does not have all the ground truth caps, consider to adjust tolerance and boxcut scale")
            

        print(unmatched_lst)


        def get_unmatched_face_id(unmatched_lst):
            # Use a regular expression to extract digits from the filenames in the unmatched_lst
            unmatched_face_id = [''.join(re.findall(r'\d+', i.split('\\')[-1])) for i in unmatched_lst]
            unmatched_face_id = [int(i) for i in unmatched_face_id]
            return unmatched_face_id

        unmatched_face_id = get_unmatched_face_id(unmatched_lst)
        print(unmatched_face_id)
        
        print(ready_md.get_face_ids())
        # merge the unmatched caps into one wall
        for i in unmatched_face_id:
            ready_md.combine_faces(1,[i])
        
        ROM_ready_pd = ready_md.get_polydata()
        second_time_remeshed = svmeshtool.remesh_polydata(ROM_ready_pd,0.1,0.1)
        write_polydata(os.path.join(specific_folder, 'export_PRED_ROM_ready_surface.vtp'), second_time_remeshed)
        
        
        
        # sort and match outlet again after merging unmatched caps into one wall
        # new wall now!! update wall.vtp
        boundary_faces_for_final_geo_not_for_simulation_path = write_final_caps(specific_folder,ready_md)
        delete_everything(boundary_face_path)
        delete_everything(gt_cap_path)
        gt_cap_path = write_gt_caps(specific_folder,gt_model_path)
        rename_highest_points_vtp(boundary_faces_for_final_geo_not_for_simulation_path)
        delete_and_rename_inlet_files(boundary_faces_for_final_geo_not_for_simulation_path, gt_cap_path, gt_inflow_pd_path, tolerance=0.5)
        move_inflow_and_wall(boundary_faces_for_final_geo_not_for_simulation_path, boundary_face_path)
        matched_caps_lst, unmatched_files_b_list = match_files(gt_cap_path,boundary_faces_for_final_geo_not_for_simulation_path, tolerance=0.5)
        # >>>[[(vtkCommonDataModelPython.vtkPolyData)0000025E7F6BD8E8, (vtkCommonDataModelPython.vtkPolyData)0000025E7F6BD888], ...]
        
        write_ordered_cap_for_pred_simulation(boundary_face_path, matched_caps_lst)
        

# lst = parse_mdl_to_list('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\sv projects\\0146_1001\\Models\\0146_1001.mdl')
# >>> lst
# [[1, 'wall', 'wall'], [10, 'renal_left', 'cap'], [11, 'renal_right', 'cap'], [12, 'SMA', 'cap'], 
# [2, 'inflow', 'cap'], [3, 'celiac_hepatic', 'cap'], [4, 'celiac_splenic', 'cap'],
# [5, 'ext_iliac_left', 'cap'], [6, 'ext_iliac_right', 'cap'], [7, 'IMA', 'cap'],
# [8, 'int_iliac_left', 'cap'], [9, 'int_iliac_right', 'cap']]
# lst, face_name_lst = sort_and_reorder(lst)
# >>> lst
# [[2, 'inflow', 'cap'], [3, 'celiac_hepatic', 'cap'], [4, 'celiac_splenic', 'cap'], 
#  [5, 'ext_iliac_left', 'cap'], [6, 'ext_iliac_right', 'cap'], [7, 'IMA', 'cap'], [8, 'int_iliac_left', 'cap'],
#  [9, 'int_iliac_right', 'cap'], [10, 'renal_left', 'cap'], [11, 'renal_right', 'cap'], [12, 'SMA', 'cap']]
# >>> face_name_lst
# ['celiac_hepatic', 'celiac_splenic', 'ext_iliac_left', 'ext_iliac_right', 'IMA', 'int_iliac_left', 'int_iliac_right', 'renal_left', 'renal_right', 'SMA']
# lines = insert_outlet_names_to_rcrt('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\sv projects\\0146_1001\\Simulations\\0146_1001\\rcrt.dat',face_name_lst)
# write_to_file('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\ROM results\\inflow_files\\rcrt_from_parsing.dat',lines)
        
        files = [f for f in os.listdir(boundary_face_path) if os.path.isfile(os.path.join(boundary_face_path, f))]
        # >>> ['inlet.vtp', 'outlet0.vtp', 'outlet1.vtp', 'outlet2.vtp', 'outlet3.vtp', 'outlet4.vtp', 'wall.vtp']
        # get rid of inlet.vtp and wall.vtp
        files = files[1:-1]
        ### get rid of .vtp in all
        cap_names = [f[:-4] for f in files]
        # >>> ['outlet0', 'outlet1', 'outlet2', 'outlet3', 'outlet4']
        # write the outlet names into rcrt.dat
        
        pred_inflow_directory = mkdir(specific_folder,'pred_inflow_files')
        copy_file(gt_inflow_file_path, 'inflow_1d.flow', pred_inflow_directory)
        copy_file(gt_inflow_file_path, 'inflow_3d.flow', pred_inflow_directory)
        





        # write_to_file(os.path.join(pred_inflow_directory,'rcrt.dat'),lines)
        # copy inflow files
        

        
        
        PRED_Params = setup_PRED_ROM_parameters(specific_folder,filename,1)
        
        Cl = Centerlines()
        

        try:
            # see if it exist already, if not, extract it, else no need to extract
            if not os.path.exists(os.path.join(specific_folder,'extracted_pred_centerlines.vtp')):
                Cl.extract_center_lines(PRED_Params)
            else:
                Cl.read(PRED_Params,os.path.join(specific_folder,'extracted_pred_centerlines.vtp'))
        except:
            edit_log(specific_folder, "pred centerline extraction failed!!!")
            continue
        
       # see if centerline file exiset
        if os.path.join(specific_folder,'extracted_pred_centerlines.vtp') is None:
            edit_log(specific_folder, "pred centerline extraction failed!!! Centerlinefile does not exist")
            continue
        extracted_pred_cl_pd = read_geo(os.path.join(specific_folder,'extracted_pred_centerlines.vtp'))
         #detect if centerline extraction is successful for simulation: see if it has BranchId
        if extracted_pred_cl_pd.GetPointData().GetArray('BranchId') is None:
            edit_log(specific_folder, "pred centerline extraction failed!!!")
            continue
        else:
            edit_log(specific_folder, "pred centerline extraction finished, start mesh generation")

        #########################################################
        ###################### modify area ######################
        # # after checking if centerline extraction is successful, now replace the CenterlineSectionArea with shoelace area
        # # first save the original CenterlineSectionArea and write it as a file in the specific folder
        # # the format should: each line is a number, the number is the area of the point
        # vmtkarea = extracted_pred_cl_pd.GetPointData().GetArray('CenterlineSectionArea')
        # vmtkarea = numpy_support.vtk_to_numpy(vmtkarea)
        # write_to_file(os.path.join(specific_folder,'vmtk_area.txt'),vmtkarea)
        # delete_attributes('CenterlineSectionArea',extracted_pred_cl_pd)
        # our_surf = read_geo(rom_ready_vtp_path)
        # CenterlineSectionArea,lst_too_small,twod_coord_dict,master_area = get_CenterlineSectionArea_ourway(extracted_pred_cl_pd,read_geo(os.path.join(specific_folder, 'export_PRED_ROM_ready_surface.vtp')))
        # add_attributes('CenterlineSectionArea',CenterlineSectionArea,extracted_pred_cl_pd)
        # write_polydata(os.path.join(specific_folder, 'extracted_pred_centerlines.vtp'), extracted_pred_cl_pd)
        # write_to_file(os.path.join(specific_folder,'our_area.txt'),CenterlineSectionArea)
        #########################################################
        #########################################################


        # Prepare inflow.dat file for OneD simulation
        #pdb.set_trace()
        gt_endpt_branchid_pair = []
        gt_bc_pair = []
        master_gt_bc_lst = []
        pred_endpt_branchid_pair = []
        endpts_for_pred_simu = []
        pred_sim_bc_matching_endpts_lst = []
        gtclpd = read_geo(gt_cl_path)
        gt_endpt_branchid_pair = pair_endpts_with_branchid(gtclpd)
        gt_bc_pair = pair_gt_bc_with_branch_id(martin_1d_input)
        master_gt_bc_lst = master_bc_list(gt_bc_pair,gt_endpt_branchid_pair) #most important!!
        pred_endpt_branchid_pair = pair_endpts_with_branchid(extracted_pred_cl_pd)
        endpts_for_pred_simu = add_face_name_to_our_sim(pred_endpt_branchid_pair,boundary_face_path)
        pred_sim_bc_matching_endpts_lst = attach_rcr_to_simulation_enpts(master_gt_bc_lst,endpts_for_pred_simu)
        write_rcrt_dat_file(pred_inflow_directory,pred_sim_bc_matching_endpts_lst)
        #pdb.set_trace()
        msh = mesh.Mesh()
        msh.outlet_face_names_file = os.path.join(specific_folder,'centerline_outlets.dat')
        # pdb.set_trace()
        msh.generate(PRED_Params,Cl) 

        
        # # parse solver input file so that it matches with Martin's
        martin_1d_input = os.path.join(martin_1d_input_path, filename[:-4]+'_1d.in')
        reparse_solver_input_use_martins(martin_1d_input, os.path.join(specific_folder,'PRED_1d_solver_output.in'))


        # get 0D solver input file first before running 1D
        Params_0D = setup_PRED_ROM_parameters(specific_folder,filename,0)
        Cl = Centerlines()
        extracted_gt_cl_pd_path = os.path.join(specific_folder,'extracted_pred_centerlines.vtp')
        Cl.read(Params_0D,extracted_gt_cl_pd_path)
        predmsh_0D = mesh.Mesh()
        predmsh_0D.set_outlet_face_names(Params_0D)
        predmsh_0D.outlet_face_names_file = os.path.join(specific_folder,'VMR_centerlines_outlets.dat')
        predmsh_0D.generate(Params_0D,Cl)
        martin_0d_input_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\Martin_result\\input_0d'
        martin_0d_input = os.path.join(martin_0d_input_path, filename[:-4]+'_0d.in')
        reparse_solver_input_use_martins_0d(martin_0d_input, os.path.join(specific_folder,'PRED_0d_solver_output.json'))
            



        # uncomment the following
        # # Run OneD simulation
        # pred_input_file_path = os.path.join(specific_folder,'PRED_1d_solver_output.in')
        # try:
        #     OneDSolver_folder_path = run_1d_simulation(pred_input_file_path)
        # except:
        #     edit_log(specific_folder, "1d simulation failed!!!")
        #     continue      
        
        # # move 1d results to specific folder
        # OneD_result_path = mkdir(specific_folder,'pred_1d_results')
        # copy_file(specific_folder, 'PRED_1d_solver_output', OneD_result_path)
        # move_all_except(OneDSolver_folder_path, OneD_result_path)


        continue




if __name__=='__main__':
    
    pdb.set_trace()
    main()
    
    # input_file_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\result_for_176\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\1d_solver_output.in'
    # run_1d_simulation(input_file_path)