import sys
# import ur sv_master python folder
sys.path.append('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\SimVascular-master\\Python\\site-packages')
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
#import vmtk


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
    fill.SetHoleSize(500.0)
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

def setup_ROM_parameters(path,model_name,order):
    if order == 1:
        Params = Parameters() # put everything is params
        Params.density = 1.06
        Params.output_directory = path
        Params.boundary_surfaces_dir = os.path.join(path,'boundary_faces')
        Params.inlet_face_input_file = 'inlet.vtp'
        Params.centerlines_output_file = os.path.join(path,'extracted_centerlines.vtp')
        Params.surface_model = os.path.join(path,'export_ROM_ready_surface.vtp')
        Params.inflow_input_file = os.path.join(path,'inflow_files','inflow_1d.flow')
        Params.model_order = 1
        Params.solver_output_file = '1d_solver_output.in' # need this to write out the solver file
        Params.model_name = model_name
        Params.outflow_bc_type = 'rcr' #rcr, resistance or coronary vmr 3d sim uses this and it calls 
        Params.outflow_bc_file =  os.path.join(path,'inflow_files\\')
        # create own .dat file? 
        Params.uniform_bc = False
        return Params
    elif order == 0:
        Params = Parameters() # put everything is params
        Params.density = 1.06
        Params.output_directory = path
        Params.boundary_surfaces_dir = os.path.join(path,'boundary_faces')
        Params.inlet_face_input_file = 'inlet.vtp'
        Params.centerlines_output_file = os.path.join(path,'extracted_centerlines.vtp')
        Params.surface_model = os.path.join(path,'export_ROM_ready_surface.vtp')
        Params.inflow_input_file = os.path.join(path,'inflow_files','inflow_1d.flow')
        Params.model_order = 0
        Params.solver_output_file = '0d_solver_output.json' # need this to write out the solver file
        Params.model_name = model_name
        Params.outflow_bc_type = 'rcr' #rcr, resistance or coronary vmr 3d sim uses this and it calls 
        Params.outflow_bc_file =  os.path.join(path,'inflow_files\\')
        # create own .dat file? 
        Params.uniform_bc = False
        #watch out!
        Params.model_order= 0
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
            file.write(line + "\n")
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
     unmatched_pred_caps= mkdir(specific_folder,'boundary_faces_unmatched')
     pred_face_ids = sv_modeler.get_face_ids()
     for i , id in enumerate(pred_face_ids):
         write_polydata(os.path.join(unmatched_pred_caps,'pred_cap_'+str(id)+'.vtp'),sv_modeler.get_face_polydata(id))
     return unmatched_pred_caps

def write_final_caps(specific_folder,sv_modeler):
    '''
     write all faces of pred surface including wall and inflow into a folder
     '''
    final_pred_caps= mkdir(specific_folder,'boundary_faces_for_final_geo_not_for_simulation') 
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

def write_ordered_cap_for_simulation(path,lst):
    for i in range(len(lst)):
        write_polydata(os.path.join(path,'outlet'+str(i)+'.vtp'),lst[i][1])

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


    def resample_centerline(Centerlines):
        """
        Function to resample centerline
        """
        centerline_calc = vmtkscripts.vmtkCenterlineResampling()
        centerline_calc.Centerlines = Centerlines
        centerline_calc.Execute()
        print('Centerline Resampler Executed')




def run_1d_simulation(input_file_path):
    OneDSolver_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\svOneDSolver\\svOneDSolver_build\\bin\\Release\\OneDSolver.exe'
    subprocess.check_output([OneDSolver_path, input_file_path], cwd = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\svOneDSolver\\svOneDSolver_build\\bin\\Release')
    OneDSolver_folder_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\svOneDSolver\\svOneDSolver_build\\bin\\Release'
    return OneDSolver_folder_path

def set_path_name():
    "path to ur result folder"
    result_master_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\Pipeline_testing_result\\Pipeline_correct_bc'

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
    
    
    for filename in os.listdir(Numi_model_path):  #iterate through all the models in the folder
        result_master_folder, svproject_path, Numi_model_path, gt_cl_path, martin_1d_input_path = set_path_name()
        
        
        # print('debug: filename ' + filename)
        pred_surf_path = os.path.join(Numi_model_path, filename)
        # print('debug: pred_surf_path' + pred_surf_path)
        specific_folder = mkdir(result_master_folder,filename[:-4])
        print(filename)
        # print('debug: specific_folder' + specific_folder)
        
        if filename[0:16] == 'final_assembly_o':
            filename = filename[24:33]+filename[-4:] #remove Numi_ from the filename
        elif filename[0:16] == 'final_assembly_u':
            filename = filename[25:34]+filename[-4:]
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
        
        # print('debug: pred_surf_path ' + pred_surf_path)
        # print('debug: specific_folder ' + specific_folder)
        # print('debug: filename ' + filename)
        # print('debug: gt_cl_path ' + gt_cl_path)
        # print('debug: gt_model_path ' + gt_model_path)
        # print('debug: path_to_write ' + path_to_write)
        # print('debug: gt_inflow_pd_path ' + gt_inflow_pd_path)

        # boxcut the model
        box_scale = 4.5
        try:
            boxcut_model,clippingbox = bryan_boxcut(gt_cl_path,pred_surf_path,filename,box_scale, gt_cl_path)
            
        except:
            edit_log(specific_folder, "boxcutting failed")
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
        boundary_face_path= mkdir(specific_folder,'boundary_faces')

        # first write all the faces of ground truth model  into a folder in order of faceID. face1.vtp, face2.vtp ....
        # then write all the faces of the predicted model into another folder 
        # match the faces of the predicted model with the ground truth model, in order of the ground truth faces
        # write the matched faces into a new folder with naming convention

        # writing the ground truth model faces into a folder (make into function) WORKED!!
        try:
            gt_cap_path = write_gt_caps(specific_folder,gt_model_path)
        except:
            edit_log(specific_folder, "something is wrong with the groundtruth model! Can't get face data")
            continue
        edit_log(specific_folder, "groundtruth caps exported: nothing is wrong with groundtruth model")
        unmatched_pred_cap_path = write_pred_caps(specific_folder,ready_md)

        # rename wall
        rename_highest_points_vtp(unmatched_pred_cap_path) # rename the largest vtp file to wall.vtp
        

        # match the faces of the predicted model with the ground truth model, in order of the ground truth faces
        
        # renamed inlet and deleted inlet vtp from gt folder
        delete_and_rename_inlet_files(unmatched_pred_cap_path, gt_cap_path, gt_inflow_pd_path, tolerance=0.2)
        move_inflow_and_wall(unmatched_pred_cap_path, boundary_face_path)
        
        # we have dir_a and dir_b containing matching caps and unmatched caps
        matched_caps_lst,unmatched_lst = match_files(gt_cap_path,unmatched_pred_cap_path, tolerance=0.2)

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
                print("predicted surface does not have all the ground truth caps, consider to adjust tolerance and boxcut scale")
                continue
        number_of_files = len([f for f in os.listdir(gt_cap_path) if os.path.isfile(os.path.join(gt_cap_path, f))])
        if len(matched_caps_lst) != number_of_files:
            edit_log(specific_folder, "predicted surface does not have all the ground truth caps, consider to adjust tolerance and boxcut scale")
            print("predicted surface does not have all the ground truth caps, consider to adjust tolerance and boxcut scale")
            continue

        print(unmatched_lst)


        def get_unmatched_face_id(unmatched_lst):
            # Use a regular expression to extract digits from the filenames in the unmatched_lst
            unmatched_face_id = [''.join(re.findall(r'\d+', i.split('\\')[-1])) for i in unmatched_lst]
            unmatched_face_id = [int(i) for i in unmatched_face_id]
            return unmatched_face_id

        unmatched_face_id = get_unmatched_face_id(unmatched_lst)
        print(unmatched_face_id)
        
        # merge the unmatched caps into one wall
        for i in unmatched_face_id:
            ready_md.combine_faces(i,[1])
        
        ROM_ready_pd = ready_md.get_polydata()
        second_time_remeshed = svmeshtool.remesh_polydata(ROM_ready_pd,0.1,0.1)
        write_polydata(os.path.join(specific_folder, 'export_ROM_ready_surface.vtp'), second_time_remeshed)
        
        pdb.set_trace()
        # new wall now!! update wall.vtp
        boundary_faces_for_final_geo_not_for_simulation_path = write_final_caps(specific_folder,ready_md)
        delete_everything(boundary_face_path)
        delete_everything(gt_cap_path)
        gt_cap_path = write_gt_caps(specific_folder,gt_model_path)
        rename_highest_points_vtp(boundary_faces_for_final_geo_not_for_simulation_path)
        delete_and_rename_inlet_files(boundary_faces_for_final_geo_not_for_simulation_path, gt_cap_path, gt_inflow_pd_path, tolerance=0.2)
        move_inflow_and_wall(boundary_faces_for_final_geo_not_for_simulation_path, boundary_face_path)
        matched_caps_lst, unmatched_files_b_list = match_files(gt_cap_path,boundary_faces_for_final_geo_not_for_simulation_path, tolerance=0.2)

        
        
        
        write_ordered_cap_for_simulation(boundary_face_path, matched_caps_lst)
        

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
        lines = insert_outlet_names_to_rcrt(bc_file_path,cap_names)
        inflow_directory = mkdir(specific_folder,'inflow_files')
        write_to_file(os.path.join(inflow_directory,'rcrt.dat'),lines)
        # copy inflow files
        copy_file(gt_inflow_file_path, 'inflow_1d.flow', inflow_directory)
        copy_file(gt_inflow_file_path, 'inflow_3d.flow', inflow_directory)



        Params = setup_ROM_parameters(specific_folder,filename,1)
        
        Cl = Centerlines()
        try:
            Cl.extract_center_lines(Params)
        except:
            edit_log(specific_folder, "centerline extraction failed!!!")
            continue
        
       # see if centerline file exiset
        if os.path.join(specific_folder,'extracted_centerlines.vtp') is None:
            edit_log(specific_folder, "centerline extraction failed!!! Centerlinefile does not exist")
            continue
        extracted_cl_pd = read_geo(os.path.join(specific_folder,'extracted_centerlines.vtp'))
         #detect if centerline extraction is successful for simulation: see if it has BranchId
        if extracted_cl_pd.GetPointData().GetArray('BranchId') is None:
            edit_log(specific_folder, "centerline extraction failed!!!")
            continue
        else:
            edit_log(specific_folder, "centerline extraction finished, start mesh generation")

        # Prepare inflow.dat file for OneD simulation

        msh = mesh.Mesh()
        msh.outlet_face_names_file = os.path.join(specific_folder,'centerline_outlets.dat')
        msh.generate(Params,Cl) 
        
        zerod_Params = setup_ROM_parameters(specific_folder,filename,0)
        
        msh.generate(zerod_Params,Cl)

        # # parse solver input file so that it matches with Martin's
        martin_1d_input = os.path.join(martin_1d_input_path, filename[:-4]+'_1d.in')
        reparse_solver_input_use_martins(martin_1d_input, os.path.join(specific_folder,'1d_solver_output.in'))



        # Run OneD simulation
        input_file_path = os.path.join(specific_folder,'1d_solver_output.in')
        try:
            OneDSolver_folder_path = run_1d_simulation(input_file_path)
        except:
            edit_log(specific_folder, "1d simulation failed!!!")
            continue
        

        
        
        # move 1d results to specific folder
        

        OneD_result_path = mkdir(specific_folder,'1d_results')
        move_all_except(OneDSolver_folder_path, OneD_result_path)





if __name__=='__main__':
    # pdb.set_trace()
    main()
    # input_file_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_fall\\result_for_176\\final_assembly_original_0176_0000_3d_fullres_0_393__surface\\1d_solver_output.in'
    # run_1d_simulation(input_file_path)