import vtk
import numpy as np
import os
import sys
from vtk.util import numpy_support
import pdb

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


def write_polydata(polydata, fname):
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == '.vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    writer.SetFileName(fname)
    writer.SetInputData(polydata)
    writer.Write()

def bryan_get_clipping_parameters(clpd): #get all three parameters
    points = numpy_support.vtk_to_numpy(clpd.GetPoints().GetData())
    CenterlineID_for_each_point = numpy_support.vtk_to_numpy(clpd.GetPointData().GetArray('CenterlineId'))
    radii = numpy_support.vtk_to_numpy(clpd.GetPointData().GetArray('MaximumInscribedSphereRadius'))
    n_keys = len(CenterlineID_for_each_point[0])
    line_dict = {}

    #create dict with keys line0, line1, line2, etc
    for i in range(n_keys):
        key = f"line{i}"  
        line_dict[key] = []

    for i in range(len(points)):
        for j in range(n_keys):
            if CenterlineID_for_each_point[i][j] == 1:
                key = f"line{j}"
                line_dict[key].append(points[i])
    
    for i in range(n_keys):
        key = f"line{i}"  
        line_dict[key] = np.array(line_dict[key])
    # Done with spliting centerliens into dictioanry

    # find the end points of each line
    lst_of_end_pts = []
    # append the very first point
    lst_of_end_pts.append(line_dict["line0"][0])
    # append the rest of the end points
    for i in range(n_keys):
        key = f"line{i}"  
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
        key = f"line{i}"
        line = line_dict[key]
        tangent_vector = line[-1] - line[-2]
        unit_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
        unit_tangent_vectors.append(unit_tangent_vector)


    return nplst_of_endpts, nplst_radii_at_caps, unit_tangent_vectors



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

import vtk
import numpy as np

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
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'RegionId')
    threshold.SetUpperThreshold(largestRegionLabel)
    threshold.SetLowerThreshold(largestRegionLabel)
    threshold.Update()

    # Convert the output of the threshold filter to vtkPolyData
    largestRegionPolyData = vtk.vtkGeometryFilter()
    largestRegionPolyData.SetInputData(threshold.GetOutput())
    largestRegionPolyData.Update()

    return largestRegionPolyData.GetOutput()

def bryan_generate_oriented_boxes(endpts,unit_tan_vectors,radius,box_scale=3):
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

    # # Write the oriented box to a .vtp file
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName(output_file)
    # writer.SetInputData(box_surfaces.GetOutput())
    # writer.Write()
    return box_surfaces.GetOutput(),pd_lst

def find_folder_path_within_parent_folder(parent_folder):
    # List all items in the parent folder
    items = os.listdir(parent_folder)
    
    # Find the first directory in the parent folder (assuming only one randomly named folder)
    random_folder = None
    for item in items:
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            random_folder = item_path
            break
    
    if random_folder is None:
        raise ValueError("No subfolder found in the parent folder")
    return random_folder

def bryan_fillholes(pd):
    # Fill the holes
    fill = vtk.vtkFillHolesFilter()
    fill.SetInputData(pd)
    fill.SetHoleSize(100000.0)
    fill.Update()
    return fill.GetOutput()

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

def get_inlet_coord(sv_proj_path):
    parent_path = os.path.join(sv_proj_path, "Simulations")
    path = find_folder_path_within_parent_folder(parent_path)
    path = os.path.join(path, "mesh-complete",'mesh-surfaces','inflow.vtp')
    inlet = read_polydata(path)
    inlet_centroid = compute_polydata_centroid(inlet)
    return inlet_centroid

def get_source_and_target_coords(path_vmr_cl,sv_proj_path):
    """
    returns in form of np.array
    """
    inlet_centroid = get_inlet_coord(sv_proj_path)
    endpts, _ , _ = bryan_get_clipping_parameters(read_polydata(path_vmr_cl))
    #remove the point that cloest to the inlet centroid
    inlet_centroid = np.array(inlet_centroid)
    dist = np.linalg.norm(endpts-inlet_centroid,axis=1)
    idx = np.argmin(dist)
    print(f"index of the cloest point to the inlet centroid is {idx}")
    print(f"the distance from that endpoint to the inlet centroid is {dist[idx]}")
    source_coord = endpts[idx]
    target_coords = np.delete(endpts,idx,axis=0)
    return source_coord, target_coords
    
def box_clip(seqseg_surf,path_vmr_cl):
    clpd = read_polydata(path_vmr_cl)
    endpts, radii, unit_vecs = bryan_get_clipping_parameters(clpd)
    boxpd, boxpdlst = bryan_generate_oriented_boxes(endpts,unit_vecs,radii,4)
    clippedpd = bryan_clip_surface(read_polydata(seqseg_surf),boxpd)
    filled = bryan_fillholes(clippedpd)
    largest = keep_largest_surface(filled)
    return largest

def get_source_and_target_coords_and_box_clip(path_vmr_cl,sv_proj_path,seqseg_surf):

    """
    source_coord: np.array
    target_coords: np.array
    box_trimmed: vtkPolyData
    """

    source_coord, target_coords = get_source_and_target_coords(path_vmr_cl,sv_proj_path)
    box_trimmed = box_clip(seqseg_surf,path_vmr_cl)
    #write_polydata(box_trimmed,'path\\box_trimmed.vtp')
    return source_coord, target_coords, box_trimmed


def example():
    path_vmr_cl = 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\centerlines\\0176_0000.vtp'
    sv_proj_path = 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\rom_sv_projects\\0176_0000'
    seqseg_surf = 'c:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2024_spring\\0176_0000.vtp'
    
    source_coord, target_coords, box_trimmed = get_source_and_target_coords_and_box_clip(path_vmr_cl,sv_proj_path,seqseg_surf)
    print(f"source_coord is {source_coord}")
    print(f"target_coords is {target_coords}")
    print(box_trimmed)
    #write_polydata(box_trimmed,'path\\box_trimmed.vtp')



