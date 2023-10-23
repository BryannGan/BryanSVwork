import vtk
import os
from vtk.util import numpy_support
import math
import numpy as np

def smooth_surface(polydata, smoothingIterations,band,angle):
    passBand = band
    featureAngle = angle
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(smoothingIterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(featureAngle)
    smoother.SetPassBand(passBand)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    print('smoothing finished')
    return smoother.GetOutput()

   

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

def get_tridata(polydata):
    tri_converter = vtk.vtkTriangleFilter()
    tri_converter.SetInputData(polydata)
    tri_converter.Update()
    return tri_converter.GetOutput() 

def write_polydata_file(input_data,fname):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(input_data)
    writer.Update()
    writer.Write()
    print('done writing file')

def write_ply_file(points_data,fname):
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(fname)
    writer.SetInputData(points_data)
    writer.Write()
    print('done writing file')

def get_points(polydata):
    # Get the points of the VTK dataset
    vtk_points = polydata.GetPoints()

    # Convert the vtkPoints object into a vtkPolyData object
    vtk_polydata_points = vtk.vtkPolyData()
    vtk_polydata_points.SetPoints(vtk_points)
    
    return vtk_polydata_points
    

def decimation_pro(polydata,ratio):
    deci = vtk.vtkDecimatePro()
    deci.SetTargetReduction(ratio)
    deci.SetInputData(polydata)
    deci.PreserveTopologyOn()
    deci.SplittingOff()
    deci.BoundaryVertexDeletionOff()
    deci.Update()
    print('done decimation')
    return deci.GetOutput()

def compute_volume(polydata):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(polydata)
    volume = mass.GetVolume()
    print("volume is ", volume)
    return volume



# def surface_remesh(polydata): #NEED WORK
#     normals = vtk.vtkPolyDataNormals()
#     normals.SetInputData(polydata)
#     normals.ComputePointNormalsOn()
#     normals.Update()

#     # Combine the original polydata with the computed point normals
#     appender = vtk.vtkAppendPolyData()
#     appender.AddInputData(polydata)
#     appender.AddInputData(normals.GetOutput())
#     appender.Update()

#     # Create a mapper
#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputData(appender.GetOutput())
#     mapper.SetScalarVisibility(True)
#     mapper.SetColorModeToDefault()

#     # Create an actor
#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)

#     # # Create a renderer and add the actor to it
#     # renderer = vtk.vtkRenderer()
#     # renderer.AddActor(actor)

#     # # Create a window and set the renderer as its active renderer
#     # window = vtk.vtkRenderWindow()
#     # window.AddRenderer(renderer)
#     # window.SetSize(800, 800)

#     # # Create an interactor and start the rendering loop
#     # interactor = vtk.vtkRenderWindowInteractor()
#     # interactor.SetRenderWindow(window)
#     # interactor.Initialize()
#     # interactor.Start()
#     print("finishd generating remesh surface")

def butterfly_subdiv(polydata,number_of_iter):
    subdivide = vtk.vtkButterflySubdivisionFilter()
    subdivide.SetInputData(polydata)
    subdivide.SetNumberOfSubdivisions(number_of_iter)
    subdivide.Update()
    output_polydata = subdivide.GetOutput()
    print('done with subdiv')
    return output_polydata

def pipeline(fname):
    directory = 'pipeline'

    ini_file = read_geo(fname)
    write_polydata_file(ini_file,fname + 'pipeline_0.vtp')

    iter1,passband1,angle1 = 5,0.01,100 #############
    fir_smooth = smooth_surface(ini_file,iter1,passband1,angle1)
    write_polydata_file(fir_smooth,fname + 'pipeline_1_'+str(iter1)+'_'+str(passband1)+'_'+str(angle1)+'.vtp')

    deci_ratio = 0.95
    decied = decimation_pro(fir_smooth,deci_ratio)
    write_polydata_file(decied,fname + 'pipeline_2_'+ str(deci_ratio)+'.vtp')

    iter2,passband2,angle2 = 20,0.01,60 #############
    sec_smooth = smooth_surface(decied,iter2,passband2,angle2)
    write_polydata_file(sec_smooth,fname + 'pipeline_3_'+str(iter2)+'_'+str(passband2)+'_'+str(angle2)+'.vtp')

    deci_ratio = 0.01
    decied2 = decimation_pro(sec_smooth,deci_ratio)
    write_polydata_file(decied2,fname + 'pipeline_4_'+ str(deci_ratio)+'.vtp')

    iter3,passband3,angle3 = 50,0.05,60 #############
    third_smooth = smooth_surface(decied2,iter3,passband3,angle3)
    write_polydata_file(third_smooth,fname + 'pipeline_5_'+str(iter3)+'_'+str(passband3)+'_'+str(angle3)+'.vtp')
    
    # subdiv_iter = 1
    # butterfly_subdiv(sec_smooth,subdiv_iter)
    # write_polydata_file(sec_smooth,fname + 'pipeline_4_'+str(subdiv_iter)+'.vtp')
    
    print('No inital and final remesh, but done with smoothing,decimation,smoothing, and subdivision')


def obeserve_effect_of_decimation_after_smoothing(fname, iter,passband,angle):
    init_pdata = read_geo(fname)
    smoothed = smooth_surface(init_pdata,iter,passband,angle)
    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]:
        decied = decimation_pro(smoothed,i)
        write_polydata_file(decied,'deci'+str(i)+'_on_smooth_'+str(iter)+'_'+str(passband)+'_'+str(angle)+'.vtp')

    print('Observe effect now')

def reposition_on_top_1(polydata1, polydata2):
    # Get the bounds of the two models
    bounds1 = polydata1.GetBounds()
    bounds2 = polydata2.GetBounds()

    # Compute the translation vector
    tx = (bounds1[0] + bounds1[1]) / 2 - (bounds2[0] + bounds2[1]) / 2
    ty = (bounds1[2] + bounds1[3]) / 2 - (bounds2[2] + bounds2[3]) / 2
    tz = (bounds1[4] + bounds1[5]) / 2 - (bounds2[4] + bounds2[5]) / 2

    # Translate the second model
    transform = vtk.vtkTransform()
    transform.Translate(tx, ty, tz)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(polydata2)
    transformFilter.Update()
    return transformFilter.GetOutput()

def reposition_on_top_2(gt_polydata, my_polydata):
    
    # Get the centroids of each model
    gt_center = gt_polydata.GetCenter()
    my_center = my_polydata.GetCenter()

    # Compute the displacement vector needed to move my_polydata on top of gt_polydata
    displacement = [gt_center[i] - my_center[i] for i in range(3)]

    # Apply the displacement to my_polydata
    transform = vtk.vtkTransform()
    transform.Translate(displacement)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(my_polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    my_polydata = transform_filter.GetOutput()

    # Write the repositioned my_polydata to a VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("repositioned_model.vtp")
    writer.SetInputData(my_polydata)
    writer.Write()

    # Return the repositioned my_polydata
    return my_polydata



### !!! src_points = numpy_support.vtk_to_numpy(finalCenterLineOutput.GetPoints().GetData())

# goal: def cut_outlet(surface, parameters)
# parameters:[list of end points, list of radii, list of unit tangent vectors]
# if using gt surface to clip pred surf: 1. find parameters from gt surface 2. clip pred surface with parameters

def bryan_get_clipping_parameters(clpd):
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

def bryan_get_radii_at_caps(clpd):
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

    lst_of_end_pts = []
    # append the very first point
    lst_of_end_pts.append(line_dict["line0"][0])

    for i in range(n_keys):
        key = f"line{i}"  
        lst_of_end_pts.append(line_dict[key][-1])

    nplst = np.array(lst_of_end_pts)
    radii_at_caps = []
    
    for i in lst_of_end_pts:
            for j in range(len(points)):
                if np.array_equal(i,points[j]):
                    radii_at_caps.append(radii[j])
    return radii_at_caps
    

def bryan_find_end_pt(clpd):
    """
    This function finds the end points of the branches in a blood vessel's centerline.
    """
    points = numpy_support.vtk_to_numpy(clpd.GetPoints().GetData())
    CenterlineID_for_each_point = numpy_support.vtk_to_numpy(clpd.GetPointData().GetArray('CenterlineId'))
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

    lst_of_end_pts = []
    # append the very first point
    lst_of_end_pts.append(line_dict["line0"][0])

    for i in range(n_keys):
        key = f"line{i}"  
        lst_of_end_pts.append(line_dict[key][-1])

    nplst = np.array(lst_of_end_pts)
    return nplst

def bryan_split_cl(clpd):
    """
    This function seperates blood vessel's centerline into multiple based on CenterlineId.
    """
    points = numpy_support.vtk_to_numpy(clpd.GetPoints().GetData())
    CenterlineID_for_each_point = numpy_support.vtk_to_numpy(clpd.GetPointData().GetArray('CenterlineId'))
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
    
    return line_dict

def bryan_find_unit_tangent_vectors(line_dictionary):
    """
    This function finds the unit tangent vector of the last point of each line in the centerline, including the first point of the first line.
    
    """
    unit_tangent_vectors = []
    #compute the unit tangent vector of the first point of the first line
    key = "line0"
    line = line_dictionary[key]
    tangent_vector = line[0] - line[1]
    unit_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
    unit_tangent_vectors.append(unit_tangent_vector)
    
    # compute the unit tangent vector of the last point of each line
    for i in range(len(line_dictionary)):
        key = f"line{i}"
        line = line_dictionary[key]
        tangent_vector = line[-1] - line[-2]
        unit_tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
        unit_tangent_vectors.append(unit_tangent_vector)
    return unit_tangent_vectors
   
# ONLY use to visualize the endpoints using uniformed box
def generate_boxes_surfaces(center_points, radius, output_file):
    """
    This function generates boxes centered at the input coordinates with a specified radius
    and writes the resulting .vtp file.

    :param center_points: NumPy array or list of center point coordinates
    :param radius: Radius of the boxes
    :param output_file: Path to the output .vtp file
    """
    box_surfaces = vtk.vtkAppendPolyData()

    # Convert the input center_points to a list, in case it is a NumPy array
    center_points = np.array(center_points).tolist()

    for center_point in center_points:
        # Create a vtkCubeSource for the box
        box = vtk.vtkCubeSource()
        box.SetCenter(center_point)
        box.SetXLength(2 * radius)
        box.SetYLength(2 * radius)
        box.SetZLength(2 * radius)
        box.Update()

        box_surfaces.AddInputData(box.GetOutput())

    box_surfaces.Update()

    # Write the combined box surfaces to a .vtp file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(box_surfaces.GetOutput())
    writer.Write()


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





def check_enclosed(polyData1,polyData2):

     # Mark points inside with 1 and outside with a 0
    select = vtk.vtkSelectEnclosedPoints()
    select.SetInputData(polyData1)
    select.SetSurfaceData(polyData2)

    # Extract three meshes, one completely inside, one completely
    # outside and on the border between the inside and outside.

    threshold = vtk.vtkMultiThreshold()
    # Outside points have a 0 value in ALL points of a cell
    outsideId = threshold.AddBandpassIntervalSet(
        0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "SelectedPoints",
        0, 1)
    # Inside points have a 1 value in ALL points of a cell
    insideId = threshold.AddBandpassIntervalSet(
        1, 1,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "SelectedPoints",
        0, 1)
    # Border points have a 0 or a 1 in at least one point of a cell
    borderId = threshold.AddIntervalSet(
        0, 1,
        vtk.vtkMultiThreshold.OPEN, vtk.vtkMultiThreshold.OPEN,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "SelectedPoints",
        0, 0)

    threshold.SetInputConnection(select.GetOutputPort())

    # Select the intervals to be output
    threshold.OutputSet(outsideId)
    threshold.OutputSet(insideId)
    threshold.OutputSet(borderId)
    threshold.Update()


    outside_polydata = threshold.GetOutput().GetBlock(outsideId).GetBlock(0)
    inside_polydata = threshold.GetOutput().GetBlock(insideId).GetBlock(0)
    border_polydata = threshold.GetOutput().GetBlock(borderId).GetBlock(0)

    # Convert the point data to numpy arrays
    outside_points_np = vtk.util.numpy_support.vtk_to_numpy(outside_polydata)
    inside_points_np = vtk.util.numpy_support.vtk_to_numpy(inside_polydata)
    border_points_np = vtk.util.numpy_support.vtk_to_numpy(border_polydata)

    # Return the numpy arrays
    return outside_points_np, inside_points_np, border_points_np


def insidepoints(points, surface, tolerance=1e-4):
    """Mark points as to whether they are inside a closed surface"""
    marker = vtk.vtkSelectEnclosedPoints()
    marker.SetInputData(points)
    marker.SetSurfaceData(surface)
    marker.SetTolerance(tolerance)
    marker.Update()
    return marker.GetOutput()

def keep_outside_points(polyData1, polyData2):
    # Mark points inside with 1 and outside with a 0
    select = vtk.vtkSelectEnclosedPoints()
    select.SetInputData(polyData1)
    select.SetSurfaceData(polyData2)
    select.Update()

    # Get the output of the vtkSelectEnclosedPoints filter
    selected_points_polydata = select.GetOutput()

    # Set the "SelectedPoints" array as the active scalar
    selected_points_polydata.GetPointData().SetActiveScalars("SelectedPoints")

    # Use a vtkThresholdPoints filter to keep only the points outside (i.e., with a scalar value of 0)
    threshold = vtk.vtkThresholdPoints()
    threshold.SetInputData(selected_points_polydata)
    threshold.ThresholdByUpper(0)  # keep only the points with a scalar value <= 0
    threshold.Update()

    # Get the output polyData with only the outside points
    outside_polydata = threshold.GetOutput()

    return outside_polydata




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


def bryan_get_ROM_surface(centerline_file,pred_surface_file,gt_cl_file=None):
    if gt_cl_file is None:
        # adjust cap according to its own centerline
        clpd = read_geo(centerline_file)
        predpd = read_geo(pred_surface_file)
        endpts, radii, unit_vecs = bryan_get_clipping_parameters(clpd)
        boxpd, boxpdlst = bryan_generate_oriented_boxes(endpts,unit_vecs,radii,'146_clippingbox.vtp',4)
        clippedpd = bryan_clip_surface(predpd,boxpd)
        filled = bryan_fillholes(clippedpd)
        largest = keep_largest_surface(filled)
        return largest
    else: # clip predicted surface according to gt centerline
        clpd = read_geo(gt_cl_file)
        predpd = read_geo(pred_surface_file)
        endpts, radii, unit_vecs = bryan_get_clipping_parameters(clpd)
        boxpd, boxpdlst = bryan_generate_oriented_boxes(endpts,unit_vecs,radii,'146_clippingbox.vtp',4)
        clippedpd = bryan_clip_surface(predpd,boxpd)
        filled = bryan_fillholes(clippedpd)
        largest = keep_largest_surface(filled)
        return largest

            

# def fill_holes_and_assign_ids(polydata, hole_size=1000.0):
#     # Fill holes
#     fillHolesFilter = vtk.vtkFillHolesFilter()
#     fillHolesFilter.SetInputData(polydata)
#     fillHolesFilter.SetHoleSize(hole_size)
#     fillHolesFilter.Update()

#     # Get the filled polydata
#     filledPolyData = fillHolesFilter.GetOutput()

#     # Get the cell array before and after filling holes
#     originalCells = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
#     filledCells = numpy_support.vtk_to_numpy(filledPolyData.GetPolys().GetData())

#     # Determine which cells are new
#     newCells = np.setdiff1d(filledCells, originalCells)

#     # Create a new array to hold the faceId values
#     faceIdArray = vtk.vtkIntArray()
#     faceIdArray.SetName("faceId")
#     faceIdArray.SetNumberOfValues(filledPolyData.GetNumberOfCells())

#     # Initialize all values to -1
#     for i in range(faceIdArray.GetNumberOfValues()):
#         faceIdArray.SetValue(i, -1)

#     # Assign unique "faceId" to each new cell
#     currentFaceId = 0
#     for cell in newCells:
#         faceIdArray.SetValue(cell, currentFaceId)
#         currentFaceId += 1

#     filledPolyData.GetCellData().AddArray(faceIdArray)

#     # Return the filled polydata with "faceId" assigned to new cells
#     return filledPolyData
def fill_holes_and_assign_ids(polydata, hole_size=1000.0):
    # Fill holes
    fillHolesFilter = vtk.vtkFillHolesFilter()
    fillHolesFilter.SetInputData(polydata)
    fillHolesFilter.SetHoleSize(hole_size)
    fillHolesFilter.Update()
    
    # Get the filled polygons
    filledPolygons = fillHolesFilter.GetOutput()
    
    # Assign unique "faceId" to each new polygon in filledPolygons
    polygons = filledPolygons.GetPolys()
    numberOfPolygons = polygons.GetNumberOfCells()
    
    faceIdArray = vtk.vtkIntArray()
    faceIdArray.SetName("faceId")
    faceIdArray.SetNumberOfValues(numberOfPolygons)
    
    for i in range(numberOfPolygons):
        faceIdArray.SetValue(i, i)
    
    filledPolygons.GetCellData().AddArray(faceIdArray)
    
    # Calculate intersection between the filled polygons and the original surface
    intersectionFilter = vtk.vtkBooleanOperationPolyDataFilter()
    intersectionFilter.SetOperationToIntersection()
    intersectionFilter.SetInputData(0, filledPolygons)
    intersectionFilter.SetInputData(1, polydata)
    intersectionFilter.Update()
    
    # Return the intersection, which should only contain the filled holes
    return intersectionFilter.GetOutput()

def get_polys_and_points(surface_polydata):
    
    # Get the points in the polydata
    points = surface_polydata.GetPoints()

    # Get the polygons in the polydata
    polygons = surface_polydata.GetPolys()

    # Initialize the polygon traversal
    polygons.InitTraversal()

    # Prepare to store the point indices of each polygon
    idList = vtk.vtkIdList()

    # Prepare the dictionary
    poly_dict = {}

    # Polygon index
    poly_index = 0

    # Iterate over all polygons
    while polygons.GetNextCell(idList):

        # Prepare the list of points for this polygon
        point_list = []

        # Iterate over the points in the polygon
        for i in range(idList.GetNumberOfIds()):
            point_id = idList.GetId(i)
            point = points.GetPoint(point_id)
            point_list.append(point)

        # Add this polygon to the dictionary
        poly_dict[poly_index] = point_list

        # Move on to the next polygon
        poly_index += 1

    return poly_dict

def sort_getpolys(surface_pd):
    g = numpy_support.vtk_to_numpy(surface_pd.GetPolys().GetData())
    g = np.split(g, np.where(g == 3)[0])
    g = [list(x[1:]) for x in g if len(x) > 1]
    return g

def vtp_to_vtk(vtp_file, vtk_file):
    # Read the .vtp file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()

    # Write the .vtk file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(vtk_file)
    writer.SetInputData(reader.GetOutput())
    writer.Write()  


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

import pdb

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
# >>> face_name_lst
# ['celiac_hepatic', 'celiac_splenic', 'ext_iliac_left', 'ext_iliac_right', 'IMA', 'int_iliac_left', 'int_iliac_right', 'renal_left', 'renal_right', 'SMA']
# lines = insert_outlet_names_to_rcrt('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\sv projects\\0146_1001\\Simulations\\0146_1001\\rcrt.dat',face_name_lst)
# write_to_file('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\ROM results\\inflow_files\\rcrt_from_parsing.dat',lines)

import os
import vtk

def write_face_pd(input_vtp_path, output_dir):
    # Read the input file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_vtp_path)
    reader.Update()

    # Get the 'ModelFaceID' attribute
    polydata = reader.GetOutput()
    face_id_array = polydata.GetPointData().GetArray('ModelFaceID')

    if face_id_array is None:
        print("Error: 'ModelFaceID' attribute not found.")
        return

    # Determine unique face IDs
    unique_face_ids = set()
    for i in range(face_id_array.GetNumberOfTuples()):
        unique_face_ids.add(face_id_array.GetTuple1(i))

    # For each unique face ID, extract the corresponding points and write them to a new .vtp file
    for face_id in unique_face_ids:
        # Create a threshold filter to extract the points
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(polydata)
        thresh.ThresholdBetween(face_id, face_id)
        thresh.SetInputArrayToProcess(0, 0, 0, 0, 'ModelFaceID')
        thresh.Update()

        # Convert the output of the threshold filter to polydata
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputData(thresh.GetOutput())
        geometry_filter.Update()

        # Write the output polydata to a .vtp file
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(os.path.join(output_dir, f"face_{int(face_id)}.vtp"))
        writer.SetInputData(geometry_filter.GetOutput())
        writer.Write()

    print("Finished writing face files.")

# write_face_pd('C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\0146_fromVMR.vtp','C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\ROM results\\faces_for_parsing')
import glob

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





def match_files(dir_a, dir_b, tolerance):
    """
    Match files from two directories based on their centroids within a certain tolerance.

    :param dir_a: path to the first directory
    :param dir_b: path to the second directory
    :param tolerance: tolerance for matching centroids
    :return: list of matched polydata pairs
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

    centroids_a = get_centroids(dir_a)
    centroids_b = get_centroids(dir_b)
    
    matched_files = []
    for path_a, centroid_a in centroids_a.items():
        for path_b, centroid_b in centroids_b.items():
            distance = ((centroid_a[0] - centroid_b[0])**2 +
                        (centroid_a[1] - centroid_b[1])**2 +
                        (centroid_a[2] - centroid_b[2])**2)**0.5
            if distance <= tolerance:
                matched_files.append([read_geo(path_a), read_geo(path_b)])
                break  # Assuming one-to-one matching
    
    return matched_files



def delete_and_rename_matched_files(pred_caps, gt_caps, input_vtp, tolerance=0.3):
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
