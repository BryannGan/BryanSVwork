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
import vtk
import os
import numpy as np


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
        Params.outflow_bc_file = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\ROM results\\inflow_files\\'

        #watch out!
        Params.model_order= 0
        return Params

if __name__=='__main__':
    
    "path to ur result folder"
    result_master_folder = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\agnostic_result'

    "path to ur svproject(gt model) folder"
    svproject_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\sv projects'

    'path to ur Numi model folder' 'all file starts with Numi_ then the model name'
    Numi_model_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\2023_summer_research\\numi_model_folder'

    'path to ur gt centerline folder'
    gt_cl_path =   'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\centerlines'

    for filename in os.listdir(Numi_model_path):  #iterate through all the models in the folder
        # filename = Numi_model_name.vtp
        # print('debug: filename ' + filename)
        pred_surf_path = os.path.join(Numi_model_path, filename)
        # print('debug: pred_surf_path' + pred_surf_path)
        specific_folder = mkdir(result_master_folder,filename[:-4])
        # print('debug: specific_folder' + specific_folder)
        filename = filename[5:] #remove Numi_ from the filename
        # specific_folder = specific_folder[:-4]
       
        
        # print('debug: filename' + filename)

        # create log file for each model
        
        edit_log(specific_folder, "Start processing model: "+filename)

        edit_log(specific_folder, "boxcutting model: "+filename)
        
        
        gt_cl_path = os.path.join(gt_cl_path,filename)
        gt_model_path = os.path.join(svproject_path, os.path.splitext(filename)[0], 'Models', filename)
        path_to_write= os.path.join(specific_folder, 'clipped_')+filename

        # print('debug: pred_surf_path ' + pred_surf_path)
        # print('debug: specific_folder ' + specific_folder)
        # print('debug: filename ' + filename)
        # print('debug: gt_cl_path ' + gt_cl_path)
        # print('debug: gt_model_path ' + gt_model_path)
        # print('debug: path_to_write ' + path_to_write)
        

        # boxcut the model
        box_scale = 5
        boxcut_model,clippingbox = bryan_boxcut(gt_cl_path,pred_surf_path,filename,box_scale, gt_cl_path)
        
        if boxcut_model is None:
            edit_log(specific_folder, "boxcutting failed")
            #need better ways to check if boxcutting failed (like detect #of caps)
            continue
        else: 
            edit_log(specific_folder, "boxcutting finished, waiting to test whether the boxcut model is valid")
        
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
        
        ################# subject to change!! hard coded #####################
        filled_md.combine_faces(1,[5])
        ######################################################################

        # save the filled model
        write_polydata(os.path.join(specific_folder, 'filled_'+ filename), filled_md.get_polydata())
        edit_log(specific_folder, "filled model saved")
        full_pd = filled_md.get_polydata() #get polydata from new object
        edit_log(specific_folder, "remeshing to get finer model for mesh generation")
        ROM_ready_pd = svmeshtool.remesh_polydata(full_pd,0.1,0.1)
        write_polydata(os.path.join(specific_folder, 'export_ROM_ready_surface.vtp'), ROM_ready_pd)
        edit_log(specific_folder, "saved remeshed model as export_ROM_ready_surface.vtp")
        
        # if ROM_ready_pd is None:
        #     edit_log(specific_folder, "remeshing failed")
        #     continue
        # else:
        #     edit_log(specific_folder, "remeshing finished,start exportiing caps")

        ready_md = sv.modeling.PolyData(ROM_ready_pd) # create new object (finer polydata with face id attribute)
        # export caps 
        boundary_face_path= mkdir(specific_folder,'boundary_faces')
        ################# subject to change!! hard coded #####################
        total_face_id = ready_md.get_face_ids() # get face id from new object [1,2,3,4,5,6,7]
        identify = ready_md.identify_caps() #[2,3,4,5] index instead of real id
        keep_true_cap = [i+1 for i in range(len(identify)) if identify[i] == True] # get the face id of the cap 
        print(keep_true_cap)    

        for i , id in enumerate(total_face_id):  
            if i == 0: #wall
                write_polydata(os.path.join(boundary_face_path,'wall.vtp'),ready_md.get_face_polydata(id)) # get polydata for each face id
            elif i == len(total_face_id)-1: #hard-coded inlet (last id)
                write_polydata(os.path.join(boundary_face_path,'inlet.vtp'),ready_md.get_face_polydata(id))
            else:
                write_polydata(os.path.join(boundary_face_path,'outlet'+str(i)+'.vtp'),ready_md.get_face_polydata(id))
        
        for i , id in enumerate(total_face_id):  
            if i == 0: #wall
                write_polydata(os.path.join(boundary_face_path,str(i)+'_'+str(id)+'wall.vtp'),ready_md.get_face_polydata(id)) # get polydata for each face id
            elif i == len(total_face_id)-1: #hard-coded inlet (last id)
                write_polydata(os.path.join(boundary_face_path,str(i)+'_'+str(id)+'inlet.vtp'),ready_md.get_face_polydata(id))
            else:
                write_polydata(os.path.join(boundary_face_path,str(i)+'_'+str(id)+'outlet'+str(i)+'.vtp'),ready_md.get_face_polydata(id))
        
        #########################################################################

        Params = setup_ROM_parameters(specific_folder,filename,1)
        Cl = Centerlines()
        Cl.extract_center_lines(Params)

        # Prepare inflow.dat file for OneD simulation

        msh = mesh.Mesh()
        msh.outlet_face_names_file = os.path.join(specific_folder,'centerline_outlets.dat')
        msh.generate(Params,Cl) 


        pdb.set_trace()
