import vtk
import numpy as np
import os
import sys
from vtk.util import numpy_support
from sv import *
from sv_rom_simulation import *
import pdb
import re 
import re

"""
Becuase 0188_0001_aorta, KDR12_aorta,KDR33_aorta, O150323_2009_aorta, O344211000_2006_aorta are all aortas

we use the same inflow file from 0176_0000 (also an aorta) for all of these 

inflow file (and folder) for each is MANAUALLY copied over

boundary conditions are uniform across all caps here

write the following for each cap

2
2
800
0.0002
7000



"""

def extract_face_ids_names(file_path):
    face_pairs = []
    # Define a regular expression to match face elements and capture id and name attributes
    face_pattern = re.compile(r'<face id="(\d+)" name="([^"]+)"')
    
    with open(file_path, 'r') as file:
        for line in file:
            match = face_pattern.search(line)
            if match:
                face_id = int(match.group(1))
                face_name = match.group(2)
                face_pairs.append([face_id, face_name])
    
    return face_pairs

svproject_path = 'C:\\Users\\bygan\\Documents\\Research_at_Cal\\Shadden_lab_w_Numi\\MIROS_paper\\MIROS_svproj'


for filename in os.listdir(svproject_path):
    
    filename = filename + '.vtp'

    gt_cl_path = os.path.join(svproject_path,filename)
    gt_model_path = os.path.join(svproject_path, os.path.splitext(filename)[0], 'Models', filename)
    gt_inflow_pd_path = os.path.join(svproject_path, os.path.splitext(filename)[0],'Simulations',os.path.splitext(filename)[0],'mesh-complete','mesh-surfaces','inflow.vtp')
    gt_mesh_surf_path = os.path.join(svproject_path, os.path.splitext(filename)[0],'Simulations',os.path.splitext(filename)[0],'mesh-complete','mesh-surfaces')
    mdl_path = os.path.join(svproject_path, os.path.splitext(filename)[0], 'Models', filename[:-4]+'.mdl')
    bc_file_path = os.path.join(svproject_path, os.path.splitext(filename)[0],'Simulations',os.path.splitext(filename)[0],'rcrt.dat')
    gt_inflow_file_path = os.path.join(svproject_path, os.path.splitext(filename)[0],'flow-files')
    #martin_1d_input = os.path.join(martin_1d_input_path, filename[:-4]+'_1d.in')
    mdl_path = os.path.join(svproject_path, os.path.splitext(filename)[0], 'Models', filename[:-4]+'.mdl')



    md = modeling.PolyData(read_polydata(gt_model_path))
    id_name_map = extract_face_ids_names(mdl_path)

    face_ids = md.get_face_ids() 
    for face_id in face_ids:
        face_polydata = md.get_face_polydata(face_id)
        
        write_polydata(os.path.join(gt_mesh_surf_path, id_name_map[face_id-1][1]+'.vtp'),face_polydata)
    
    print('done, move to next model')



pdb.set_trace()
    

