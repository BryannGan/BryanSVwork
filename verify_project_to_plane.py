####################################################
#   functions to verify project_points_to_plane    #
####################################################
import vtk
import numpy as np

def project_points_to_plane(points, normal_vector, d=0):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    if np.allclose(normal_vector, [1, 0, 0]):
        u = np.array([0, 1, 0])
    else:
        u = np.array([1, 0, 0])
    
    v = np.cross(normal_vector, u)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    
    projected_points_2d = []
    
    for point in points:
        projection = point - np.dot(point, normal_vector + d) * normal_vector
        projected_points_2d.append(projection)
    
    return np.array(projected_points_2d)

def create_sphere_actor(center, radius, color):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center)
    sphere.SetRadius(radius)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    return actor

def visualize_points(original_points, projected_points):
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    # Add original points in red
    for point in original_points:
        actor = create_sphere_actor(point, 0.1, (1, 0, 0))  # Red
        renderer.AddActor(actor)
    # Add projected points in blue
    for point in projected_points:
        actor = create_sphere_actor(point, 0.1, (0, 0, 1))  # Blue
        renderer.AddActor(actor)
    # Add axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)  # Set length of axes
    axes.SetShaftTypeToCylinder()
    #axes.SetAxisTitleTextScaleFactor(0.5)
    renderer.AddActor(axes)
    # Set up the camera
    renderer.ResetCamera()
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background
    # Start rendering
    render_window.Render()
    interactor.Start()

def generate_circle_points(center, radius, num_points=50, normal_vector=[0, 0, 1]):
    t = np.linspace(0, 2 * np.pi, num_points)
    circle_points = []
    # Create a set of points in the XY plane centered at the origin
    for angle in t:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        circle_points.append([x, y, z])
    # Rotate the points to align with the desired plane normal
    circle_points = np.array(circle_points)
    # Translate points to the center
    circle_points += center
    return circle_points

# Example usage
center = np.array([0, 0, 0])
radius = 5
num_points = 50
normal_vector = np.array([0, 1, 1])  # Normal vector of the projection plane

# Generate points that form a circle
circle_points = generate_circle_points(center, radius, num_points)

# Project the circle points onto the plane
projected_points_2d = project_points_to_plane(circle_points, normal_vector)

# Visualize the points and their projections
visualize_points(circle_points, projected_points_2d)

####################################################
# END functions to verify project_points_to_plane  #
####################################################
import vtk
import numpy as np

def visualize_2d_points(points_2d):
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Add points in blue
    for point in points_2d:
        actor = create_sphere_actor_2d(point, 0.1, (0, 0, 1))  # Blue
        renderer.AddActor(actor)
    
    # Set up the camera for 2D rendering
    camera = renderer.GetActiveCamera()
    camera.SetParallelProjection(True)
    renderer.ResetCamera()

    # Adjust camera to fit all points
    renderer.ResetCamera()
    renderer.SetBackground(1, 1, 1)  # White background

    # Start rendering
    render_window.Render()
    interactor.Start()

def create_sphere_actor_2d(center, radius, color):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center[0], center[1], 0)
    sphere.SetRadius(radius)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    return actor

# Example usage
center = np.array([0, 0, 0])
radius = 5
num_points = 50
normal_vector = np.array([0, 1, 1])  # Normal vector of the projection plane

# Generate points that form a circle
circle_points = generate_circle_points(center, radius, num_points)

# Project the circle points onto the plane
projected_points_2d = project_points_to_plane(circle_points, normal_vector)

# Visualize the 2D points
visualize_2d_points(projected_points_2d)

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

import pdb
pdb.set_trace() 



def visualize_2d_coordinates(points_2d):
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Add 2D points in blue
    for point in points_2d:
        actor = create_2d_sphere_actor(point, 0.1, (0, 0, 1))  # Blue
        renderer.AddActor(actor)
    
    # Set up the camera for 2D rendering
    camera = renderer.GetActiveCamera()
    camera.SetParallelProjection(True)
    renderer.ResetCamera()

    # Adjust camera to fit all points
    renderer.ResetCamera()
    renderer.SetBackground(1, 1, 1)  # White background

    # Start rendering
    render_window.Render()
    interactor.Start()

def create_2d_sphere_actor(center, radius, color):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(center[0], center[1], 0)
    sphere.SetRadius(radius)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    return actor

# Example usage of the 2D visualization
center = np.array([0, 0, 0])
radius = 5
num_points = 50
normal_vector = np.array([0, 1, 1])  # Normal vector of the projection plane

# Generate points that form a circle
circle_points = generate_circle_points(center, radius, num_points)

# Project the circle points onto the plane and convert to 2D coordinates
projected_points_2d = project_points_to_2d_plane(circle_points, normal_vector)

# Visualize the 2D coordinates
visualize_2d_coordinates(projected_points_2d)