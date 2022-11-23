# Library imports
import argparse
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import openmesh as om

# Our imports
import system as msys
from . import demo_systems

def ui_callback(state: dict):
    """
    The callback function for the Polyscope application. Manages the user interface, the polyscope application, and the system.
    """
    
    # System parameters section
    if(psim.TreeNodeEx("System Parameters",
                       flags=psim.ImGuiTreeNodeFlags_DefaultOpen)):
    
        # Reset simulation button
        if (psim.Button("Reset")):
            state['ui_is_running'] = False
            _initialize_system(state)
            _update_polyscope_mesh(state)
            
        psim.TreePop()
    
    # System control section
    if (psim.TreeNodeEx("Simulation Control",
                       flags=psim.ImGuiTreeNodeFlags_DefaultOpen)):
        
        # Start / Stop simulation button
        start_stop_button_text = "Start" if not state['ui_is_running'] else "Stop"
        if (psim.Button(start_stop_button_text)):
            state['ui_is_running'] = not state['ui_is_running']
        
        # Step size slider
        changed, state['ui_h'] = psim.SliderFloat(
            "Step size", state['ui_h'], v_min=0.001, v_max=1.0)
        
        # Steps per frame
        changed, state['ui_steps_per_frame'] = psim.InputInt(
            "ui_int", state['ui_steps_per_frame'], step=1, step_fast=10)
    
    # Update the system and polyscope mesh when the application is running
    if state['ui_is_running']:
        _update_system(state)
        _update_polyscope_mesh(state)

def _initialize_system(state: dict):
    """
    Initialize the system with specified point mass and spring stiffness
    """
    state['system'] = demo_systems.make_triangle_mesh_system(
        state['mesh'], state['ui_point_mass'], state['ui_spring_stiffness'])

def _update_system(state: dict):
    """
    Update our system by taking the specified amount of steps with specified step size.
    """
    for i in range(state['ui_steps_per_frame']):
            state['system'].step(state['ui_h'])

def _initialize_polyscope_mesh(state: dict):
    """
    Initialize the polyscope mesh to be a curve network where each edge corresponds to a spring in our system.
    """
    system = state['system']
    edges = np.array([[qi, c.p0()[1]] 
                      for (qi, (q, cs)) in enumerate(zip(system.q, system.cons))
                      for c in cs
                      if c.p0()[1] is not None])
    state['ps_mesh'] = ps.register_curve_network(
            name='ps_mesh', nodes=system.q, edges=edges)

def _update_polyscope_mesh(state:dict):
    """
    Update the polyscope curve network according to the current vertex positions of the system.
    """
    state['ps_mesh'].update_node_positions(state['system'].q)

def main(filename: str):
    """
    Creates a Polyscope application visualizing a mass-spring system simulated using projective dynamics.
    
    Parameters:
        filename (str): filename of the triangle mesh used to initialize the mass-spring system
    """
    
    # Read the mesh
    mesh = om.read_trimesh(filename)
    
    # Set the application name
    ps.set_program_name('Projective Dynamics Demo')

    # Initialize polyscope; should be called exactly once
    ps.init()

    # Set up our scene
    ps.set_up_dir('z_up')
    ps.set_ground_plane_mode('none')

    '''
    `state` is a dictionary representing the application state.
    '''
    state = {
        'mesh' : mesh,
        'system' : None,
        'ui_point_mass' : 1,
        'ui_spring_stiffness' : 1,
        'ps_mesh' : None,
        'ui_is_running': False,
        'ui_h': 0.01,
        'ui_steps_per_frame': 10
    }
    # Initialize the system and polysope mesh
    _initialize_system(state)
    _initialize_polyscope_mesh(state)
    
    # Set the user callback
    ps.set_user_callback(lambda: ui_callback(state))

    # Pass control flow to polyscope, displaying the interactive window
    ps.show()


if __name__ == '__main__':
    """
    Parse command line arguments and run the Polyscope application.
    """
    
    # Parse our command line arguments
    parser = argparse.ArgumentParser()
    
    # Require a mesh filename as input
    parser.add_argument("filename",
                        help="the input mesh filename",
                        type=str)
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Pass the filename to our main function
    main(args.filename)
