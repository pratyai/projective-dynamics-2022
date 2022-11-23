# Library imports
import argparse
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import openmesh as om

# Our imports
import system as msys
from . import demo_systems

# Custom item width
MY_ITEM_WIDTH = 100

def ui_callback(state: dict):
    """
    The callback function for the Polyscope application. Manages the user interface, the polyscope application, and the system.
    """
    
    ui_system_parameters(state)
    ui_simulation_control(state)
    
    # Update the system and polyscope mesh when the application is running
    if state['ui_is_running']:
        _update_system(state)
        _update_polyscope_mesh(state)

def ui_system_parameters(state:dict):
    """
    System parameters section of the UI.
    """
    if(psim.TreeNodeEx("System Parameters",
                       flags=psim.ImGuiTreeNodeFlags_DefaultOpen)):
        
        # Reset simulation button
        if (psim.Button("Reset")):
            state['ui_is_running'] = False
            _initialize_system(state)
            _update_polyscope_mesh(state)
        
        psim.PushItemWidth(MY_ITEM_WIDTH)
        # Point mass float input
        changed, state['ui_point_mass'] = psim.InputFloat(
            "Point mass", state['ui_point_mass'])
        
        # Spring stiffness float input
        changed, state['ui_spring_stiffness'] = psim.InputFloat(
            "Spring stiffness", state['ui_spring_stiffness']) 
        psim.PopItemWidth()
        
        # Pin selected nodes
        if (psim.Button("Pin Selection")): _pin_selected_nodes(state)
        
        psim.TreePop()

def ui_simulation_control(state:dict):
    """
    Simulation control section of the UI.
    """
    if (psim.TreeNodeEx("Simulation Control",
                       flags=psim.ImGuiTreeNodeFlags_DefaultOpen)):
        
        # Start / Stop simulation button
        start_stop_button_text = "Start" if not state['ui_is_running'] else "Stop"
        if (psim.Button(start_stop_button_text)):
            state['ui_is_running'] = not state['ui_is_running']
        
        psim.PushItemWidth(MY_ITEM_WIDTH)
        # Step size slider
        changed, state['ui_h'] = psim.InputFloat(
            "Step size", state['ui_h'])
        
        # Steps per frame
        changed, state['ui_steps_per_frame'] = psim.InputInt(
            "Steps Per Frame", state['ui_steps_per_frame'], step=1, step_fast=10)
        psim.PopItemWidth()

        psim.TreePop()

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

def _pin_selected_nodes(state: dict):
    """
    If a node is selected, pin it in our system.
    """
    # Do nothing if nothing is selected
    if not ps.have_selection(): return
    
    # Otherwise get the selection
    ps_mesh, element_index = ps.get_selection()
    
    # And pin the corresponding vertex
    state['system'].pinned.add(element_index)

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
