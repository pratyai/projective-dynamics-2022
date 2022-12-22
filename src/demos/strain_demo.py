# Library imports
import argparse
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import openmesh as om

# Our imports
import system as msys
from . import demo_systems
import sys
import os
import subprocess

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
        if (state['record']):
            ps.screenshot(transparent_bg=False)


def ui_system_parameters(state: dict):
    """
    System parameters section of the UI.
    """
    if (psim.TreeNodeEx("System Parameters",
                        flags=psim.ImGuiTreeNodeFlags_DefaultOpen)):

        # Initialize / reset simulation button
        if (psim.Button("Initialize / Reset")):
            state['ui_is_running'] = False
            _initialize_system(state)
            _update_polyscope_mesh(state)

        psim.PushItemWidth(MY_ITEM_WIDTH)
        # Point mass float input
        changed, state['ui_mass_matrix_scalar'] = psim.InputFloat(
            "Point mass", state['ui_mass_matrix_scalar'])

        # Spring stiffness float input
        changed, state['ui_singular_values_epsilon'] = psim.SliderFloat(
            "Singular values epsilon", state['ui_singular_values_epsilon'], v_min=0, v_max=1)
        psim.PopItemWidth()

        # Pin selected nodes
        if (psim.Button("Pin Selected Vertex")):
            _pin_selected_nodes(state)
        pinned_indices = sorted(state['ui_pinned_indices'])
        if pinned_indices:
            psim.SameLine()
            if (psim.Button("Clear Selection")):
                state['ui_pinned_indices'].clear()
            psim.TextUnformatted("{}".format(pinned_indices))

        psim.TreePop()


def ui_simulation_control(state: dict):
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
    epsilon = state['ui_singular_values_epsilon']
    singular_values_range = (1 - epsilon, 1 + epsilon)
    state['system'] = demo_systems.make_triangle_mesh_strain_system(
        state['mesh'],
        state['ui_mass_matrix_scalar'],
        singular_values_range,
        state['ui_pinned_indices'])


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
    if not ps.have_selection():
        return

    # Otherwise get the selection
    ps_mesh, element_index = ps.get_selection()

    # And pin the corresponding vertex
    state['ui_pinned_indices'].append(element_index)


def edge(c):
    '''
    Compute the edge for the constraint `c` (index form).
    '''
    q = c.q()
    return [q[0][1], q[1][1]]


def _initialize_polyscope_mesh(state: dict):
    """
    Initialize the polyscope mesh to be a curve network where each edge corresponds to a spring in our system.
    """
    system = state['system']
    mesh = state['mesh']
    state['ps_mesh'] = ps.register_surface_mesh(
        name='ps_mesh',
        vertices=mesh.points(),
        faces=mesh.face_vertex_indices())
    state['ps_mesh'].set_edge_width(1)


def _update_polyscope_mesh(state: dict):
    """
    Update the polyscope curve network according to the current vertex positions of the system.
    """
    state['ps_mesh'].update_vertex_positions(state['system'].q)


def main(args: argparse.Namespace):
    """
    Creates a Polyscope application visualizing a mass-spring system simulated using projective dynamics.

    Parameters:
        filename (str): filename of the triangle mesh used to initialize the mass-spring system
    """

    # Read the mesh
    mesh = om.read_trimesh(args.filename)

    # Set the application name
    ps.set_program_name('Projective Dynamics Demo')

    # Initialize polyscope; should be called exactly once
    ps.init()

    # Set up our scene
    ps.set_up_dir('z_up')
    ps.set_ground_plane_height_factor(1)

    '''
    `state` is a dictionary representing the application state.
    '''
    state = {
        'mesh': mesh,
        'system': None,
        'ui_mass_matrix_scalar': 1,
        'ui_singular_values_epsilon': 0,
        'ui_pinned_indices': [],
        'ps_mesh': None,
        'ui_is_running': False,
        'ui_h': 0.01,
        'ui_steps_per_frame': 10,
        'record': args.record
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

    # Optional argument to record the application screen
    parser.add_argument("--record",
                        help="whether or not to record the application screen",
                        action="store_true")

    # Parse the command line arguments
    args = parser.parse_args()

    # Pass the args to our main function
    main(args)

    # Optionally record the application screen
    if (args.record):
        subprocess.run(
            f"mkdir -p ../demos/ &&ffmpeg -y -framerate 30 -pattern_type glob -i 'screenshot_*.png' -c:v libx264 -pix_fmt yuv420p ../demos/{os.path.basename(args.filename)}.mp4 && rm screenshot_*.png",
            shell=True)
