import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import system as msys
import demo_systems as demos
import sys


def ui_callback(state: dict, system):
    '''
    `state` is a dictionary of ui states to be updated. It is a dictionary now for flexibility.
    '''
    button_text = "Start" if not state['ui_is_running'] else "Stop"
    if (psim.Button(button_text)):
        state['ui_is_running'] = not state['ui_is_running']
    changed, state['ui_h'] = psim.SliderFloat(
        "Step size", state['ui_h'], v_min=0.001, v_max=1.0)
    if 'ps_mesh' not in state:
        def comp_line_segs(): return np.array([[qi, c.p0()[1]]
                                               for (qi, (q, cs)) in enumerate(zip(system.q, system.cons))
                                               for c in cs
                                               if c.p0()[1] is not None])
        state['ps_mesh'] = ps.register_curve_network(
            name='ps_mesh', nodes=system.q, edges=comp_line_segs())
    if state['ui_is_running']:
        system.step(state['ui_h'])
        state['ps_mesh'].update_node_positions(system.q)


def main(s: msys.System):
    # Set the application name
    ps.set_program_name('Projective Dynamics Demo')

    # Initialize polyscope; should be called exactly once
    ps.init()

    # Set up our scene
    ps.set_up_dir('z_up')
    ps.set_ground_plane_mode('none')

    state = {
        'ui_is_running': False,
        'ui_h': 0.01,
    }
    ps.set_user_callback(lambda: ui_callback(state, s))

    # Pass control flow to polyscope, displaying the interactive window
    ps.show()


if __name__ == '__main__':
    # Simple command line argument handling
    if (len(sys.argv) != 2):
        print("Usage: python demo3.py mesh.obj")
        exit()
    
    # Create a system fom our triangle mesh
    filename = sys.argv[1]
    s = demos.make_triangle_mesh_system(filename)
    main(s)
