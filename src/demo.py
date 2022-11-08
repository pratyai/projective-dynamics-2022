import numpy as np
import numpy.linalg as la
import polyscope as ps
import polyscope.imgui as psim

# Our imports
import utils
from system import System

V = None # Vx3 vertex matrix
F = None # Fx3 face matrix
ps_mesh = None # Polyscope mesh handle

system = None # Our system

ui_is_running = False # Whether or not the simulation is running
ui_h = 0.1 # Step size h
   
def main():
    # Set the application name
    ps.set_program_name("Projective Dynamics Demo")
    
    # Initialize polyscope; should be called exactly once
    ps.init()
    
    # Set up our scene
    ps.set_up_dir("z_up")
    ps.set_user_callback(callback)
    
    # Create our grid mesh and register it to polyscope
    global V, F, ps_mesh
    V, F = utils.grid_mesh(-1, 1, 4, -1, 1, 4)
    ps_mesh = ps.register_surface_mesh("my mesh", V, F, smooth_shade=True)
    
    # UNCOMMENT this to create our system
    # system_initialize()
        
    # Pass control flow to polyscope, displaying the interactive window
    ps.show()

# User defined callback function
def callback():
    # Start / stop simulation button
    global ui_is_running
    button_text = "Start" if not ui_is_running else "Stop"
    if(psim.Button(button_text)):
        ui_is_running = not ui_is_running
        
    # Step size slider
    global ui_h
    changed, ui_h = psim.InputFloat("Step size", ui_h) 
    
    # Simulation update logic
    global V, F, mesh_update
    if ui_is_running:
        # Translate by ui_h
        V += np.array([0, 0, ui_h])
        
        # UNCOMMENT the following to run the system update code!
        # V = mesh_update()
        
        ps_mesh.update_vertex_positions(V) # Update the polyscope mesh vertex positions

# User defined mesh update function
def mesh_update():
    system_update(ui_h)
    return system.q.reshape(V.shape)

# Initialize our system
def system_initialize():
    global V, system

    N = V.shape[0] # N = V
    q = V # Our configuration vector q is V
    m = 1 # Constant mass
    M = np.ones(q.shape[0]) * m / (N*N) # Mass matrix

    # Construct our system
    system = System(q=q, q1=None, M=np.kron(np.diagflat(M), np.identity(System.D)))
    
    # Pin two corner vertices
    system.pinned.add(0)
    system.pinned.add(N-1)
    
    # Add spring constraints along spring lines
    for i in range(N):
        for j in range(N):
            L = la.norm(V[i] - V[j]) # Length of edge i, j
            if i+1 < N:
                system.add_spring(k=1, L=L/(N-1), q_idx=i*N+j, p0_idx=(i+1)*N+j)
            if i-1 >= 0:
                system.add_spring(k=1, L=L/(N-1), q_idx=i*N+j, p0_idx=(i-1)*N+j)
            if j+1 < N:
                system.add_spring(k=1, L=L/(N-1), q_idx=i*N+j, p0_idx=i*N+j+1)
            if j-1 >= 0:
                system.add_spring(k=1, L=L/(N-1), q_idx=i*N+j, p0_idx=i*N+j-1) 

# Update our system
def system_update(h: float):
    system.step(h)

if __name__ == "__main__":
    main()