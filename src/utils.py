import numpy as np

'''
Construct a grid mesh.
'''
def grid_mesh(u_min: float, u_max: float, u_num: int, v_min: float, v_max: float, v_num: int):
    param = lambda u, v : [u, v, 0]
    return _param_surface(u_min, u_max, u_num, v_min, v_max, v_num, param)

'''
Construct a mesh by discretizing a surface parameterized by f(u, v) = (x, y, z)
Return the Vx3 vertex matrix and Fx3 face matrix representing the discretized surface.
'''
def _param_surface(u_min: float, u_max: float, u_num: int, v_min: float, v_max: float, v_num: int, param: callable):
    assert u_num >= 1
    assert v_num >= 1

    # Construct our vertex and face matrices
    n_v = (u_num + 1) * (v_num + 1)
    n_t = 2 * u_num * v_num
    V = np.zeros((n_v, 3))
    F = np.zeros((n_t, 3))
    
    # Tesselate the parametric surface
    v_i = 0 # Vertex index
    f_i = 0 # Face index
    u_delta = (u_max - u_min) / u_num
    v_delta = (v_max - v_min) / v_num
    
    # Iterate over our discretization
    for i in range(u_num + 1):
        for j in range(v_num + 1):
            
            # Add the vertex
            u = u_min + i * u_delta
            v = v_min + j * v_delta
            V[v_i] = param(u, v)
            
            # Add the two corresponding triangles for each quad
            if (i < u_num and j < v_num):
                a = v_i
                b = v_i + 1
                c = a + v_num + 1
                d = c + 1
                F[f_i] = [a, b, c]
                f_i = f_i + 1
                F[f_i] = [c, b, d]
                
                # Update the face index
                f_i = f_i + 1
            
            # Update the vertex index
            v_i = v_i + 1
    
    return V, F