import numpy as np
import numpy.typing as npt
import numpy.linalg as la


def unit(v: npt.NDArray):
    '''
    Return the unit vector of the non-zero vector v.
    v := a vector (typically of size 3).
    '''
    return v / la.norm(v)


def tri_are(p: npt.NDArray):
    '''
    Return the area of the triangle specified by `p`.
    p := `3 x D(=3)` matrix, each row being one vertex of the triangle.
    '''
    e01, e02 = p[1, :] - p[0, :], p[2, :] - p[0, :]
    return la.norm(np.cross(e01, e02))/2


def mass_matrix_fem_trimesh(q: npt.NDArray, indices: npt.NDArray):
    '''
    Construct the mass matrix of a finite triangular mesh of uniform unit density (i.e. scale it to appropriate density).
    q := `n x D(=3)` matrix, each row being the location of a vertex.
    indices := `m x 3` matrix, each row being the indices of one triangle.
    '''
    n = q.shape[0]
    M = np.zeros((n, n))
    m = indices.shape[0]
    for i in range(m):
        tri = indices[i, :]
        p = q[tri, :]
        A = tri_are(p)
        # mass matrix of individual element.
        m = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) * A/12
        # TODO: how to write it efficiently?
        for (i, r) in enumerate(tri):
            for (j, c) in enumerate(tri):
                M[r, c] += m[i, j]
    return M
