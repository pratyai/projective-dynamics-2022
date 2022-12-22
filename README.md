# Cloth Simulation with Projective Dynamics

Pratyai Mazumder, Tal Rastopchin, and Konstantinos Stavratis.

An implementation of cloth simulation using projective dynamics. Written in Python. Created as the final project of the Physically-Based Simulation in Computer Graphics course at ETH Zürich the fall of 2022.

## How to Run

The two demo programs that we have put together are `src/demos/spring_demo.py` and `src/demos/strain_demo.py`. Each demo program reads a .obj triangle mesh and allows the user to interactively model cloth simulation. The `spring_demo` models the cloth with a mass-spring system and the `strain_demo` models the cloth as a triangle mesh with a continuous strain energy.

### Prerequisites

Your local Python environment requires the following libraries to run our demos:

* [Numba](https://numba.pydata.org/)
* [NumPy](https://numpy.org/)
* [OpenMesh](https://pypi.org/project/openmesh/)
* [Polyscope](https://polyscope.run/py/)
* [SciPy](https://scipy.org/)

### `spring_demo`
From within the `src` directory, we run the `spring_demo` with the input triangle mesh `data/square_100_unstructured.obj` as follows:

```
python -m demos.spring_demo ../data/square_100_unstructured.obj
```
(You might need to find the Polyscope application window after the program starts.) Hovering over the `controls` button of the `Polyscope` panel on the left-hand side of the application explains view and menu navigation. All of the panels on the left-hand side of the application allow the user to change how the 3D scene looks.

The `Command UI` panel on the right-hand side of the screen allows the user to control the cloth simulation.

#### System Parameters

The system parameters subpanel allows the user to control the simulation's underlying system parameters.

1. The `Initialize / reset` button both applies the specified system parameters as well as resets the simulation. This means that every time after the user changes the point mass, spring stiffness, or pinned vertices, **this button must be clicked to update the system and apply the parameters.**

2. The `Point mass` float input field allows the user to specify the mass of each point. (This value is used as a scalar coefficient during the construction of the underlying system's lumped mass matrix.)

3. The `Spring Stiffness` float input field allows the user to specify the spring stiffness constant `k` used for every spring in the system.

4. The `Pin Selected Vertex` button allows the user to click on vertices and mark them as pinned. To do this, the user must left-click on a vertex. **Note: it is important that the `Selection` panel that pops up under the `Command UI` panel specifies that the selection is a *`node`* and not an *`edge`***. After selecting a vertex, the user can then click `Pin Selected Vertex` and it will get added to the list of pinned vertices. The user can see the indices of the pinned vertices displayed underneath the `Pin Selected Vertex` button. To the right of the button there is a `Clear Selection` button which clears the list of pinned vertices.

#### Simulation Control

The simulation control subpanel allows the user to control the simulation.

1. The `Start` / `Stop` button start and stop the simulation, respectively.

2. The `Step size` float input field allows the user to specify the implicit Euler step size `h`. The default value of `0.01` seems to work well.

3. The `Steps per frame` integer input field allows the user to specify the number of projective dynamics implicit Euler steps to take each frame. The default value of `10` seems to work well. (Any smaller value and the application slows down due to the overhead of copying the vertex positions from our system's representation to Polyscope's curve network representation.)

### `strain_demo`

From within the `src` directory, we run the `strain_demo` with the input triangle mesh `data/square_100_unstructured.obj` as follows:

```
python -m demos.strain_demo ../data/square_100_unstructured.obj
```

The `Command UI` panel on the right-hand side of the screen allows the user to control the cloth simulation.

#### System Parameters

The system parameters subpanel allows the user to control the simulation's underlying system parameters.

1. The `Initialize / reset` button and `Point mass` float input field are identical to that of the `spring_demo`.

2. The `Singular values epsilon` float input field allows the user to specify "how much triangles can stretch past the strain constraint being locally satisfied". (Behind the scenes projective dynamics uses singular value decomposition (SVD) to "project" the current configuration of a triangle onto the constraint maniold SO(3). To ensure the rotation matrix (orthonormal matrix in the SVD) is in fact orthnormal, we clip the singular values to the range [1, 1]. The epsilon value is used to clip the singular values to [1 - epsilon, 1 + epsilon], allowing isotropic strain limiting where the strain constraint now is not binary and has a controllable epsilon threshold for "how much the constraint needs to be satisfied.")

3. The `Pin Selected Vertex` button is identical to that of the `spring_demo`. **The only difference is that when selecting a vertex, the `Selection` panel that pops up under the `Command UI` panel needs to specify that the selection is a *`vertex`* and no other halfedge mesh element**.

#### Simulation Control

The simulation control subpanel is identical to that of the `spring_demo`.

## References

1. Sofien Bouaziz et al. 2014. Projective dynamics: fusing constraint projections for fast simulation. [https://www.projectivedynamics.org/projectivedynamics.pdf](https://www.projectivedynamics.org/projectivedynamics.pdf)
