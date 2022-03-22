# Geomviz: Visualizing Differential Geometry

Goal: Get intuition on Differential Geometry by visualizing manifolds.

- Deadline: Thursday 04/21/2022.
- Team of 2-3 students.

# Register your team

First come, first serve. Send a message on slack to register your team.

- [ ] Stiefel manifold: TBD.
- [ ] Grassmanian manifold: TBD.
- [ ] Manifold of symmetric positive definite (SPD) matrices: TBD.
- [ ] Hyberbolic spaces: TBD.
- [ ] Special Euclidean group: TBD.
- [ ] Special Orthogonal group: TBD.
- [ ] Heisenberg group: TBD.
- [ ] Discrete curves: TBD.
- [ ] Manifold of beta distributions: TBD.
- [ ] Manifold of categorical distributions: TBD.

# Guidelines

- Create a new folder in this folder with the name of your manifold.
- In this new folder:
  - Add a python file named `[your-manifold].py`, e.g. `grassmanian.py`
    - This file will have the core functions for your project (details below).
  - Add a Jupyter notebook named `[your-manifold].ipynb` that represents the bulk of your project.
- Make sure to indicate which teammate did which part of the work.
- You can submit code to this repository anytime until the deadline.

# Code Structure 

Design from Elodie Maignant, PhD student at INRIA (France) and contributor to Geomstats.

− In your file `[your-manifold].py`:
  - Create a python class named after your manifold `class YourManifold`.
  - Add visualization utils as methods from this class, such as:
    - `plot`: draws the manifold (e.g. cone for SPD matrices)
    - `plot_grid`: draws the manifold with a geodesic grid.
    - `plot_rendering`: draws the manifold with regularly sampled data.
    - `scatter`: plot a point cloud. Inherits matplotlib.scatter parameters.
    
    - `plot_geodesic` allows to visualise a (discretised) geodesic. Takes either point and tangent vec as parameters, or initial point and end point as parameters.
    
    − `plot_vector_field` allows to visualise vectors fields. Takes points and tangent vecs as parameters.
    
− In your notebook `[your-manifold].ipynb`:
  - Show how your visualization can be used.


Additional remarks:
- Your code should be documented with docstrings, see [here for docstring guidelines](https://github.com/geomstats/geomstats/blob/master/docs/contributing.rst#writing-docstrings).
- You do not have to implement all of the methods listed above. You can implement methods not listed.
- Your functions should work with only matplotlib.
- You can add options for your functions to be interactive, using plotly or bokeh in addition to matplotlib.
