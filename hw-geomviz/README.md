# Geomviz: Visualizing Differential Geometry

Goal: Get intuition on Differential Geometry by visualizing manifolds.

- Deadline: Monday 01/30/2023
- Team of 2-4 students.

## Register your team

First come, first serve. Send a message on slack to register your team.

- [ ] Stiefel manifold: TBD.
- [ ] Grassmanian manifold:  TBD.
- [ ] Manifold of symmetric positive definite (SPD) matrices:  TBD.
- [ ] Hyberbolic spaces:  TBD.
- [ ] Special Euclidean group:  TBD.
- [ ] Heisenberg group: TBD.
- [ ] Discrete curves:  TBD.
- [ ] Manifold of beta distributions:  TBD.
- [ ] Manifold of categorical distributions: TBD.

## Guidelines

More to come.

- Submit your work as a [Pull Request (PR)](https://opensource.com/article/19/7/create-pull-request-github) to this Geomstats GitHub repository.
- You can submit code to your PR anytime until the deadline.

## Important Remarks

- The elementary operations (exp, log, geodesics, etc) are already implemented in [geomstats/geometry](https://github.com/geomstats/geomstats/tree/master/geomstats/geometry):
  - Search for the file that represents your manifold in this folder.
  - Your goal is not to re-implement them, but to use them and visualize them.
- Some visualizations are already implemented in [geomstats/visualization.py](https://github.com/geomstats/geomstats/blob/master/geomstats/visualization.py) for some manifolds.
  - Search if your manifold has some visualization implemented and think about how to improve it.
  - You can use what is already implemented.
- The folders of [examples](https://github.com/geomstats/geomstats/tree/master/examples) and [notebooks](https://github.com/geomstats/geomstats/tree/master/notebooks) show you examples on how to use the operations on your manifold.
  - Search for your manifold in these files to get ideas.
- Higher-dimensional manifolds: (i) do not plot them or (ii) find a "trick" to plot them
- Note that some visualizations are already available in Geomstats for some manifolds listed above.

## Code Structure 

Design from Elodie Maignant, PhD student at INRIA (France) and contributor to Geomstats.

− In your file `[your-manifold].py`:
  - Create a python class named after your manifold `class YourManifold`.
  - Add visualization utils as methods from this class, such as:
    - `plot`: draws the manifold (e.g. cone for SPD matrices)
    - `plot_grid`: draws the manifold with a geodesic grid.
    - `plot_rendering`: draws the manifold with regularly sampled data.
    - `plot_tangent_space`: draws the tangent space to the manifold at a point.
    - `scatter`: plot a point cloud. Inherits matplotlib.scatter parameters.
    
    - `plot_geodesic` allows to visualise a (discretised) geodesic. Takes either point and tangent vec as parameters, or initial point and end point as parameters.
    
    − `plot_vector_field` allows to visualise vectors fields. Takes points and tangent vecs as parameters.
    
− In your notebook `[your-manifold].ipynb`, respect the following outline:
  - 1. Introduction.
  - 2. Mathematical definition of [your manifold].
  - 3. Uses of [your manifold] in real-world applications.
  - 4. Visualization of elementary operations on [your manifold]
    - Showcase your visualization can be used by plotting the inputs and outputs of operations such as exp, log, geodesics.
  - 5. Conclusion and references


## Additional remarks

- Your code should be documented with docstrings, see [here for docstring guidelines](https://github.com/geomstats/geomstats/blob/master/docs/contributing.rst#writing-docstrings).
- You do not have to implement all of the methods listed above. You can implement methods not listed.
- Your visualization functions should work with only matplotlib.
- You can add options for your functions to be interactive, using plotly or bokeh in addition to matplotlib.
- Here is an [example of a visualization project by Elodie Maignant](https://github.com/geomstats/geomstats/blob/master/notebooks/16_real_world_applications__visualizations_in_kendall_shape_spaces.ipynb).


## Grading Criteria

TBD.
