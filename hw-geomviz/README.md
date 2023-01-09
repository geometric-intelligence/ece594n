# Geomviz: Visualizing Differential Geometry

Goal: Get intuition on Differential Geometry by visualizing manifolds.

- Deadline: Monday 01/30/2023
- Team of 1-4 students.

## Register your team

First come, first serve. Send a message on slack to register your team.

- [ ] Stiefel manifold: TBD.
- [ ] Grassmanian manifold:  TBD.
- [ ] Manifold of symmetric positive definite (SPD) matrices:  TBD.
- [ ] Special Euclidean group:  TBD.
- [ ] Discrete curves:  TBD.
- [ ] Manifold of beta distributions:  TBD.
- [ ] Manifold of categorical distributions: TBD.

## Guidelines

- Clone the GitHub repository of the class with [git clone](https://github.com/git-guides/git-clone).
- Create a conda environment with `conda env create -f environment.yml`.
- Create a new GitHub branch on your computer with [git checkout -b](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches).
- Run the code in the folder associated to the manifold you chose (the "manifold folder").
- Create tests that verify that the code runs correctly:
  - In your manifold folder, create a file name `test_visualization_[name of your manifold].py`
  - In that .py file, copy-paste the content of this [file](https://github.com/geomstats/geomstats/blob/master/tests/tests_geomstats/test_visualization.py).
  - Adapt the code so that it tests the functions of your manifold: you should have one test function per function in your manifold.
  - Test that your tests run using `pytest test_visualization_[name of your manifold].py`.
- Verify that the code follows the Code Structure given in the next section.
- Put the code to international coding style standards using `black [name-of-python-file.py]` and `isort [name-of-python-file.py]` and `flake8 [name-of-python-file.py]`.
- Submit your work as a [Pull Request (PR)](https://opensource.com/article/19/7/create-pull-request-github) to the Geomstats GitHub repository.
- You can submit code to your PR anytime until the deadline.

## Information about Code Structure 

Design from Elodie Maignant, PhD student at INRIA (France) and contributor to Geomstats.

− In your manifold folder, you should see a file `[your-manifold].py`:
  - With a python class named after your manifold `class YourManifold`.
  - With visualization utils as methods from this class, such as:
    - `plot`: draws the manifold (e.g. cone for SPD matrices)
    - `plot_grid`: draws the manifold with a geodesic grid.
    - `plot_rendering`: draws the manifold with regularly sampled data.
    - `plot_tangent_space`: draws the tangent space to the manifold at a point.
    - `scatter`: plot a point cloud. Inherits matplotlib.scatter parameters.
    
    - `plot_geodesic` allows to visualise a (discretised) geodesic. Takes either point and tangent vec as parameters, or initial point and end point as parameters.
    
    − `plot_vector_field` allows to visualise vectors fields. Takes points and tangent vecs as parameters.


## Additional remarks

- Your code should be documented with docstrings, see [here for docstring guidelines](https://github.com/geomstats/geomstats/blob/master/docs/contributing.rst#writing-docstrings).
- Your visualization functions should work with only matplotlib.
- Your code should pass the GitHub Action tests when you push your PR to Geomstats GitHub.


## Grading Criteria

- The code style is clean: it passes the GitHub Actions Linting Tests.
- There is one test function per visualization function.
- The code runs: it passes the GitHub Actions Testing.
- The visualizations are clean and will be merged into Geomstats.
