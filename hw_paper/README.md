# Papers

Goal: Get proficiency in cutting-edge equivariant, geometric and topological deep learning.

## Register to present one research paper

First come, first serve. Send a message on slack to register your team.

### Equivariant Deep Learning.

01/22/2024 Wednesday.
- [ ] Cohen and Welling. Group Equivariant neural networks. (2016).
- [ ] Marcos et al. Rotation equivariant vector field networks. (2017).

01/24/2024 Monday.
- [ ] Hinton et al. Matrix capsules with EM routing. (2018).
- [ ] Cohen and Welling. Steerable CNNs. (2017). AND. Weiler et al. 3D Steerable cnns: Learning rotationally equivariant features in volumetric data. (2018).

01/29/2024 Wednesday. 
- [ ] Cohen et al. Spherical CNNs (2018). AEsteves et al. Learning SO(3) equivariant representations with spherical CNNs. (2018).
    
01/31/2024 Monday. 
- [ ] Cohen et al. Gauge equivariant convolutional networks and the icosahedral CNN (2019).
- [ ] Finzi et al. A practical method for constructing equivariant multilayer perceptrons for arbitrary matrix groups (2021).

02/05/2024 Wednesday. 
- [ ] Davidson et al. Hyperspherical Variational Auto-Encoders. (2018).
- [ ] Ganea et al. Hyperbolic Neural Networks. (2018)

02/07/2024 
- [ ] Qi et al. Pointnet: Deep learning on point sets for 3d classification and segmentation. (2017). Qi et al. PointNet++: Pointnet++: Deep hierarchical feature learning on point sets in a metric space (2017).
- [ ] Thomas et al. Tensor field networks: Rotation-and translation-equivariant neural networks for 3D point clouds. (2018).

02/12/2024	Monday. 
- [ ] Chami et al. Hyperbolic Graph Convolutional Neural Networks. (2019) AND Liu et al. Hyperbolic Graph Neural Networks. (2019).
- [ ] Brehmer et al. Geometric Algebra Transformers (2023).

02/14/2024	Wednesday.
- [ ] Dosovitski et al. An Image is Worth 16x16 Words: Transformers Image Recognition at Scale. (2021).
- [ ] Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. (2021).

02/19/2024 Monday - President's Day. No class.

02/21/2024	Wednesday. 
- [ ] Masci et al. Geodesic convolutional neural networks on Riemannian manifolds. (2015).
- [ ] Boscaini et al. Learning shape correspondence with anisotropic convolutional neural networks (2016). 

02/26/2024	Monday. 
- [ ] Kipf and Welling. GCNs: Semi-Supervised Classification with Graph Convolutional Networks. (2016).
- [ ] Gilmer et al. Neural Message Passing for Quantum Chemistry. (2017).
    
02/28/2024	Wednesday. 
- [ ] De Haan et al. Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs. (2021).
- [ ] Veličković et al. Graph Attention Networks. (2018). AND: Brody et al. How Attentive Are Graph Attention Networks? (2022)

03/04/2024	Monday. 
- [ ] Feng et al. Hypergraph Neural Networks (2019).
- [ ] Bodnar et al. Weisfeiler and lehman go cellular: CW networks (2021).

03/06/2024	Wednesday. 
- [ ] Fuchs et al. SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks. (2020).
- [ ] Hutchinson et al. Lietransformer: Equivariant self-attention for lie groups. (2021). 


## Guidelines

- [Create a GitHub account](https://github.com/).
- [Download and install Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
- Join the slack channel hw_geomviz: this will be where you can easily ask questions to Nina and to the class.

1. Set-up

From a terminal in your computer:

- Clone the GitHub repository **of the class ece594n** with [git clone](https://github.com/git-guides/git-clone).
- In the cloned folder `ece594n`, create a conda environment `conda env create -f environment.yml`.
- Activate the conda environemnt with `conda activate ece594n`.
- Create a new GitHub branch  with [git checkout -b](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches).
- Run the code in the folder associated to the manifold you chose (the "manifold folder").
  - The code might not run anymore due to package/module not found: 
    - in this case, use: `import sys; sys.path.append("path/to/folder/with/file")`
  - The code might not run anymore due to (ii) geomstats changes since last year: 
    - in this case, check the names of Python files, classes and functions.

2. Create tests that verify that the code runs correctly.

From the GitHub branch on your computer:

- In your manifold folder, create a file name `test_visualization_[name-of-your-manifold].py`
- In that .py file, copy-paste the content of this [file](https://github.com/geomstats/geomstats/blob/master/tests/tests_geomstats/test_visualization.py).
- Adapt the code so that it tests the **visualization functions of your manifold**: you should have one test function per function in your manifold.
- Test that your tests run using `pytest test_visualization_[name-of-your-manifold].py`.
- Remove portions of code that are not visualization-related and are duplicated from existing code in the geomstats repository.
- Verify that the code follows the Code Structure given in the next section.
- Put your code to international coding style standards using `black [name-of-your-manifold].py` and `isort [name-of-your-manifold].py` and `flake8 [name-of-your-manifold].py`, and similarly for the test file `test_visualization_[name-of-your-manifold].py`.
- Document your code with docstrings, see [here for docstring guidelines](https://github.com/geomstats/geomstats/blob/master/docs/contributing.rst#writing-docstrings).

3. Prepare the notebook for submission to Geomstats
- In your manifold folder, edit the .ipynb to change everything that you do not understand.
- If there exists a notebook on geomstats on a similar topic, create one notebook that merges the two.

4. Submit your Pull Request.
- Clone the GitHub repository **[of Geomstats](https://github.com/geomstats/geomstats)** with [git clone](https://github.com/git-guides/git-clone).
- Create a new GitHub branch **[of Geomstats]** on your computer with [git checkout -b](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches).
- **Expert project only:** you might need to pull an existing branch into your new GitHub branch:
  - Add the remote repository (different from origin): `git remote add ninamiolane git@github.com:ninamiolane/geomstats.git`
  - From your local branch: `git pull ninamiolane the-existing-branch`.
- On your branch, in the Geomstats' folder [visualization](https://github.com/geomstats/geomstats/tree/master/geomstats/visualization): add your `[name-of-your-manifold].py`.
- On your branch, in the Geomstats' folder [tests_geomstats](https://github.com/geomstats/geomstats/tree/master/tests/tests_geomstats): add your `test_visualization_[name-of-your-manifold].py`
- Submit your work as a [Pull Request (PR)](https://opensource.com/article/19/7/create-pull-request-github) to the Geomstats GitHub repository.
- You can submit code to your PR (i.e., modify your PR) anytime until the deadline.
- Your code should pass the GitHub Action tests: it will automatically verify that your code runs.
- Your code will receive a code review. Address the review, and resubmit.

![Using GitHub.](/lectures/figs/github.png)

## Information about Code Structure 

Design from Elodie Maignant, PhD student at INRIA (France) and contributor to Geomstats.

− In your manifold folder, you should see a file `[name-of-your-manifold].py`:
  - With a python class named after your manifold `class YourManifold`.
  - With visualization utils as methods from this class, such as:
    - `plot`: draws the manifold (e.g. cone for SPD matrices)
    - `plot_grid`: draws the manifold with a geodesic grid.
    - `plot_rendering`: draws the manifold with regularly sampled data.
    - `plot_tangent_space`: draws the tangent space to the manifold at a point.
    - `scatter`: plot a point cloud. Inherits matplotlib.scatter parameters.
    
    - `plot_geodesic` allows to visualise a (discretised) geodesic. Takes either point and tangent vec as parameters, or initial point and end point as parameters.
    
    − `plot_vector_field` allows to visualise vectors fields. Takes points and tangent vecs as parameters.


## Grading Criteria

- The code style is clean: it passes the GitHub Actions' **Linting** (10%).
- The code is properly documented following the documentation guidelines (10%).
- The code runs: it passes the GitHub Actions' **Testing** (20%).
- There is one test function per visualization function (20%).
- You have made substantial improvements compared to the current version of the code, or you chose an expert project (20%).
- You have addressed the comments in the code review and merged into Geomstats (20%).
