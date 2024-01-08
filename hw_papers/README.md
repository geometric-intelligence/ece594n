# Papers

Goal: Get proficiency in cutting-edge equivariant, geometric and topological deep learning.

## Register to present one research paper

First come, first serve. Send a message on slack to register.

### Symmetries

01/22/2024 Wednesday.
- [ ] Cohen and Welling. Group Equivariant neural networks. (2016).
- [ ] Marcos et al. Rotation equivariant vector field networks. (2017).

01/24/2024 Monday.
- [ ] Cohen and Welling. Steerable CNNs. (2017).
- [ ] Lenssen et al. Group Equivariant Capsule Networks (2018).

01/29/2024 Wednesday. 
- [ ] Cohen et al. Spherical CNNs (2018).
- [ ] Finzi et al. A practical method for constructing equivariant multilayer perceptrons for arbitrary matrix groups (2021).

01/31/2024 Monday. 
- [ ] Cohen et al. Gauge Equivariant convolutional networks and the icosahedral CNN (2019).
- [ ] De Haan et al. Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs. (2021).

### Point Clouds and Meshes

02/05/2024 Wednesday. 
- [ ] Masci et al. Geodesic convolutional neural networks on Riemannian manifolds. (2015).
- [ ] Boscaini et al. Learning shape correspondence with anisotropic convolutional neural networks (2016). 

02/07/2024 
- [ ] Qi et al. Pointnet: Deep learning on point sets for 3d classification and segmentation. (2017). Qi et al. PointNet++: Pointnet++: Deep hierarchical feature learning on point sets in a metric space (2017).
- [ ] Thomas et al. Tensor field networks: Rotation-and translation-equivariant neural networks for 3D point clouds. (2018).

### Graphs and Topological Domains.

02/12/2024	Monday.
- [ ] Kipf and Welling. GCNs: Semi-Supervised Classification with Graph Convolutional Networks. (2016).
- [ ] Gilmer et al. Neural Message Passing for Quantum Chemistry. (2017).

02/14/2024	Wednesday.
- [ ] Chami et al. Hyperbolic Graph Convolutional Neural Networks. (2019) AND Liu et al. Hyperbolic Graph Neural Networks. (2019).
- [ ] Ganea et al. Hyperbolic Neural Networks. (2018).

02/19/2024 Monday - President's Day. No class.

02/21/2024	Wednesday.
- [ ] Feng et al. Hypergraph Neural Networks (2019).
- [ ] Bodnar et al. Weisfeiler and lehman go cellular: CW networks (2021).

### Transforners / Attention-based

02/26/2024	Monday. 
- [ ] Veličković et al. Graph Attention Networks. (2018). AND: Brody et al. How Attentive Are Graph Attention Networks? (2022)
- [ ] Dwivedi et al. A Generalization of Transformer Networks to Graphs. (2019).

02/28/2024	Wednesday. 
- [ ] Dosovitski et al. An Image is Worth 16x16 Words: Transformers Image Recognition at Scale. (2021).
- [ ] Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. (2021).

03/04/2024	Monday.
- [ ] Tai et al. Equivariant Transformer Networks. (2019).
- [ ] Fuchs et al. SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks. (2020).

03/06/2024	Wednesday. 
- [ ] Hutchinson et al. Lietransformer: Equivariant self-attention for lie groups. (2021).
- [ ] Brehmer et al. Geometric Algebra Transformers (2023).


## Guidelines

- [Create a GitHub account](https://github.com/).
- [Download and install Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
- Join the slack channel hw_papers: this will be where you can easily ask questions to Nina and to the class.

1. Set-up

From a terminal in your computer:

- Clone the GitHub repository **of the class ece594n** with [git clone](https://github.com/git-guides/git-clone).
- In the cloned folder `ece594n`, create a conda environment `conda env create -f environment.yml`.
- Activate the conda environemnt with `conda activate ece594n`.
- Create a new GitHub branch  with [git checkout -b](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches).

2. Create a jupyter notebook that shows your deep learning pipeline in action.

From the GitHub branch on your computer:

- Create a new folder inside `hw_papers` called [first-author-title-year].
- Inside this folder:
  - Add a environment.yml with the Python packages required to run your code. Note that you ca
  - Add Python files as needed.
  - Add a Jupyter notebook called `main.ipynb` that showcases the technique from your paper. Needs to run in class.
  - Add a PDF with the slides from your presentation in class (30 minutes).

3. Submit your Pull Request.
- Submit your work as a [Pull Request (PR)](https://opensource.com/article/19/7/create-pull-request-github) to this GitHub repository.
- Your code should pass the GitHub Action tests: it will automatically verify that your code runs, and that your code is clean.
- You can submit code to your PR (i.e., modify your PR) anytime until the day of your presentation.

4. Prepare your presentation, using the following structure:
- Introduction
- Related Works
- Methods
- Results
Use the figures of the paper as much as possible. You can show YouTube videos.


## Grading Criteria

- The presentation of the equivariant, geometric or topological concepts is precise and intuitive.
- The presentation in class is clear, the slides are clean, the paper has been understood. 
- The code style is clean: it passes the GitHub Actions' **Linting** and it is properly documented.
- The code runs.