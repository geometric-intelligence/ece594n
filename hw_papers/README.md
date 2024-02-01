# Papers

Goal: Get proficiency in cutting-edge equivariant, geometric and topological deep learning.

## Register to present one research paper

First come, first serve. Send a message on the #general channel on slack to register.

### Symmetries

01/22/2024 Monday.
- [X] Daniel Ralston -- Cohen and Welling. Group Equivariant neural networks. (2016).
- [ ] Marcos et al. Rotation equivariant vector field networks. (2017).

01/24/2024 Wednesday.
- [X] Seoyeon. Cohen and Welling. Steerable CNNs. (2017).
- [ ] Lenssen et al. Group Equivariant Capsule Networks (2018).

01/29/2024 Monday. 
- [X] Ozgur Guldogan -- Finzi et al. A practical method for constructing equivariant multilayer perceptrons for arbitrary matrix groups (2021).
- [ ] Ruhe et al. Clifford Group Equivariant Neural Networks (2024).

01/31/2024 Wednesday.
- [ ] Cohen et al. Gauge Equivariant convolutional networks and the icosahedral CNN (2019).
- [ ] De Haan et al. Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs. (2021).

### Point Clouds and Meshes

02/05/2024 Monday.
- [X] Abhijit Brahme -- Masci et al. Geodesic convolutional neural networks on Riemannian manifolds. (2015).
- [X] Louisa Cornelis -- Boscaini et al. Learning shape correspondence with anisotropic convolutional neural networks (2016). 

02/07/2024 Wednesday.
- [X] Ana Cardenas -- Qi et al. Pointnet: Deep learning on point sets for 3d classification and segmentation. (2017). Qi et al. PointNet++: Pointnet++: Deep hierarchical feature learning on point sets in a metric space (2017). 
- [X] Jacob Lyons -- Thomas et al. Tensor field networks: Rotation-and translation-equivariant neural networks for 3D point clouds. (2018).

### Graphs and Topological Domains.

02/12/2024	Monday.
- [X] Caitlyn Linehan -- Kipf and Welling. GCNs: Semi-Supervised Classification with Graph Convolutional Networks. (2016).
- [X] Sofia Gonzalez Garcia -- Gilmer et al. Neural Message Passing for Quantum Chemistry. (2017).

02/14/2024	Wednesday.
- [ ] Chami et al. Hyperbolic Graph Convolutional Neural Networks. (2019) AND Liu et al. Hyperbolic Graph Neural Networks. (2019).
- [X] Tyler Hattori -- Ganea et al. Hyperbolic Neural Networks. (2018).

02/19/2024 Monday - President's Day. No class.

02/21/2024	Wednesday.
- [X] Shawn Catudal -- Feng et al. Hypergraph Neural Networks (2019).
- [X] Liyan Tan -- Bodnar et al. Weisfeiler and lehman go cellular: CW networks (2021).

### Transforners / Attention-based

02/26/2024	Monday. 
- [X] Alan Raydan -- Veličković et al. Graph Attention Networks. (2018). AND: Brody et al. How Attentive Are Graph Attention Networks? (2022)
- [X] Alexander Davydov -- Dwivedi et al. A Generalization of Transformer Networks to Graphs. (2019).

02/28/2024	Wednesday. 
- [X] Jiahua Chen -- Dosovitski et al. An Image is Worth 16x16 Words: Transformers Image Recognition at Scale. (2021).
- [X] Shih-Cheng Hsiao -- Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. (2021).

03/04/2024	Monday.
- [X] Zhiyu Xue -- Tai et al. Equivariant Transformer Networks. (2019).
- [ ] Beckers et al. Fast, Expressive SE(n) Equivariant Networks through Weight-Sharing in Position-Orientation Space. (2024).

03/06/2024	Wednesday. 
- [X] George Hulsey -- Hutchinson et al. Lietransformer: Equivariant self-attention for lie groups. (2021).
- [ ] Brehmer et al. Geometric Algebra Transformers (2023).

## Guidelines

- [Create a GitHub account](https://github.com/).
- [Download and install Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
- Join the slack channel hw_papers: this will be where you can easily ask questions to Nina and to the class.

1. Set-up

From a terminal in your computer:

- Clone the GitHub repository **of the class ece594n** with [git clone](https://github.com/git-guides/git-clone).
- Create a new GitHub branch  with [git checkout -b](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches).

2. Create a jupyter notebook that shows your deep learning pipeline in action.

From the GitHub branch on your computer:

- Create a new folder inside `hw_papers` called [first-author-title-year].
- Inside this folder:
  - Add a environment.yml with the Python packages required to run your code.
  - Add Python files as needed.
  - Add a Jupyter notebook, or a Google Colab, called `main.ipynb` that showcases the technique from your paper. Needs to run in class.
  - Add a PDF with the slides from your presentation in class (30 minutes).

3. Submit your Pull Request.
- Submit your work as a [Pull Request (PR)](https://opensource.com/article/19/7/create-pull-request-github) to this GitHub repository.
- You can submit code to your PR (i.e., modify your PR) anytime until the day of your presentation.

4. Prepare your presentation (30 min), using the following structure:

- Introduction
- Related Works
- Background
- Methods
- Results
- Demonstration of the code.
  
Use the figures of the paper as much as possible. You can show YouTube videos.


## Grading Criteria

- The presentation of the equivariant, geometric or topological concepts is precise and intuitive.
- The presentation in class is clear, the slides are clean, the paper has been understood. 
- The code style is clean: it passes the GitHub Actions' **Linting** and it is properly documented.
- The code runs.
