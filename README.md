# UCSB ECE 594N Geometric Machine Learning for Biomedical Imaging and Shape Analysis

Welcome!

This is the GitHub repository for the course:

ECE 594N: Geometric Machine Learning for Biomedical Imaging and Shape Analysis at UC Santa Barbara, Spring 2022.

- Instructor: [Prof. Nina Miolane](https://www.ece.ucsb.edu/people/faculty/nina-miolane), [BioShape Lab](https://bioshape.ece.ucsb.edu/), UC Santa Barbara.
- Lectures: Tuesdays, Thursdays 4 PM - 5.30 PM in location TBD.

You can access and run the lecture slides and lab notebooks by clicking on the Binder link below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bioshape-lab/ece594n/main?filepath=lectures)

# Communicating

- Join ECE 594n Slack workspace with your @ucsb.edu email address through the invitation you have received via email.

- Make it a habit to check ECE 594n Slack several times a week.

- Slack is our prefered way of communicating. Avoid emails and use Slack to ask questions about syllabus, lectures, project.

# Resources

The content of this class relies on the following resources:
- [Geomstats software](https://github.com/geomstats/geomstats) by Geomstats open-source contributors.
- [Geomstats tutorials](https://github.com/geomstats/geomstats/blob/master/notebooks/) including introductory notebooks by Adele Myers.
- Paper by Guigui, Miolane, Pennec 2022. Introduction to Riemannian Geometry and Geometric Statistics: from basic theory to implementation with Geomstats. Founadtions and Trends in Machine Learning.
- [Pytorch-geometric software]()
- Paper by Bronstein et al 2021. Geometric Deep Learning.


# Syllabus

Advances in biomedical imaging techniques have enabled us to access the shapes of a variety of structures: organs, cells, proteins. Since biological shapes are related to physiological functions, shape data may hold the key to unlock outstanding mysteries in biomedicine — specifically when combined with genomics and transcriptomics data.

<img src="https://raw.githubusercontent.com/bioshape-lab/ece594n/master/fig_readme.png" height="120px" width="120px" align="left">

Machine learning is poised to play a major role in analyzing this new wealth of imaging information and testing novel biomedical hypotheses.
In this class you will learn how to perform geometric machine learning on biomedical images and shapes. The course will cover basics of geometric machine learning and delve into specific methods for shape analysis of proteins, cells and organs. This course will feature guest lectures from invited speakers.

# Outline

The course is organized as follows:

Unit 0: Elements of Riemannian Geometry
- Manifolds, Connection, Riemannian Metric
Unit 1: Computational Models of Shapes
- Hand-crafted features of shapes: signed distance function, m-reps, s-reps, harmonics, laplacians?
- Objects: Point clouds, Curves, Graphs, Meshes (Justin's)
- Shape-invariant object transformations
- Quotient spaces: Kendall shape spaces, Curve shape spaces, Surface shape spaces
- Articulated models
- Deformations
Unit 2: Geometric Learning: Machine Learning on (Shape) Manifolds (Nina's)
- Supervised learning: predicting from a shape, predicting a shape?
- Unsupervised learning: Dimension reduction methods: PGA on shapes, Riemannian VAE
- Unsupervised learning: Clustering methods: Riemannian Meanshift, Riemannian KMeans, Riemannian EM
Unit 3: Geometric Deep Learning (Cohen's)
- Graph Neural Networks
- Group Equivariant Neural Networks
- Gauge Equivariant Neural Networks
Unit 4: Applications to Biomedical Shape Analyses
- Invited speakers?

# Grading

- HW1 (Reading): 25% - Deadline TBD.
- Mid-term project (Reproducibility): 20% - Deadline TBD.
- HW2 (Reading): 25% - Deadline TBD.
- Final project (Analysis): 30% - Deadline 05/27/2022.
- Extra-credits (Geomstats): up to +10% - Deadline: 05/27/2022.

Details will be given during the first lecture.


# Thank you and best wishes for the new Academic Year! ☺
