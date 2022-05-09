# Reviews

Goal: Read and implement/review published papers in Geometric Machine Learning.

- Deadline: Tuesday 06/07/2022
- Teams of 1 student.

_*Important Note:*_ This HW assignment has different guidelines for Bio and ECE students. Look for the guidelines that correspond to your home department!


## Guidelines for Bio Students

1. Choose one of the topics below, in the section "Topics for Bio Students": write on slack to have the topic assigned to you.
If there is another topic you are interested in that is not listed, you are free to propose an alternative to Prof. Miolane for approval.
1. Identify and read 5+ published research papers on your topic.
2. Summarize them in a Jupyter notebook. The notebook should be ~8 pages in length and have the following structure:

I. Introduction (Markdown)

- Set the context and introduce your topic? Why is this important?

II. Literature review: For each of the papers you found:

- Give its summary
- explain how it fits into the topic you are assigned
- explain how it answers the question asked by your topic
- explain how it relates to the other papers

III. Conclusion

- Give your personal conclusions on the topic.

## Topics for Bio Students

- [x] **Assigned to Rebecca Martin.** “Proteins representations - protein sequences or protein shapes”
    - What are meaningful representations of proteins that can best answer questions in biology using Machine Learning?
- [ ] “Cells representations - cell genetics or cell morphologies”
    - What are meaningful representations of cells that can best answer questions in biology using Machine Learning?
- [ ] “Brain representations with brain shapes”
    - Which conditions or pathologies are the most correlated with brain shapes?
- [ ] “Heart representations with heart shapes”
    - Which conditions or pathologies are the most correlated with heart shapes?

## Guidelines for ECE Students

1. Choose 1 of the papers listed below (in the section "Papers for ECE students". The (\*) indicates topics that we will not fully cover in class, thus harder to understand and implement): write on slack to have the paper assigned to you. If there is a paper you like that is not listed, you are free to propose an alternative to Prof. Miolane for approval.
2. Implement the algorithm in the paper using Geomstats elementary operations. 
3. Present your implementation in a Jupyter notebook. The notebook should be ~4 pages in length (when printed to pdf) and have the following structure:
    
    I. Introduction (Markdown)
    
    - Set the context. What problem is this paper trying to solve? Why is this important?
    - Brief literature review: Identify 3+ related papers/models and briefly explain how they relate to one another (1 - 2 sentences per paper).
    
    II. Background and Model (Markdown)
    
    - Introduce and explain the model and any necessary mathematical background. Use LaTeX for equations.
    
    III. Implementation (Code)
    
    - Implement the model (for the manifold used for the paper). All code should be documented.
    
    IV. Demonstration and Analysis (Code / Markdown)
    
    - Demonstrate the model on a dataset and analyze the results, using visualizations when possible. The exact analyses you perform are up to you, but some general suggestions include:
        - Comparing the results to conventional algorithms already implemented in libraries such as scikit-learn. For example, if your model is Geodesic PCA, compare your results to conventional PCA run on the same dataset.
        - Demonstrating the effects of different hyperparameter choices
        - Analyzing computational cost


## Papers for ECE students

### **Representation Learning**

**Manifold / Sub-Manifold Learning**

- [ ] Goh, A., Vidal, R.: Clustering and dimensionality reduction on riemannian manifolds. In: CVPR. IEEE Computer Society (2008)
- [ ] Hauberg, S. (2015). Principal curves on Riemannian manifolds. *IEEE transactions on pattern analysis and machine intelligence*, *38*(9), 1915-1921.
- [ ] Lin, T., & Zha, H. (2008). Riemannian manifold learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *30*(5), 796-809.
- [ ] Huckemann, Stephan, Thomas Hotz, and Axel Munk (2010). “Intrinsic Shape Analysis: Geodesic PCA for Riemannian Manifolds modulo Isometric Lie Group Actions”. In: Statistica Sinica 20.1, pp. 1–58.
- [ ] Zhang, M., Fletcher, P.T.: Probabilistic principal geodesic analysis. In: Advances in Neural Information Processing Systems. pp. 1178–1186 (2013)
- [ ] Zhang, Y., Xing, J., & Zhang, M. (2019). Mixture probabilistic principal geodesic analysis. In *Multimodal Brain Image Analysis and Mathematical Foundations of Computational Anatomy* (pp. 196-208). Springer, Cham.
- [ ] Sommer, S., Lauze, F., & Nielsen, M. (2014). Optimization over geodesics for exact principal geodesic analysis. *Advances in Computational Mathematics*, *40*(2), 283-313.
- [ ] Pennec, X. (2018). Barycentric subspace analysis on manifolds. *The Annals of Statistics*, *46*(6A), 2711-2746.
- [ ] Harandi, M., Hartley, R., Salzmann, M., & Trumpf, J. (2016). Dictionary learning on Grassmann manifolds. In *Algorithmic advances in Riemannian geometry and applications*
 (pp. 145-172). Springer, Cham.

**Embeddings**

- [ ] Nickel, Maximillian and Douwe Kiela (2017). “Poincaré Embeddings for Learning Hierarchical Representations”. In: Advances in Neural Information Processing Systems 30. Ed. by I Guyon, U V Luxburg, S Bengio, H Wallach, R Fergus, et al. Curran Associates, Inc.

### Regression / Interpolation / Extrapolation

- [ ] Gawlik, Evan S. and Melvin Leok (June 2018). “Interpolation on Symmetric Spaces Via the Generalized Polar Decomposition”. en. In: Foundations of Computational Mathematics 18.3, pp. 757–788. issn: 1615-3375, 1615-3383. doi: 10.1007/s10208- 017-9353-0.
- [ ] Hinkle, J., Muralidharan, P., Fletcher, P. T., & Joshi, S. (2012, October). Polynomial regression on Riemannian manifolds. In *European conference on computer vision*
 (pp. 1-14). Springer, Berlin, Heidelberg.
- [ ] Kim, Kwang-Rae, Ian L. Dryden, Huiling Le, and Katie E. Severn (Dec. 2020). “Smoothing Splines on Riemannian Manifolds, with Applications to 3D Shape Space”. en. In: Journal of the Royal Statistical Society: Series B (Statistical Methodology).
- [x] **Assigned to Jax Burd.** Gousenbourger, P. Y., Massart, E., & Absil, P. A. (2019). Data fitting on manifolds with composite Bézier-like curves and blended cubic splines. *Journal of Mathematical Imaging and Vision*, *61*(5), 645-671.
- [ ] Hinkle, J., Fletcher, P. T., & Joshi, S. (2014). Intrinsic polynomials for regression on Riemannian manifolds. *Journal of Mathematical Imaging and Vision*, *50*(1), 32-52.
- [ ] Steinke, F., Hein, M., & Schölkopf, B. (2010). Nonparametric regression between general Riemannian manifolds. *SIAM Journal on Imaging Sciences*, *3*(3), 527-563.
- [ ] Singh, N., Vialard, F. X., & Niethammer, M. (2015). Splines for diffeomorphisms. *Medical image analysis*, *25*(1), 56-71.

### Kernel Methods (*)

- [ ] Jayasumana, S., Hartley, R., Salzmann, M., Li, H., & Harandi, M. (2015). Kernel methods on Riemannian manifolds with Gaussian RBF kernels. *IEEE transactions on pattern analysis and machine intelligence*, *37*(12), 2464-2477.
- [ ] Jayasumana, S., Hartley, R., Salzmann, M., Li, H., & Harandi, M. (2013). Kernel methods on the Riemannian manifold of symmetric positive definite matrices. In *proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 73-80).

### Time-Series Analysis (*)

- [ ] Bourmaud, G., Mégret, R., Giremus, A., & Berthoumieu, Y. (2013, September). Discrete extended Kalman filter on Lie groups. In *21st European Signal Processing Conference (EUSIPCO 2013)*(pp. 1-5). IEEE.
- [ ] Unscented Kalman Filtering on Riemannian Manifolds. Søren Hauberg, Françous Lauze, and Kim S. Pedersen. *Journal of Mathematical Imaging and Vision*, 46(1):103-120, May 2013.

### Generative Modeling (*)

- [ ] De Bortoli, V., Mathieu, E., Hutchinson, M., Thornton, J., Teh, Y. W., & Doucet, A. (2022). Riemannian score-based generative modeling. *arXiv preprint arXiv:2202.02763*.

### Deep Learning (*)

- [ ] Masci, J., Boscaini, D., Bronstein, M., & Vandergheynst, P. (2015). Geodesic convolutional neural networks on riemannian manifolds. In *Proceedings of the IEEE international conferenc*
- [ ] Cohen, T., Weiler, M., Kicanaoglu, B., & Welling, M. (2019, May). Gauge equivariant convolutional networks and the icosahedral CNN. In *International conference on Machine learning*(pp. 1321-1330). PMLR.
- [ ] Cohen, T. S., Geiger, M., Köhler, J., & Welling, M. (2018). Spherical cnns. *arXiv preprint arXiv:1801.10130.*
- [x] **Assigned to Xinling Yu.** Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., & Riley, P. (2018). Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. *arXiv preprint arXiv:1802.08219*
