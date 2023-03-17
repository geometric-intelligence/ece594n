# Predicting Cell Behavior from Cell Morphology during Cell-Cell Shear

### ECE594N HW Shapes 

__Samuel Feinstein, Amil Khan, Lauren Washington, Alice Mo__

- __Sam__ provided the idea of the project, data, and pre-processing of the data
- __Amil__ built the dictionary representation, wrote helper functions, and performed cell analysis
- __Lauren__ provided the idea of using SRV, and provided initial visualizations
- __Alice__ lead the entire visualization portion

As a side note, our first initials are one letter away from SALAD.

📸 Images from Liam Dow

---

#### Data Source

> Ehsan Sadeghipour, Miguel A Garcia, William James Nelson, Beth L Pruitt (2018) Shear-induced damped oscillations in an epithelium depend on actomyosin contraction and E-cadherin cell adhesion eLife 7:e39640 https://doi.org/10.7554/eLife.39640



# Introduction and Motivation

Cell-cell shear, or the action of cells sliding past each other, has roles in development, disease, and wound healing. Throughout development cells are moving past each other in every stage of development. These biomechanical cues have influences on differentiation, cell shape, behavior, the proteome, and the transcriptome. 

Previous research on shear focused on fluid shear so in this paper they focused on cell-cell shear which has been well characterized. Epithelial cells known as MDCK cells were used on a MEMS device which can be precisely displaced to create consistent cell-cell shear forces. Using new segmentation and machine learning techniques we are reanalyzing the data to use the changes in cell shape to predict cell behavior/migration.
