MPI-FAUST DATA SET

Copyright (c) 2014 Federica Bogo, Javier Romero, Matthew Loper, Michael Black, 
Max-Planck Institute for Intelligent Systems, Tuebingen

This document describes the full MPI-FAUST dataset, as published in

Bogo, F., Romero, J., Loper, M., Black, M.:
"FAUST: Dataset and evaluation for 3D mesh registration", CVPR 2014

For questions and comments contact faust@tue.mpg.de.



CHANGELOG
=========

2014/06/20      First version of the archives.



HOW TO USE
==========

1) Get the data.
    The data is part of the same archive as this README. For the directory 
    layout, see below.

2) Compute the scan-to-scan correspondences for the data in the testing set.
    There are two challenges: intra-subject (defined over 60 scan pairs) and
    inter-subject (defined over 40 scan pairs). 
    The scan pairs included in each challenge are listed in the files "intra_challenge.txt"
    and "inter_challenge.txt" in test/challenge_pairs. For the format of these files, see below.
    You can submit your results for one of the challenges, or for both. Each challenge
    requires the submission of a separate zip archive file. For the required archive format,
    see below.
     
3) Log in on the website.
    For logging in, you should use the same account you used for downloading the data.

4) Once logged in, select a challenge and create a new method, or modify an existing one.
    If you want to submit your results for both the intra- and inter-subject challenges,
    follow points 4) and 5) twice, once per challenge. 

5) Upload your data to the website.
    Upload the zip archive containing your results. We will automatically compute
    the errors of your method, and generate a visualization of your results.
    You can then choose to either publish your results to the website, or keep
    them private.



DIRECTORY STRUCTURE
===================

This section describes the structure of the full MPI-FAUST dataset.

training/
    Training set.
    
training/scans/
    The 100 scans included in the training set.
    
training/registrations/
    100 registrations, one for each scan included in the training set.
        
training/ground_truth_vertices/
    For each scan included in the training set, we provide the list
    of the vertices we consider reliably registered by the corresponding registration.
    For the format of these files, see below. 
    
test/
    Testing set. For this set, only the scans are given. The registrations for
    these scans are withheld for evaluation purposes.
    
test/scans/
    The 200 scans included in the testing set.
    
challenge_pairs/
    Two TXT files, each providing a list of scan pairs. "intra_challenge.txt" lists
    the pairs included in the intra-subject challenge. "inter_challenge.txt" lists
    the pairs included in the inter-subject challenge.



DATA FORMAT
===========

Scans, registrations:
    Given as PLY files. 
    Each registration's filename reports the same id of the corresponding scan's filename:
    e.g., "tr_reg_000.ply" is the registration corresponding to scan "tr_scan_000.ply".
    
Ground-truth vertices:
    Given as TXT files.
    Each filename reports the same id of the corresponding scan's filename:
    e.g., "tr_gt_000.txt" gives the ground-truth vertices for scan "tr_scan_000.ply".
    Each file has a number of lines equal to the corresponding scan's number of vertices.
    Each line reports either a value of 1, denoting a reliably registered vertex, or 0, otherwise.

Challenge pairs list:
    Given as TXT files.
    Each line reports a scan pair for which we require the computation of correspondences.
    E.g., a line reporting:
    000_001
    denotes the pair given by the scan "test_scan_000.ply" and the scan "test_scan_001.ply". 


SUBMISSION FILE FORMAT
======================

Zip archive:
    Each challenge requires the submission of a different zip archive.
    Each archive must contain one TXT file per each scan pair in the challenge (see below for the
    TXT file format). So, your zip file must contain 60 TXT files if you are submitting results
    for the intra-subject challenge, 40 if you are submitting results for the inter-subject challenge.

Per-pair TXT files:
    Each TXT file must be named according to the notation used in the challenge pairs files.
    E.g., a file providing results for the scan pair 000_001 must be named "000_001.txt".   
    For each vertex of the first scan in the pair, you must provide a corresponding 3D point on the
    surface of the second scan.
    The corresponding point is provided as a line containing its X, Y, Z coordinates.
    E.g., a file containing as initial lines:
    
    1.111111 2.222222 3.333333
    4.444444 5.555555 6.666666
    
    associates vertices 0 and 1 of the first scan to the 3D points [1.111111, 2.222222, 3.333333] and
    [4.444444, 5.555555, 6.666666], respectively.
    If the provided corresponding point is not a surface point, we compute the closest
    point on the surface and use this.
  


FURTHER INFORMATION
===================

More information and data can be obtained from http://faust.is.tue.mpg.de.

The dataset is published as
    Bogo, F., Romero, J., Loper, M., Black, M.:
    "FAUST: Dataset and evaluation for 3D mesh registration", CVPR 2014
    
If you use this work, please cite:

@inproceedings{Bogo:CVPR:2014,
    title = {{FAUST}: Dataset and evaluation for {3D} mesh registration},
    author = {Bogo, Federica and Romero, Javier and Loper, Matthew and Black, Michael J.},
    booktitle = { Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    address = {Piscataway, NJ, USA},
    publisher = {IEEE},
    month = jun,
    year = {2014}
}