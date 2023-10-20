"""Functions for image pre-processing in identification_of_amd notebook."""

import os

import cv2
import numpy as np


def load_images(folder_name):
    """Read in image data given a specific folder.

    Parameters
    ----------
    folder_name : string
        Path to folder containing desired images.

    Returns
    -------
    images : list of numpy arrays
        Desired images stacked into a list.
    """
    images = []
    for file_name in os.listdir(folder_name):
        img = cv2.imread(os.path.join(folder_name, file_name)).astype(np.uint8)
        if img.shape != (750, 500, 3):
            img = cv2.resize(img, (750, 500))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)

    return images


def remove_background(img):
    """Remove dark background from an image file.

    Parameters
    ----------
    img : numpy array
        Image containing dark background to be removed.

    Returns
    -------
    shape_on_white : numpy array
        Desired image with foreground shape only.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    retval, thresh = cv2.threshold(gray_img, 127, 255, 0)
    img_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, img_contours, -1, (255, 255, 255))
    _, shape_on_white = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    return shape_on_white
