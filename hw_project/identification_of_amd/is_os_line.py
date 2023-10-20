"""Function for extracting IS/OS line in identification_of_amd notebook."""

import cv2

import preprocessing as pre


def extract_is_os_line(oct_img):
    """Extract IS/OS line from a given retinal image.

    Parameters
    ----------
    oct_img : numpy array
        Image containing retinal scan.

    Returns
    -------
    max_contour : numpy array
        Image containing the extracted IS/OS line as a contour.
    """
    # Convert `oct_img` to grayscale
    gray_img = cv2.cvtColor(oct_img, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray_img, 0, 125, cv2.THRESH_BINARY)

    # Finding contours within retina or photoreceptor
    contours, _ = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )

    # Loop through contours and identify contour with the largest area as the IS/OS line
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    return max_contour


def line_extraction_process(images_normal, images_amd, images_mh):
    """Extract IS/OS lines from a dataset of retinal images.

    Parameters
    ----------
    images_normal : list of numpy arrays
        Retinal images labeled as normal.
    images_amd : list of numpy arrays
        Retinal images labeled as positive for AMD.
    images_mh : list of numpy arrays
        Retinal images labeled as positive for MH.

    Returns
    -------
    is_os_lines_normal : list of numpy arrays
        Images containing the extracted IS/OS line in contour form.
    is_os_lines_amd : list of numpy arrays
        Images containing the extracted IS/OS line in contour form.
    is_os_lines_mh : list of numpy arrays
        Images containing the extracted IS/OS line in contour form.
    """
    is_os_lines_normal = []
    for i in range(len(images_normal)):
        img = pre.remove_background(images_normal[i])
        is_os_line = extract_is_os_line(img)
        is_os_lines_normal.append(is_os_line)

    is_os_lines_amd = []
    for i in range(len(images_amd)):
        img = pre.remove_background(images_amd[i])
        is_os_line = extract_is_os_line(img)
        is_os_lines_amd.append(is_os_line)

    is_os_lines_mh = []
    for i in range(len(images_mh)):
        img = pre.remove_background(images_mh[i])
        is_os_line = extract_is_os_line(img)
        is_os_lines_mh.append(is_os_line)

    return is_os_lines_normal, is_os_lines_amd, is_os_lines_mh
