"""Function for extracting IS/OS line in identification_of_amd notebook."""

import cv2


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
