"""Process for obtaining diff_x and diff_y metrics over entire dataset."""

import cv2
import numpy as np

import is_os_line as isos


def contour_drawn_to_contour_shape(image, is_os_line):
    """Convert contour from drawn over image to blank canvas for shape analysis.

    Parameters
    ----------
    image : numpy array
        Image from which the IS/OS line has been extracted.
    is_os_line : numpy array
        Extracted IS/OS line in the form of a contour.

    Returns
    -------
    image_copy : numpy array
        Mostly white image with contour shape highlighted in green.
    """
    image_copy = 255 + np.zeros_like(image)
    cv2.drawContours(
        image=image_copy,
        contours=is_os_line,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return image_copy


def remove_value(input_list, value):
    """Remove instances of `value` in the input list."""
    return [x for x in input_list if x != value]


def diff_x(image, tolerance):
    """Compute horizontal width of largest gap in IS/OS line."""
    # Unpack `tolerance` tuple for top-side, bottom-side, left-hand, and right-hand sides
    (tol_top, tol_bot, tol_left, tol_right) = tolerance

    # Measure largest gap in contour line belonging to 'image'
    diff_x = []
    num_rows, num_cols, _ = image.shape
    for i in range(num_rows - tol_top - tol_bot):
        current_gap = 0
        largest_gap = 0
        for j in range(num_cols - tol_left - tol_right):
            if all(image[i + tol_top, j + tol_left, 0] == [255, 255, 255]):
                current_gap += 1
                if current_gap > largest_gap:
                    largest_gap = current_gap
            else:
                current_gap = 0
        diff_x.append(largest_gap)

    return diff_x


def diff_y(image, tolerance):
    """Compute vertical width of IS/OS line at each column."""
    # Unpack `tolerance` tuple for top-side, bottom-side, left-hand, and right-hand sides
    (tol_top, tol_bot, tol_left, tol_right) = tolerance

    # Measure contour thickness in `image`
    diff_y = []
    num_rows, num_cols, _ = image.shape
    for j in range(num_cols - tol_left - tol_right):
        y_min = 0
        y_max = 0
        for i in range(num_rows - tol_top - tol_bot):
            if (
                image[i + tol_top, j + tol_left, 0] == 0
                and image[i + tol_top, j + tol_left, 1] == 255
            ):
                y_min = i + tol_top
                break
        for i in range(num_rows - y_min - tol_top - tol_bot):
            if (
                image[i + y_min, j + tol_left, 0] == 0
                and image[i + y_min + tol_top, j + tol_left, 1] == 255
            ):
                y_max = i + y_min + tol_top
        diff_y.append(y_max - y_min - 1 - tol_top)

    return diff_y


def get_metrics(images_normal, images_amd, images_mh, tolerance):
    """Compute the desired x and y coordinates for classification of image type.

    Parameters
    ----------
    images_normal : list of numpy arrays
        Retinal images labeled as normal.
    images_amd : list of numpy arrays
        Retinal images labeled as positive for AMD.
    images_mh : list of numpy arrays
        Retinal images labeled as positive for MH.
    tolerance : tuple of four items
        Tolerance values for top-side, bottom-side, left-hand, and right-hand sides.

    Returns
    -------
    x_y_pairs_normal : list of tuples
        Average largest gap in contour line, contour thickness in images_normal.
    x_y_pairs_amd : list of tuples
        Average largest gap in contour line, contour thickness in images_amd.
    x_y_pairs_mh : list of tuples
        Average largest gap in contour line, contour thickness in images_mh.
    """
    is_os_lines_normal, is_os_lines_amd, is_os_lines_mh = isos.line_extraction_process(
        images_normal, images_amd, images_mh
    )

    normal_copies = []
    amd_copies = []
    mh_copies = []
    for i in range(len(images_normal)):
        normal_copies.append(
            contour_drawn_to_contour_shape(images_normal[i], is_os_lines_normal[i])
        )
    for i in range(len(images_amd)):
        amd_copies.append(
            contour_drawn_to_contour_shape(images_amd[i], is_os_lines_amd[i])
        )
    for i in range(len(images_mh)):
        mh_copies.append(
            contour_drawn_to_contour_shape(images_mh[i], is_os_lines_mh[i])
        )

    avg_vwidths_normal = []
    avg_vwidths_amd = []
    avg_vwidths_mh = []
    for image in normal_copies:
        vwidth_normal = diff_y(image, tolerance)
        normal_nonzero = remove_value(vwidth_normal, 0)
        if len(normal_nonzero) == 0:
            avg_vwidths_normal.append(sum(vwidth_normal))
        else:
            avg_vwidths_normal.append(sum(vwidth_normal) / len(normal_nonzero))
    for image in amd_copies:
        vwidth_amd = diff_y(image, tolerance)
        amd_nonzero = remove_value(vwidth_amd, 0)
        if len(amd_nonzero) == 0:
            avg_vwidths_amd.append(sum(vwidth_amd))
        else:
            avg_vwidths_amd.append(sum(vwidth_amd) / len(amd_nonzero))
    for image in mh_copies:
        vwidth_mh = diff_y(image, tolerance)
        mh_nonzero = remove_value(vwidth_mh, 0)
        if len(mh_nonzero) == 0:
            avg_vwidths_mh.append(sum(vwidth_mh))
        else:
            avg_vwidths_mh.append(sum(vwidth_mh) / len(mh_nonzero))

    avg_hwidths_normal = []
    avg_hwidths_amd = []
    avg_hwidths_mh = []
    for image in normal_copies:
        hwidth_normal = diff_y(image, tolerance)
        hwidth_normal_nonmax = remove_value(hwidth_normal, max(hwidth_normal))
        if len(hwidth_normal_nonmax) == 0:
            avg_hwidths_normal.append(sum(hwidth_normal_nonmax))
        else:
            avg_hwidths_normal.append(
                sum(hwidth_normal_nonmax) / len(hwidth_normal_nonmax)
            )
    for image in amd_copies:
        hwidth_amd = diff_y(image, tolerance)
        hwidth_amd_nonmax = remove_value(hwidth_amd, max(hwidth_amd))
        if len(hwidth_amd_nonmax) == 0:
            avg_hwidths_amd.append(sum(hwidth_amd_nonmax))
        else:
            avg_hwidths_amd.append(sum(hwidth_amd_nonmax) / len(hwidth_amd_nonmax))
    for image in mh_copies:
        hwidth_mh = diff_y(image, tolerance)
        hwidth_mh_nonmax = remove_value(hwidth_mh, max(hwidth_mh))
        if len(hwidth_mh_nonmax) == 0:
            avg_hwidths_mh.append(sum(hwidth_mh_nonmax))
        else:
            avg_hwidths_mh.append(sum(hwidth_mh_nonmax) / len(hwidth_mh_nonmax))

    x_y_pairs_normal = []
    x_y_pairs_amd = []
    x_y_pairs_mh = []
    for i in range(len(images_normal)):
        x_y_pairs_normal.append((8 * avg_hwidths_normal[i], avg_vwidths_normal[i]))
    for i in range(len(images_amd)):
        x_y_pairs_amd.append((8 * avg_hwidths_amd[i], avg_vwidths_amd[i]))
    for i in range(len(images_mh)):
        x_y_pairs_mh.append((8 * avg_hwidths_mh[i], avg_vwidths_mh[i]))

    return x_y_pairs_normal, x_y_pairs_amd, x_y_pairs_mh
