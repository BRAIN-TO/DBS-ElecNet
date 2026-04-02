import numpy as np
from skimage import morphology
from sklearn.linear_model import RANSACRegressor


def get_single_line(end, img_shape, centroid, direction):
    """
    Generate a binary mask image indicating the locations of one predicted lines
    within the 3D image volume, and dilate it.

    Args:
        end: defining the slice range of the segment of interest.
        img_shape: Shape of the 3D image (D, H, W), used to ensure points lie within bounds.
        centroid: A 3-element array specifying a point on the line.
        direction: A 3-element array specifying the direction vector of the line.

    Returns:
        A binary mask with ones along both lines (with dilation of radius 6) and zeros elsewhere.
    """

    # locate all intersecting cells
    step = 0.1
    z = np.arange(end, img_shape[-1], step)

    # the k here is the scalar in the line definition equation (y = point + k * direction)
    k = (z - centroid[-1]) / direction[-1]
    elec = np.floor(centroid + np.outer(k, direction)).astype(int)

    # pick out the valid ones
    cond = (elec[:, 0] > 0) & (elec[:, 0] < img_shape[0]) & \
           (elec[:, 1] > 0) & (elec[:, 1] < img_shape[1]) & \
           (elec[:, 2] > 0) & (elec[:, 2] < img_shape[2])

    # create the line image
    final_result = np.zeros(img_shape)
    final_result[tuple(elec[cond].T)] = 1

    return final_result

def get_line(ends, img_shape, left_centroid, left_direction, right_centroid, right_direction, br_mask, radius=6):
    """
    Generate a binary mask image indicating the locations of two predicted lines (left and right)
    within the 3D image volume, and dilate it.

    Args:
        ends: defining the slice range of the segment of interest.
        img_shape: Shape of the 3D image (D, H, W), used to ensure points lie within bounds.
        left_centroid: A 3-element array specifying a point on the left line.
        left_direction: A 3-element array specifying the direction vector of the left line.
        right_centroid: A 3-element array specifying a point on the right line.
        right_direction: A 3-element array specifying the direction vector of the right line.

    Returns:
        A binary mask with ones along both lines (with dilation of radius 6) and zeros elsewhere.
    """

    # locate all intersecting cells for both left and right
    final_result = get_single_line(ends[0], img_shape, left_centroid, left_direction)
    final_result = final_result + get_single_line(ends[1], img_shape, right_centroid, right_direction)
    final_result = final_result * br_mask
    final_result = morphology.isotropic_dilation(final_result, radius=radius)

    return final_result

def fit_best_fit_line_RANSAC(points):
    X = points[:, :2]  # Use the first two coordinates (x, y) as features
    y = points[:, 2]   # Use the third coordinate (z) as the target
    
    # Create a linear model with RANSAC
    ransac = RANSACRegressor(min_samples=2, random_state=42, loss='absolute_error')
    ransac.fit(X, y)
    
    # Get the inliers
    inlier_mask = ransac.inlier_mask_
    inlier_points = points[inlier_mask]
    
    # Fit the best fit line to the inliers using SVD
    centroid = np.mean(inlier_points, axis=0)
    centered_points = inlier_points - centroid
    U, S, Vt = np.linalg.svd(centered_points)
    direction = Vt[0]
    
    return centroid, direction


def fit_best_fit_line(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    U, S, Vt = np.linalg.svd(centered_points)
    direction = Vt[0]
    
    return centroid, direction