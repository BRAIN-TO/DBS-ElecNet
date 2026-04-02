import numpy as np
from scipy.ndimage import laplace
from skimage.filters import frangi
from utils import get_line, fit_best_fit_line_RANSAC, fit_best_fit_line

from skimage import morphology
import nibabel as nib

############################## helper functions ##############################
def prepare_fitting(thresholded_image):
    # Split in half than fit
    half = thresholded_image.shape[0] // 2
    left = np.zeros_like(thresholded_image)
    right = np.zeros_like(thresholded_image)
    left[:half,:,:] = thresholded_image[:half,:,:]
    right[half:,:,:] = thresholded_image[half:,:,:]

    # Fit the line
    left_pts = np.transpose(np.nonzero(left))
    right_pts = np.transpose(np.nonzero(right))

    return left_pts, right_pts

##############################################################################

def elec_segment(opt):
    file_split = opt.input.rindex('/')
    nname = opt.input[file_split+1:].split('_skull')[0]
    save_folder = opt.input[:file_split]
    
    img = nib.load(opt.input)
    br_mask = nib.load(opt.br_mask).get_fdata()
    seg = nib.load(opt.synthseg).get_fdata()
    # get the csf and other parts of the brain as a mask
    csf = (seg != 24) & (seg != 4) & (seg != 43) & (seg != 44) & (seg != 5) & (seg != 15)
    csf = morphology.isotropic_erosion(csf, radius=3)
    # get the end of electrode as the bottom of VentralDC
    _,__,left_end = np.min(np.transpose(np.nonzero(seg == 28)), axis=0)
    _,__,right_end = np.min(np.transpose(np.nonzero(seg == 60)), axis=0)
    ends = (left_end+5, right_end+5)
    print("SALINE start -", nname)

    ######################################### Apply the Laplacian filter #####################################
    brain_mask = morphology.isotropic_erosion(br_mask, radius=20)
    # Apply the filter
    laplacian_filtered = laplace(img.get_fdata())
    pos = np.where(laplacian_filtered>0, laplacian_filtered, 0)
    neg = -np.where(laplacian_filtered<0, laplacian_filtered, 0)
    # normalize it
    max_pos = np.max(pos)
    max_neg = np.max(neg)
    normalized_pos = pos / max_pos
    normalized_neg = neg / max_neg

    # Threshold the normalized image
    thresholded_pos = normalized_pos > opt.laplacian_threshold
    thresholded_neg = normalized_neg > opt.laplacian_threshold
    thresholded_img = thresholded_pos | thresholded_neg
    thresholded_img = thresholded_img.astype(np.uint8) * brain_mask.astype(np.uint8)
    thresholded_img = morphology.isotropic_closing(thresholded_img, radius=3)
    thresholded_img = morphology.remove_small_objects(thresholded_img, 5)
    thresholded_img = morphology.isotropic_dilation(thresholded_img, radius=2)
    # Mask out the csf area
    lap = thresholded_img * csf

    # Save the Laplacian image
    if opt.save_intermediate:
        nifti = nib.Nifti1Image(lap, img.affine, img.header)
        nib.save(nifti, save_folder+'/'+nname+'-thr-lap.nii.gz')
    print("laplacian image done")

    ######################################### Apply the Frangi filter #####################################
    brain_mask = morphology.isotropic_erosion(br_mask, radius=2)
    # Apply the filter
    filtered = frangi(img.get_fdata())
    filtered = filtered * brain_mask.astype(np.uint8)
    min_val = np.min(filtered)
    max_val = np.max(filtered)
    normalized_filtered_frangi = (filtered - min_val) / (max_val - min_val)

    # Threshold the normalized image
    thresholded_image = normalized_filtered_frangi > opt.frangi_threshold
    thresholded_image = morphology.isotropic_closing(thresholded_image, radius=3)
    thresholded_image = morphology.remove_small_objects(thresholded_image, 5)
    thresholded_image = morphology.isotropic_closing(thresholded_image, radius=8)
    thresholded_image = morphology.isotropic_dilation(thresholded_image, radius=2)
    frangi_img = thresholded_image * csf
    # Save the thresholded image
    if opt.save_intermediate:
        fran = nib.Nifti1Image(frangi_img, img.affine, img.header)
        nib.save(fran, save_folder+'/'+nname+'_frangi.nii.gz')
    
    # find overlaps of frangi and laplacian
    thresholded_image = frangi_img & lap
    if opt.save_intermediate:
        nifti = nib.Nifti1Image(thresholded_image, img.affine, img.header)
        nib.save(nifti, save_folder+'/'+nname+'_comb.nii.gz')
    print("frangi image done")

    ######################################### Fit the line #####################################
    left_pts, right_pts = prepare_fitting(thresholded_image)

    if left_pts.shape[0] != 0 and right_pts.shape[0] != 0:
        _,__,left_max = np.max(left_pts, axis=0)
        _,__,right_max = np.max(right_pts, axis=0)
        _,__,left_min = np.min(left_pts, axis=0)
        _,__,right_min = np.min(right_pts, axis=0)
        left_range = left_max-left_min
        right_range = right_max-right_min
    
    if left_pts.shape[0] == 0 or right_pts.shape[0] == 0 or left_range < 10 or right_range < 10:
        print("ah oh, frangi only now. Lowering threshold...")
        # if one side is empty use the newly thresholded frangi instead:
        thresholded_image = normalized_filtered_frangi > opt.lower_frangi_threshold
        thresholded_image = thresholded_image * csf
        thresholded_image = morphology.isotropic_closing(thresholded_image, radius=3)
        frangi_img = morphology.remove_small_objects(thresholded_image, 5)
        if opt.save_intermediate:
            nifti = nib.Nifti1Image(frangi_img, img.affine, img.header)
            nib.save(nifti, save_folder+'/'+nname+'_frangi.nii.gz')
        # do the splitting and fit
        left_pts, right_pts = prepare_fitting(frangi_img)

        if left_pts.shape[0] == 0 or right_pts.shape[0] == 0:
            print("ah oh, frangi only didn't work. Trying laplacian only...")
            left_pts, right_pts = prepare_fitting(lap)
            if left_pts.shape[0] == 0 or right_pts.shape[0] == 0:
                print("OH NO, laplacian only still didn't work... SKIP")
                return

        _,left_y_max,__ = np.max(left_pts, axis=0)
        _,right_y_max,__ = np.max(right_pts, axis=0)
        _,left_y_min,__ = np.min(left_pts, axis=0)
        _,right_y_min,__ = np.min(right_pts, axis=0)

        if left_y_max-left_y_min > 30 or right_y_max-right_y_min > 30:
            # if the points contain obvious outliers, use RANSAC
            # print("RANSAC TIME")
            left_centroid, left_direction = fit_best_fit_line_RANSAC(left_pts)
            right_centroid, right_direction = fit_best_fit_line_RANSAC(right_pts)
        else:
            # if no obvious outliers normal line fitting works better
            # print("Normal line fitting")
            left_centroid, left_direction = fit_best_fit_line(left_pts)
            right_centroid, right_direction = fit_best_fit_line(right_pts)
    else:
        # print("Normal line fitting")
        left_centroid, left_direction = fit_best_fit_line(left_pts)
        right_centroid, right_direction = fit_best_fit_line(right_pts)

    final_result = get_line(ends, img.get_fdata().shape, left_centroid, left_direction, right_centroid, right_direction, br_mask)
    # Save the image
    nifti = nib.Nifti1Image(final_result, img.affine, img.header)
    nib.save(nifti, save_folder+'/'+nname+'_saline_elec.nii.gz')
    print("SALINE done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segmenting the electrode")
    # folder paths (required)
    parser.add_argument('--input', type=str, required=True, help='path to the input data')
    parser.add_argument('--br_mask', type=str, required=True, help='path to the corresponding brain mask')
    parser.add_argument('--synthseg', type=str, required=True, help='path to the corresponding synthseg segmentation')
    # other parameters
    parser.add_argument('--laplacian_threshold', type=float, default=0.21, help='the threshold value for laplacian filtered image')
    parser.add_argument('--frangi_threshold', type=float, default=0.25, help='the threshold value for frangi filtered image')
    parser.add_argument('--lower_frangi_threshold', type=float, default=0.2, help='the threshold value for frangi filtered image if the higher one failed')
    parser.add_argument('--expand_radius', type=int, default=6, help='the radius of the final dialation')
    parser.add_argument('--save_intermediate', action='store_true', help='save intermediate images (thresholded laplacian and frangi images; combined image) if specified')

    opt = parser.parse_args()    
    elec_segment(opt)
