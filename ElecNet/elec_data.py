import nibabel as nib
import torchio as tio

import torch
from torch.utils.data import Dataset

import numpy as np
from skimage.transform import resize

INPUT_SIZE = 128

############################## helper functions ##############################
def resize_to(img, target_length):
    img_shape = img.shape
    target_size = (np.array(img_shape) * (target_length / max(img_shape))).astype(int)
    return resize(img, output_shape=target_size, anti_aliasing=False, preserve_range=True)

def img_preprocess(img, pad_to=INPUT_SIZE, is_mask=False):
    """
    Resizes a 3D image to fit within a (pad_to, pad_to, pad_to) cube, pads with zeros if necessary,
    and normalizes the pixel values to the range [-1, 1].

    Args:
        img (torch.Tensor or np.ndarray): 3D input of shape (D, H, W).
        pad_to (int): Target size for each dimension (default is 256).

    Returns:
        np.ndarray: Padded and normalized 3D image of shape (pad_to, pad_to, pad_to).
    """
    # if the difference is small, just crop; otherwise do resizing
    crop_slices = []
    if 0 < (max(img.shape) - pad_to) < 10:
        for dim_size in img.shape:
            if dim_size > pad_to:
                start = (dim_size - pad_to) // 2
                end = start + pad_to
                crop_slices.append(slice(start, end))
        else:
            crop_slices.append(slice(0, dim_size))  # keep full dimension if smaller
        img = img[crop_slices[0], crop_slices[1], crop_slices[2]]
    
    # padding
    d, h, w = img.shape
    max_dim = max(img.shape)
    pad_d = max_dim - d
    pad_h = max_dim - h
    pad_w = max_dim - w
    if any(v > 0 for v in [pad_d, pad_h, pad_w]):
        padded_img = np.pad(img, 
                            pad_width=((0, pad_d), (0, pad_h), (0, pad_w)), 
                            mode='constant', 
                            constant_values=0)  # Pad with zeros

    # resize according to longest side
    if max_dim != pad_to:
        padded_img = resize_to(padded_img, pad_to)
    
    if not is_mask:
        # normalize
        padded_img = (padded_img - padded_img.mean()) / padded_img.std()
    return padded_img

def undo_normalization_and_pad(img, ori_shape):
    if 0 < (max(ori_shape) - INPUT_SIZE) < 10:
        cut_shape = np.where(np.array(ori_shape) < INPUT_SIZE, ori_shape, INPUT_SIZE).astype(int)
        img = img[:cut_shape[0], :cut_shape[1], :cut_shape[2]]
        pad_width = []
        for dim_size, target_size in zip(img.shape, ori_shape):
            diff = target_size - dim_size
            before = diff // 2
            after = diff - before
            pad_width.append((before, after))
        img = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        # resize to original
        img = resize_to(img, max(ori_shape))
        # undo padding by cropping
        img = img[:ori_shape[0], :ori_shape[1], :ori_shape[2]]
    return img
