import torch
from torch.utils.data import DataLoader
import os, glob, time

import nibabel as nib
from tqdm import tqdm
from elec_unet import UNet3D
from elec_data import img_preprocess, undo_normalization_and_pad

BATCH_SIZE = 5

def load_data(input_file):
    """
    Load and preprocess the input NIfTI image.

    Returns
    -------
    post : torch.Tensor
        Preprocessed image of shape (1, 1, D, H, W).
    affine : np.ndarray
        Affine matrix from the original image.
    ori_size : tuple
        Spatial shape of the original image.
    """
    img = nib.load(input_file)
    post = img_preprocess(img.get_fdata())
    post = torch.tensor(post).float().unsqueeze(0).unsqueeze(0)
    return post, img.affine, img.get_fdata().shape

def inference(model, input_file, device, threshold=0.5):
    model.eval()
    model.to(device)

    # load data
    file_split = input_file.rindex('/')
    nname = input_file[file_split+1:].split('_skull')[0]
    save_folder = input_file[:file_split]
    post, img_affine, ori_size = load_data(input_file)

    with torch.no_grad():
        imgs = post.to(device)

        logits = model(imgs)
        outputs = torch.sigmoid(logits)

        # Save the end result as nifti
        probs = outputs.cpu()
        np_probs = probs[0].squeeze(0).numpy()
        # undo padding and resizing
        np_probs = undo_normalization_and_pad(np_probs, ori_size)
        pred_mask = (np_probs > threshold).astype(float)  # binary mask
        nifti_img = nib.Nifti1Image(pred_mask, img_affine)
        nib.save(nifti_img, save_folder+'/'+nname+'_elec.nii.gz')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segmenting the electrode")
    # folder paths (required)
    parser.add_argument('--input', type=str, required=True, help='path to the input data')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to the saved model checkpoint')
    parser.add_argument('--device', type=str, required=True, help='cpu or cuda')

    opt = parser.parse_args()

    # load model
    model = UNet3D(in_channels=1, out_channels=1, base_filters=32)
    ckpt = torch.load(opt.ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt)

    # prediction
    inference(model, opt.input, opt.device)
