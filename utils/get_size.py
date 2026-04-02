import nibabel as nib

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segmenting the electrode")
    # file path (required)
    parser.add_argument('--input', type=str, required=True, help='path to the input data')
    
    opt = parser.parse_args()

    img = nib.load(opt.input).get_fdata()
    dim1, dim2, dim3 = img.shape
    print(f'{dim1}x{dim2}x{dim3}')  # (dim1, dim2, dim3)