# DBS-ElecNet
### A method for automated localization and segmentation of DBS electrodes in clinical MRI
Vanessa H. Yu, Edward Chen, Jürgen Germann, Alexandre Boutet, Andres M. Lozano, Kâmil Uludağ, and Sriranga Kashyap

*Accepted at the 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI), London, UK*

Poster available at: [https://zenodo.org/records/19430997](https://zenodo.org/records/19430997)

## Table of Contents
<!-- 1. [Data Preprocessing](#data-preprocessing) -->
1. [Requirements](#requirements)
2. [SALINE Electrode Segmentation](#SALINE-electrode-segmentation)
3. [DBS-ElecNet Inference](#DBS-ElecNet-Inference)

## Requirements

### Python

- Python 3.9+
- Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### External command-line tools

The `run_saline` and `run_elecnet` scripts call external neuroimaging tools that must be available on your `PATH`:

- **ANTs** (`ResampleImage`, `N4BiasFieldCorrection`)
- **FreeSurfer / SynthStrip / SynthSeg** (`mri_synthstrip`, `mri_synthseg`)

## SALINE electrode segmentation:

This pipeline performs single- or dual-electrode localization using the SALINE framework.

**Command:**
```bash
./run_saline <nifti_file> <num_elec>
```
**Required arguments:**

* `nifti_file`: Path to the input MRI volume (NIfTI format).

* `num_elec`: Number of implanted electrodes (1 or 2).

**Output:**

* `subject_saline_elec.nii.gz`: Electrode segmentation in the native image space.

* `subject_1mm_iso.nii.gz`: Input MRI resampled to 1 mm isotropic resolution.

* `subject_1mm_iso_saline_elec.nii.gz`: Electrode segmentation in 1 mm isotropic space.


## DBS-ElecNet Inference:

This pipeline performs electrode segmentation using DBS-ElecNet.

**Command:**
```bash 
./run_elecnet <nifti_file> <device>
```
**Required arguments:**

* `nifti_file`: Path to the input MRI volume (NIfTI format).

* `device`: Inference device, following PyTorch conventions (cpu, cuda, cuda:0, etc.).

**Output:**

* `subject_elec.nii.gz`: Electrode segmentation in the native image space.

* `subject_1mm_iso.nii.gz`: Input MRI resampled to 1 mm isotropic resolution.

* `subject_1mm_iso_elec.nii.gz`: Electrode segmentation in 1 mm isotropic space.

## Note:

- All preprocessing steps are performed automatically, including resampling to 1 mm isotropic resolution, skull stripping, N4 bias field correction, and (for SALINE) SynthSeg.  
- Intermediate files are removed upon completion **except for the 1 mm isotropic resampled MRI** (`subject_1mm_iso.nii.gz`), which is retained for reference.  
- Final outputs are provided in **both 1 mm isotropic space and the original native image space**.
