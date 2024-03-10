import json
import nibabel as nb 
import numpy as np
import pandas as pd


def convert_2d(mask, nifti):
    """
    Convert 4d nifti to 2d dataframe using mask
    """
    nonzero_indx = np.nonzero(mask)
    nifti_2d = nifti[nonzero_indx]
    return nifti_2d.T


def convert_4d(mask, nifti_data):
    """
    Convert 2d dataframe back to 4d nifti format using mask
    """
    nifti_4d = np.zeros(mask.shape + (nifti_data.shape[0],), 
                        dtype=nifti_data.dtype)
    nifti_4d[mask, :] = nifti_data.T
    return nifti_4d


def get_subj_fp(dataset, subj_c=None):
    """ 
    Given dataset (primary, replication, nki) and subj label - pull filepaths for
    functional scans and co2 signals (for each run), as well as subject
    mask
    """
    func_list = []
    physio_list = []
    if dataset == 'nki':
        mask = f'data/nki/nki_mni_brain_mask.nii.gz'
        subjects = pd.read_csv('data/nki/nki_subjects.csv')
        for s in subjects.subject:
            subj = f'sub-{s}'
            func = f'data/nki/func/raw/{subj}_ses-NFB3_task-MSIT_bold_st_mt_norm_proc.nii.gz'
            physio = f'data/nki/physio/raw/{subj}_ses-NFB3_task-MSIT_physio.tsv.gz'
            physio_json = f'data/nki/physio/raw/{subj}_ses-NFB3_task-MSIT_physio.json'
            func_list.append(func)
            physio_list.append((physio, physio_json))


    elif dataset in ['primary', 'replication']:
        runs = ['1', '2', '3']
        mask = f'data/cardiff/sub{subj_c}_mask.nii'
        for r in runs:
            if dataset == 'primary':
                func = f'data/cardiff/{dataset}/sub{subj_c}.3backvis.gas.scan{r}.std.5mm.hp.z.nii.gz'
                physio = f'data//cardiff/{dataset}/CO2_data/s{subj_c}.3back.gas.scan{r}_CO2_HRFconv.txt'
            elif dataset == 'replication':
                func = f'data/cardiff/{dataset}/sub{subj_c}.3backvis.nogas.scan{r}.std.5mm.hp.z.nii.gz'
                physio = f'data/cardiff/{dataset}/CO2_data/s{subj_c}.3back.nogas.scan{r}_CO2_HRFconv.txt'
            func_list.append(func)
            physio_list.append(physio)

    return func_list, physio_list, mask


def load_data(subj, dataset):
    f_list, p_list, mask_fp = get_subj_fp(dataset, subj)

    # Load mask
    mask = nb.load(mask_fp)
    mask_bin = mask.get_fdata() > 0

    # load functional data
    func_data = []
    physio_data = []
    for func_fp, physio_fp in zip(f_list, p_list):
        func_data.append(load_subject_func(func_fp, mask_bin))
        physio_data.append(load_subject_physio(physio_fp, dataset))

    return func_data, physio_data, mask


def load_subject_func(fp, mask):
    # Load scan
    nifti = nb.load(fp)
    nifti_data = nifti.get_fdata()
    nifti_data = convert_2d(mask, nifti_data)
    return nifti_data


def load_subject_physio(fp, dataset):
    # Load scan
    if dataset == 'nki':
        physio_data = pd.read_csv(fp[0], delimiter='\t', compression='gzip')
        physio_json = json.load(open(fp[1], 'rb'))
        physio_data.columns = physio_json['Columns']
    elif dataset in ['primary', 'replication']:
        physio_data = np.loadtxt(fp)
    return physio_data


def write_nifti(data, output_file, mask):
    """
    Given data in 2d format, write out to 4d nifti file using mask
    """
    mask_bin = mask.get_fdata() > 0
    nifti_4d = np.zeros(mask.shape + (data.shape[0],), 
                        dtype=data.dtype)
    nifti_4d[mask_bin, :] = data.T

    nifti_out = nb.Nifti2Image(nifti_4d, mask.affine)
    nb.save(nifti_out, output_file)
