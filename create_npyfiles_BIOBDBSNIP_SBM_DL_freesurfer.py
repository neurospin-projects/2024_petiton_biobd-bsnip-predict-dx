import os
from glob import glob
from tqdm import tqdm # optional

import numpy as np
import pandas as pd

from utils import read_pkl, get_predict_sites_list

# inputs
BIOBD_PATH = "/neurospin/psy/biobd/derivatives/freesurfer_v7.1.1/ses-V1"
BSNIP_PATH = "/neurospin/psy/bsnip1/derivatives/freesurfer_v7.1.1"
BIOBDBSNIP_OG_DATA_DIR = "/neurospin/signatures/psysbox_analyses/202104_biobd-bsnip_cat12vbm_predict-dx/"
DATA_DIR = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"
SPLITS_DICT = DATA_DIR+"indices_tr_te_all_lrncurv_splits_DL.pkl"

# outputs
LH_DATA = DATA_DIR+"surface-lh_data.npy"
RH_DATA = DATA_DIR+"surface-rh_data.npy"
META_DATA = DATA_DIR+"metadata.tsv"

def get_participants_loso():
    participants_filename = os.path.join(BIOBDBSNIP_OG_DATA_DIR, "biobd-bsnip_cat12vbm_participants.csv")
    participants = pd.read_csv(participants_filename) 

    dict_=read_pkl(SPLITS_DICT)
    all_participants_LOSO_max_train_size = []
    for site in get_predict_sites_list():
        participant_ids_one_test_site = participants.iloc[dict_[site+"-8"][1]]["participant_id"].values
        all_participants_LOSO_max_train_size.append(list(participant_ids_one_test_site))

    all_participants_LOSO_max_train_size=np.concatenate(all_participants_LOSO_max_train_size,axis=0)
    assert len(all_participants_LOSO_max_train_size) == 861
    return all_participants_LOSO_max_train_size

def get_subjects(fs_path):
    """Get the ordered list of subjects whose freesurfer output folder exists"""
    regex = os.path.join(fs_path, 'sub-*')
    sub_list = []
    for sub_path in tqdm(glob(regex)):
        sub_id = sub_path.split('/')[-1]
        sub_list.append(sub_id)
    sub_list.sort()
    return sub_list

def check_channels_are_the_same(biobd_subs, bsnip_subs):
    reference_data = None
    all_subs = [(BIOBD_PATH, sub) for sub in biobd_subs] + [(BSNIP_PATH, sub) for sub in bsnip_subs]

    for i, (base_path, sub) in enumerate(all_subs):
        if i<len(biobd_subs): filepath = os.path.join(base_path, f"{sub}", "channels.txt")
        else : filepath = os.path.join(base_path, f"{sub}", "ses-V1/channels.txt")
        assert os.path.exists(filepath), f"Path doesn't exist: {filepath}"
        dat = np.loadtxt(filepath, dtype=str)
        if i ==0: print("channels are :",dat)

        if reference_data is None:
            reference_data = dat
        else:
            if not np.array_equal(dat, reference_data):
                raise ValueError(f"channels.txt differs for sub-{sub} at {filepath}")

    print("✅ All channels.txt files are identical.")

def concat_npy(dataset_name, biobd_subs, bsnip_subs):
    # set up variables based on the chosen dataset
    if dataset_name == 'biobd':
        main_path = os.path.join(BIOBD_PATH, 'sub-*', 'xhemi-textures.npy')
        subs = biobd_subs
    elif dataset_name == 'bsnip1':
        main_path = os.path.join(BSNIP_PATH, 'sub-*', 'ses-V1', 'xhemi-textures.npy')
        subs = bsnip_subs
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    lh_data = []
    rh_data = []

    for sub in tqdm(subs):
        # read the metrics
        sub_npy = main_path.replace('sub-*',sub)
        sub_npy = np.load(sub_npy)

        # separate them by hemisphere*
        lh_sub = sub_npy[0,:4,:]
        # we keep the thickness, curvature, and sulcal morphology measures but leave out the area meaqures at row index 2 of the array
        lh_sub = lh_sub[[0, 1, 3], :] 

        rh_sub = sub_npy[0,4:,:]
        rh_sub = rh_sub[[0, 1, 3], :] 

        lh_data.append(lh_sub)
        rh_data.append(rh_sub)

    # concat
    lh_data = np.stack(lh_data, axis=0)
    rh_data = np.stack(rh_data, axis=0)
    print(lh_data.shape, rh_data.shape)
    # put them in the format asked by Corentin
    lh_data = np.transpose(lh_data, axes=[0,2,1])
    rh_data = np.transpose(rh_data, axes=[0,2,1])
    print(lh_data.shape, rh_data.shape)

    return lh_data, rh_data

def save_npy_and_metadata_files(save_npy=False, save_metadata=False):
    biobd_subs = get_subjects(BIOBD_PATH)
    bsnip_subs = get_subjects(BSNIP_PATH)
    listbiobd_bsnip_subs=biobd_subs+bsnip_subs
    all_participants_LOSO_max_train_size = get_participants_loso()

    all_participants_LOSO_max_train_size= ["sub-"+s for s in all_participants_LOSO_max_train_size]

    subjects_not_in_list=[sub for sub in all_participants_LOSO_max_train_size if sub not in listbiobd_bsnip_subs]
    assert len(all_participants_LOSO_max_train_size) - len(subjects_not_in_list)==763

    to_keep = [sub for sub in all_participants_LOSO_max_train_size if sub in listbiobd_bsnip_subs]
    assert len(to_keep)==763

    biobd_subs = [sub for sub in biobd_subs if sub in to_keep]
    bsnip_subs = [sub for sub in bsnip_subs if sub in to_keep]
    print(len(biobd_subs), len(bsnip_subs))

    check_channels_are_the_same(biobd_subs, bsnip_subs)

    biobd_lh, biobd_rh = concat_npy('biobd', biobd_subs, bsnip_subs)
    bsnip_lh, bsnip_rh = concat_npy('bsnip1', biobd_subs, bsnip_subs)

    biobd_bsnip_lh = np.concatenate([biobd_lh, bsnip_lh], axis=0)
    biobd_bsnip_rh = np.concatenate([biobd_rh, bsnip_rh], axis=0)
    biobd_bsnip_subs = biobd_subs + bsnip_subs

    # print(biobd_bsnip_lh.shape, biobd_bsnip_rh.shape)
    print(len(biobd_bsnip_subs), biobd_bsnip_subs[:5], biobd_bsnip_subs[-5:])

    # save npy files for left and right hemispheres
    if save_npy:
        np.save(LH_DATA, biobd_bsnip_lh)
        np.save(RH_DATA, biobd_bsnip_rh)

    participants_filename = os.path.join(BIOBDBSNIP_OG_DATA_DIR, "biobd-bsnip_cat12vbm_participants.csv")
    participants = pd.read_csv(participants_filename) 
    biobd_bsnip_subs = [s[4:] if s.startswith('sub-') else s for s in biobd_bsnip_subs]

    participants_subjectsLOSO = participants[participants["participant_id"].isin(biobd_bsnip_subs)]
    # reorder participants_subjectsLOSO to match the order of participant_ids in `biobd_bsnip_subs`
    ordered_df = participants_subjectsLOSO.set_index("participant_id").loc[biobd_bsnip_subs].reset_index()
    metadata_df = ordered_df[["participant_id", "age", "sex", "site","dx"]]
    metadata_df["sex"] = metadata_df["sex"].replace({0: "male", 1: "female"})

    # save metadata file
    if save_metadata : metadata_df.to_csv(META_DATA, sep='\t', index=False)

def main():
    save_npy_and_metadata_files()

if __name__ == "__main__":
    main()
