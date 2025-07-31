from torch.utils.data.dataset import Dataset
from abc import ABC
import pandas as pd
import numpy as np
import os
from typing import Callable
from utils import get_predict_sites_list, read_pkl, get_reshaped_4D
import nibabel, json, gc

DATA_DIR = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"
BINARY_BRAIN_MASK = DATA_DIR+"mni_cerebrum-gm-mask_1.5mm.nii.gz"
SPLITS_DICT = DATA_DIR+"indices_tr_te_all_lrncurv_splits_DL.pkl"
BIOBDBSNIP_OG_DATA_DIR = "/neurospin/signatures/psysbox_analyses/202104_biobd-bsnip_cat12vbm_predict-dx/"

class BipolarDataset(ABC, Dataset):

    nb_split = 0

    def __init__(self, preproc: str='vbm', 
                 split: str='train',transforms: Callable[[np.ndarray], np.ndarray]=None,  site = "Baltimore", datasize_idx = None):
        

        self.site = site
        assert split in ['train', 'test'], "Unknown split: %s"%split
        assert self.site in get_predict_sites_list()
        self.preproc = preproc
        self.split = split
        self.transforms = transforms
        self.datasize_idx = datasize_idx
        
        if self.transforms : print("transforms applied")

        self.labels = None
        self.data = None
        self.nb_split = BipolarDataset.nb_split
        self.indir = os.getcwd()
        self.mask_filename = BINARY_BRAIN_MASK

        if self.datasize_idx is not None :
            self.df = self.load_images()
            self.Xim = self.df['Xim']
            self.y = self.df['y']
            self.dict_splits = self.df["dict_splits"]
            size = self.site+"-"+str(self.datasize_idx)
            self.trainsplit, self.testsplit = self.dict_splits[size] 
        
        if self.split == "test":
            flat_data = self.Xim[self.testsplit]
            self.labels = self.y[self.testsplit] 
        
        if self.split == "train":
            flat_data = self.Xim[self.trainsplit]
            self.labels = self.y[self.trainsplit]

            
        assert len(self.labels)==len(flat_data),"There aren't as many labels as there are images"

        data = get_reshaped_4D(flat_data, self.mask_filename)
        self.data = np.reshape(data, (data.shape[0], 1, *data.shape[1:])) # reshapes to (nbsubjects, 1, 3D image shape)

        assert len(self.labels)==len(self.data)
        assert self.labels is not None, "labels are missing"
        assert self.data is not None, "data is missing"
        
        self.shape = np.shape(self.data)
        gc.collect()
    
    def __getitem__(self, idx: int):
        sample, target = self.data[idx], self.labels[idx]

        if self.transforms:
            sample = self.transforms(sample)

        return sample, target.astype(np.float32), idx 
    
    def __len__(self):
        return len(self.labels)
    
    def get_nb_split(self):
        return self.nb_split
    
    def reset_nb_split(self):
        BipolarDataset.nb_split = 0
    

    def __str__(self):
        return "%s-%s-%s"%(type(self).__name__, self.preproc, self.split)

    def load_images(self):
        # Mask
        mask_img = nibabel.load(BINARY_BRAIN_MASK)
        mask_arr = mask_img.get_fdata() != 0
        assert np.sum(mask_arr != 0) == 331695

        participants_filename = os.path.join(BIOBDBSNIP_OG_DATA_DIR, "biobd-bsnip_cat12vbm_participants.csv")
        imgs_flat_filename = os.path.join(BIOBDBSNIP_OG_DATA_DIR, "biobd-bsnip_cat12vbm_mwp1-gs-flat.npy")
        participants = pd.read_csv(participants_filename) 

        # load images
        Xim = np.load(imgs_flat_filename, mmap_mode='r')

        assert Xim.shape[1] == np.sum(mask_arr != 0)
        msk = np.ones(participants.shape[0]).astype(bool)
        y = participants["dx"][msk].values

        dict_=read_pkl(SPLITS_DICT)

        dataset = dict(Xim=Xim, y=y,dict_splits=dict_)
        gc.collect()

        return dataset
    