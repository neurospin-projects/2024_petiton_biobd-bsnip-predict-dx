
import pandas as pd
import numpy as np
import sys, re, os, json
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler

# sbm data
from deep_learning_sbm.data_transforms import create_transforms_for_biobdbsnip
from deep_learning_sbm.datasets_biobdbsnip import ClassificationDataset
from surfify.utils import icosahedron, interpolate_data

# plotting
import matplotlib.pyplot as plt
from nilearn import plotting
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib

# vbm plotting
from nilearn.image import new_img_like


# sbm plotting
from nilearn import datasets
import nibabel.freesurfer.io as fsio

from utils import read_pkl, save_pkl, get_reshaped_4D, remove_zeros, get_predict_sites_list

# VBM voxelwise data
from torchvision.transforms.transforms import Compose
from deep_learning_vbm.transforms import Crop, Padding, Normalize
from deep_learning_vbm.BD_dataset import BipolarDataset

sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer

VBMLOOKUP_FILE = "/drf/local/spm12/tpm/labels_Neuromorphometrics.xml"
VOL_FILE_VBM = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
DATA_DIR=ROOT+"data/processed/"
BINARY_BRAIN_MASK = DATA_DIR+"mni_cerebrum-gm-mask_1.5mm.nii.gz"
ROI_VBM_FILE = DATA_DIR+"VBM_ROI_allfolds_280725.csv"
ROI_SBM_FILE_DESTRIEUX = DATA_DIR+"SBM_ROI_Destrieux_allfolds_280725.csv"
ATLAS_DF = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=';')
BIOBDBSNIP_OG_DATA_DIR = "/neurospin/signatures/psysbox_analyses/202104_biobd-bsnip_cat12vbm_predict-dx/"
LEARNING_CRV_TL_DIR = ROOT+"models/VBM_TL/TL_learningcurve/"
PATH_DICT_5DE_TL_SBM=ROOT+"reports/means_TL_5DE_ypred_train_and_test_N8.pkl"
SPLITS_DICT_SBM_VERTEXWISE="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/dict_splits_test_all_trainset_sizes_N763.pkl"
MODALITIES_SBM_VERTEXWISE = ["surface-lh_data", "surface-rh_data"]
DATA_DIR_SBM_VERTEXWISE= "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/"
ALL_SUB_ROIS_SBM = ['Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', \
                             'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'Right-Thalamus',\
                                  'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', \
                                    'Right-Amygdala', 'Right-Accumbens-area', 'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent',
                              '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-VentralDC', 'Left-vessel',
                               'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-VentralDC',
                                 'Right-vessel', 'Right-choroid-plexus', '5th-Ventricle', 'Optic-Chiasm', 'CC_Posterior', 'CC_Mid_Posterior', 
                                 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior']

PATH_SBM_DESTRIEUX="/drf/local/freesurfer-7.1.1/subjects/fsaverage_sym/label/"


# outputs
IMAGE_VOXELWISE_VBM_FWDMAP_PATH = ROOT+"reports/forward_maps_VBM_voxelwise.nii"
PATH_FWDMAP_VOXELWISE_PER_ROI_NEUROMORPHOMETRICS=ROOT+"reports/dict_cov_by_roi_neuromorphometrics_for_voxelwise_fwd_map.json"
IMG_PATH_FWD_MAP_PNG_VBM=ROOT+"reports/forwardmaps_images/fwd_map_vbm_voxelwise.png"
COVARIANCES_VBM_VOXELWISE = ROOT+'reports/covariances_fwd_model_VBM_voxelwise.npy'
COVARIANCES_SBM_VERTEXWISE = ROOT+'reports/covariances_fwd_model_SBM_vertexwise.npy'
COVARIANCES_SBM_VERTEXWISE_RESHAPED_LH = ROOT+'reports/covariances_fwd_model_SBM_vertexwise_reshaped_lh.npy'
COVARIANCES_SBM_VERTEXWISE_RESHAPED_RH = ROOT+'reports/covariances_fwd_model_SBM_vertexwise_reshaped_rh.npy'
IMG_PATH_FWD_MAP_PNG_SBM=ROOT+"reports/forwardmaps_images/"

def compute_covariance(Xtest_, scores_test, verbose=False):
    # Number of samples (n) and features (m)
    n, m = Xtest_.shape
    if verbose: print("n,m",n,m)
    
    # Mean of the scores_test
    mean_scores = np.mean(scores_test)
    
    # Mean of each feature in Xtest
    mean_Xtest = np.mean(Xtest_, axis=0)
    if verbose: print("mean of each column/voxel of Xtest : ",np.shape(mean_Xtest))
    
    # Center the features and scores
    X_centered = Xtest_ - mean_Xtest
    scores_centered = scores_test - mean_scores

    if verbose: print("X_centered",np.shape(X_centered), type(X_centered))

    if verbose: print("np.shape X_centered.T",np.shape(X_centered.T), type(X_centered.T))
    if verbose: print("scores_centered",np.shape(scores_centered), type(scores_centered))
    
    # Compute covariance between each feature and scores_test
    covariance = (X_centered.T@scores_centered) / (n-1) # n-1 pour prendre en compte les degrés de liberté de l'échantillon vs. la pop totale
    return covariance

def get_vbm_roi():
    """
    retrieve the df of VBM ROIs and the corresponding list of ROI names
    """
    dfroi = pd.read_csv(ROI_VBM_FILE)
    dfroi = remove_zeros(dfroi) # remove rois with values equal to zeros across all participants
    list_rois = [r for r in dfroi.columns if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]
    assert len(list_rois)==280, "there should be 280 rois"
    return dfroi, list_rois

def get_sbm_roi():
    """
    retrieve the df of SBM ROIs and the corresponding list of ROI names
    """
    dfroi = pd.read_csv(ROI_SBM_FILE_DESTRIEUX)
    list_rois = list(dfroi.columns)
    list_rois = [r for r in list_rois if r.endswith("_area") or r.endswith("_thickness") or r in ALL_SUB_ROIS_SBM]
    assert len(list_rois) == 296 + 34
    return dfroi, list_rois

def apply_transforms(dataset, type_feature="vbm"):
    """
    applies transforms that were applied to VBM voxelwise images before being fed to the classifier
    """
    assert type_feature in ["vbm","sbm"]

    all_transformed_images = []
    for i in range(len(dataset)):
        sample, target, idx = dataset[i]  # This applies transforms
        if type_feature=="sbm":
            lh_data, rh_data = sample["surface-lh_data"], sample["surface-rh_data"]
            lh_data, rh_data = np.array(lh_data), np.array(rh_data)
            sample = np.concatenate((lh_data, rh_data),axis=1)
        all_transformed_images.append(sample)
    all_transformed_images = np.array(all_transformed_images)
    return all_transformed_images

def interpolate_data_multimod(data, by=1, up_indices=None):
    n_samples, n_vertices, n_features = data.shape
    upsampled = []

    for f in range(n_features):
        print(f, " /n_features")
        data_f = data[:, :, f:f+1]  # keep 3D (n_samples, n_vertices, 1)
        up_f = interpolate_data(data_f, by=by, up_indices=up_indices)
        upsampled.append(up_f)

    # stack features back
    upsampled = np.concatenate(upsampled, axis=2)  # shape: (n_samples, n_new_vertices, n_features)
    return upsampled


def reverse_initial_transform(x, ico_order=5):
    """
    x (numpy array): shape (n_samples, n_features, n_vertices_downsampled)
        downsampled + axis-swapped data.

    returns
        upsampled_data (numpy array): shape (n_samples, n_vertices_full, n_features)
            reconstructed data at order 7.
    """
    # undo swapaxes (back to (n_samples, n_vertices, n_features))
    x = np.swapaxes(x, 1, 2)
    print("swapaxes undone ",np.shape(x))

    # get ico vertices
    from_verts, _ = icosahedron(ico_order)
    to_verts, _ = icosahedron(7)
    # print("from_verts ", np.shape(from_verts),"  ", np.shape(to_verts)) #(10242, 3)    (163842, 3)
    # print(type(from_verts))# np array

    # print("x.shape ", x.shape) #(1, 10242, 3)

    n_samples, n_vertices, n_features = x.shape
    # print("n_samples, n_vertices, n_features  ",n_samples, n_vertices, n_features ) #1 10242 3

    upsampled = interpolate_data_multimod(x, by=7 - ico_order)

    return upsampled



def get_vbm_wholebrain(site):
    input_transforms = Compose([Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),  Normalize()])

    dataset_train = BipolarDataset(site=site, preproc="vbm", split="train", transforms=input_transforms, datasize_idx=8, N763=False)
    dataset_test = BipolarDataset(site=site,preproc="vbm", split="test",transforms=input_transforms, datasize_idx=8, N763=False) 
    
    dataset_train = apply_transforms(dataset_train)
    dataset_test = apply_transforms(dataset_test)
    print("dataset_train ",np.shape(dataset_train), type(dataset_train))
    print("dataset_test ",np.shape(dataset_test), type(dataset_test))
    return dataset_train, dataset_test

def get_scores_voxelwise_VBM(dl_tl_5de_tr, dl_tl_5de_te, site):
    """
    get classificaiton scores for a specific site for 5-DE TL and VBM voxelwise features
    """
    maskvbm_tr = (
        (dl_tl_5de_tr["site"] == site) &
        (dl_tl_5de_tr["train_idx"] == 8) &
        (dl_tl_5de_tr["combination"] == 1)
    )
    maskvbm_te = (
        (dl_tl_5de_te["site"] == site) &
        (dl_tl_5de_te["train_idx"] == 8) &
        (dl_tl_5de_te["combination"] == 1)
    )
    dl_tl_5de_tr_one_combination = dl_tl_5de_tr[maskvbm_tr].copy()
    dl_tl_5de_te_one_combination = dl_tl_5de_te[maskvbm_te].copy()
    ypred_voxelwise_vbm_tr = dl_tl_5de_tr_one_combination["mean_ypred"].values
    ypred_voxelwise_vbm_te = dl_tl_5de_te_one_combination["mean_ypred"].values
    ypred_voxelwise_vbm_tr = ypred_voxelwise_vbm_tr[0]
    ypred_voxelwise_vbm_te = ypred_voxelwise_vbm_te[0]
    return ypred_voxelwise_vbm_tr, ypred_voxelwise_vbm_te

# def get_svm_wholebrain():
# include data transforms
def get_fwd_map_roi(scores, dfroi, list_rois, formula_res):
    all_Xim_tests, all_scores_tests = [], []
    ##loop on sites
    for site in get_predict_sites_list():
        # get participants train and test
        ids_tr = scores[site]["participant_ids_tr"]
        ids_te = scores[site]["participant_ids_te"]
        train_df = dfroi.set_index("participant_id").loc[ids_tr].reset_index()
        test_df = dfroi.set_index("participant_id").loc[ids_te].reset_index()

        X_train= train_df[list_rois].values
        X_test= test_df[list_rois].values

        if formula_res:
            combined_df = pd.concat([train_df, test_df], axis=0)

            residualizer = Residualizer(data=combined_df[list_rois+["age","sex","site","dx"]], \
                                        formula_res=formula_res, formula_full=formula_res+"+dx")
            Zres = residualizer.get_design_mat(combined_df[list_rois+["age","sex","site","dx"]])

            residualizer.fit(X_train, Zres[:len(X_train)])
            X_train = residualizer.transform(X_train, Zres[:len(X_train)])
            X_test = residualizer.transform(X_test, Zres[len(X_train):])

        scaler_ = StandardScaler()
        X_train = scaler_.fit_transform(X_train)
        X_test = scaler_.transform(X_test)
        # print("X_train :",type(X_train), np.shape(X_train))
        # print("X_test :",type(X_test), np.shape(X_test))
        # print("scores test :", type(scores[site]["score test"]), np.shape(scores[site]["score test"]))

        all_scores_tests.append(scores[site]["score test"])
        all_Xim_tests.append(X_test)

    # concatenate scores and input brain measures for all LOSO CV test sites
    all_Xim_tests = np.concatenate(all_Xim_tests,axis=0)
    all_scores_tests = np.concatenate(all_scores_tests,axis=0)
    return all_Xim_tests, all_scores_tests

def get_fwd_map_vbm_voxelwise(dl_tl_5de_tr, dl_tl_5de_te):
    all_Xim_tests, all_scores_tests = [], []
    ##loop on sites
    for site in get_predict_sites_list():
        # get prediction scores
            y_pred_tr, y_pred_te = get_scores_voxelwise_VBM(dl_tl_5de_tr, dl_tl_5de_te, site)
            print("y_pred_te ",np.shape(y_pred_te))
            _, dataset_test = get_vbm_wholebrain(site)
            dataset_test = dataset_test.squeeze()
            print("dataset test squeeze ",dataset_test.shape)
            dataset_test = dataset_test.reshape(dataset_test.shape[0], -1) 

            print("dataset test squeeze ",dataset_test.shape)
            all_Xim_tests.append(dataset_test.squeeze())
            all_scores_tests.append(y_pred_te)
            
    # concatenate scores and input brain measures for all LOSO CV test sites
    all_Xim_tests = np.concatenate(all_Xim_tests,axis=0)
    all_scores_tests = np.concatenate(all_scores_tests,axis=0)
    return all_Xim_tests, all_scores_tests

def plot_glassbrain_VBM_ROI(dict_plot=None, title=""): 
    """
        Aim : plot glassbrain of specfic ROI from covariance values obtained with an SVM-RBF and VBM ROI features
    """    

    for k,v in dict_plot.items():
        print(k, "  ",round(v,4))

    matching_rows = ATLAS_DF[ATLAS_DF["ROIabbr"].isin(list(dict_plot.keys()))]
    roi_names = matching_rows["ROIname"].tolist()
    print("roi_names ", roi_names)

    ref_im = nib.load(VOL_FILE_VBM)
    ref_arr = ref_im.get_fdata()
    # labels = sorted(set(np.unique(ref_arr).astype(int))- {0}) # 136 labels --> 'Left Inf Lat Vent', 
    # 'Right vessel', 'Left vessel' missing in data
    texture_arr = np.zeros(ref_arr.shape, dtype=float)
    
    for name, val in dict_plot.items():
        # each baseid is the number associated to the ROI in the nifti image
        baseids = ATLAS_DF[(ATLAS_DF['ROIabbr'] == name)]["ROIbaseid"].values

        name = ATLAS_DF[(ATLAS_DF['ROIabbr'] == name)]["ROIname"].values
        name = name[0]

        int_list = list(map(int, re.findall(r'\d+', baseids[0])))
        if "Left" in name: 
            if len(int_list)==2: baseid = int_list[1]
            else : baseid = int_list[0]
        else : 
            baseid = int_list[0]
        texture_arr[ref_arr == baseid] = val

    print("nb unique vals :",len(np.unique(texture_arr)), " \n",np.unique(texture_arr))
    print(np.shape(texture_arr))

    cmap = plt.cm.coolwarm
    vmin = np.min(texture_arr)
    vmax = np.max(texture_arr)
    print("vmin vmax texture arr", vmin,"     ",vmax)
    percentile_95 = np.quantile(np.abs(list(dict_plot.values())), 0.95)
    print("95th percentile ", percentile_95)
    texture_im = nib.Nifti1Image(texture_arr, ref_im.affine)

    if vmin==0:
        # if all values are positive, the color map should be a gradient from white (0) to red (max value)
        red_from_coolwarm = plt.cm.coolwarm(vmax)
        cmap = LinearSegmentedColormap.from_list("white_to_red", ["white", red_from_coolwarm])
    
    plotting.plot_glass_brain(
        texture_im,
        display_mode="ortho",
        colorbar=True,
        cmap=cmap,
        plot_abs=False ,
        alpha = 0.6 ,
        threshold=percentile_95,
        title=title)
    plotting.show() 

def uncrop(cropped, original_shape):
    """
    Place cropped array back into original shape with zeros filled in.
    
    cropped: np.ndarray (already cropped by Crop)
    original_shape: tuple, shape before cropping
    """
    out = np.zeros(original_shape, dtype=cropped.dtype)

    img_shape = np.array(original_shape)
    size = np.array(cropped.shape)
    indexes = []

    for ndim in range(len(img_shape)):
        delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
        indexes.append(slice(delta_before, delta_before + size[ndim]))

    out[tuple(indexes)] = cropped
    return out

def unpad(padded, original_shape):
    """
    Remove padding applied by Padding class.

    padded: np.ndarray (already padded)
    original_shape: tuple, shape before padding
    """
    slices = []
    for orig_i, final_i in zip(original_shape, padded.shape):
        diff = final_i - orig_i
        before = diff // 2
        after = before + orig_i
        slices.append(slice(before, after))
    for _ in range(len(padded.shape) - len(original_shape)):
        slices.append(slice(0, padded.shape[_]))
    return padded[tuple(slices)]

def get_sbm_vertex_wise_data(site="Baltimore", traindata_size=8, metrics=["thickness", "curv", "sulc"], ico_order=5):

    data_dict = {}
    for mod in MODALITIES_SBM_VERTEXWISE:
        data_path = os.path.join(DATA_DIR_SBM_VERTEXWISE, f"{mod}.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"Loading {data_path}...")
        all_data = np.load(data_path)  # Shape: (n_subjects, n_vertices, n_metrics)
        data_dict[mod] = all_data
        print(f"Loaded {mod}: {all_data.shape}")

    split_data = read_pkl(SPLITS_DICT_SBM_VERTEXWISE)
    split_key = site + "_" + str(traindata_size)

    initial_transform, _ = create_transforms_for_biobdbsnip( 
        modalities=MODALITIES_SBM_VERTEXWISE,
        data_dir=DATA_DIR_SBM_VERTEXWISE,
        ico_order=ico_order,
        metrics=metrics,
        overwrite=False,
        train_indices=split_data[split_key]["train"]
    )

    metadata = pd.read_csv(DATA_DIR_SBM_VERTEXWISE + "metadata.tsv", sep="\t")

    dataset_train = ClassificationDataset(
        data_dict=data_dict,
        metadata=metadata,
        indices=split_data[split_key]["train"],
        transforms=None,
        initial_transform=initial_transform
    )

    dataset_test = ClassificationDataset(
        data_dict=data_dict,
        metadata=metadata,
        indices=split_data[split_key]["test"],
        transforms=None,
        initial_transform=initial_transform
    )

    # print("dataset_train ",np.shape(dataset_train), type(dataset_train))
    # print("dataset_test ",np.shape(dataset_test), type(dataset_test))

    dataset_train = apply_transforms(dataset_train, type_feature="sbm")
    dataset_test = apply_transforms(dataset_test, type_feature="sbm")
    print("dataset_train ",np.shape(dataset_train), type(dataset_train))
    print("dataset_test ",np.shape(dataset_test), type(dataset_test))

    return dataset_train, dataset_test

def get_fwd_map_sbm_vertexwise():
    tl = read_pkl(PATH_DICT_5DE_TL_SBM)
    all_scores_tests, all_Xim_tests = [], []
    for site in get_predict_sites_list():
        all_scores_tests.append(tl[1][site]["ypred_te"]) # combination number = 1
        _, dataset_te = get_sbm_vertex_wise_data(site)
        all_Xim_tests.append(dataset_te)
        print(np.shape(dataset_te))

    all_scores_tests=np.array(all_scores_tests)
    all_scores_tests=np.concatenate(all_scores_tests,axis=0)
    all_Xim_tests=np.array(all_Xim_tests)
    all_Xim_tests=np.concatenate(all_Xim_tests, axis=0)
    print(np.shape(all_Xim_tests), type(all_Xim_tests))
    # Flatten from (763, 3, 20484) --> (763, 3 * 20484) where 3 is for the 3 metrics ["thickness", "curv", "sulc"]
    all_Xim_tests = all_Xim_tests.reshape(all_Xim_tests.shape[0], -1) # (763, 3 * 20484)
    assert np.shape(all_scores_tests) == (763,)
    assert np.shape(all_Xim_tests) == (763, 3 * 20484)
    return all_Xim_tests, all_scores_tests

def get_destrieux_lh_rn_names_and_labels():
    # Left and right annotation files
    lh_annot_file = PATH_SBM_DESTRIEUX+"lh.aparc.a2009s.annot"
    rh_annot_file = PATH_SBM_DESTRIEUX+"rh.aparc.a2009s.annot"

    # Read annotation files
    lh_labels, _, lh_names = fsio.read_annot(lh_annot_file)
    rh_labels, _, rh_names = fsio.read_annot(rh_annot_file)
    # lh_labels / rh_labels: arrays of shape (163842,), label index for each vertex (163842 = nb of vertices)
    # lh_ctab: color table (mapping index --> RGBA + integer ID)
    # lh_names: list of ROI names

    lh_names = [roi.decode("utf-8") for roi in lh_names]
    rh_names = [roi.decode("utf-8") for roi in rh_names]
    assert lh_names==rh_names # the two lists are supposed to be the same 
    # adding substrings to differentiate the two hemispheres
    lh_names=["lh_"+roi for roi in lh_names]
    rh_names=["rh_"+roi for roi in rh_names]
    return lh_names, rh_names, lh_labels, rh_labels

def create_SBM_plot_from_dict_roi_cov(dict_cov, threshold=None):
    """
    creates SBM ROI surface plots for ROI wise covariances 
    only for thickness ROI (only ROIs over threshold)
    """
    lh_names, rh_names , lh_labels, rh_labels = get_destrieux_lh_rn_names_and_labels()
    lh_names = [r for r in lh_names if "Medial_wall" not in r and "Unknown" not in r]
    rh_names = [r for r in rh_names if "Medial_wall" not in r and "Unknown" not in r]
    dict_cov= {k.replace("_thickness",""):v for k,v in dict_cov.items()}
    assert set(list(dict_cov.keys()))==set(lh_names+rh_names),"mismatch between roi names in dict and for plotting"

    # Left hemisphere
    cov_lh_roi = np.full_like(lh_labels, np.nan, dtype=float)
    for idx, roi_name in enumerate(lh_names):
        if roi_name in dict_cov:
            cov_lh_roi[lh_labels == idx] = dict_cov[roi_name]

    # Right hemisphere
    cov_rh_roi = np.full_like(rh_labels, np.nan, dtype=float)
    for idx, roi_name in enumerate(rh_names):
        if roi_name in dict_cov:
            cov_rh_roi[rh_labels == idx] = dict_cov[roi_name]

    fsaverage = datasets.fetch_surf_fsaverage('fsaverage7')
    # threshold for each hemisphere
    if threshold:
        cov_lh_roi[np.abs(cov_lh_roi) < threshold] = np.nan
        cov_rh_roi[np.abs(cov_rh_roi) < threshold] = np.nan


    if np.all(cov_lh_roi[np.isfinite(cov_lh_roi)] <= 0) and np.all(cov_rh_roi[np.isfinite(cov_rh_roi)] <= 0):
        cmap='Blues_r'
    else: 
        cmap='coolwarm'

    # Left hemisphere
    fig_lh, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 5), constrained_layout=True)

    plotting.plot_surf_stat_map(
        fsaverage.infl_left,
        stat_map=cov_lh_roi,
        hemi='left',
        view='lateral',
        bg_map=fsaverage.sulc_left,
        colorbar=True,
        title="Left hemisphere ROI-wise lateral",
        axes=axes[0],
        cmap=cmap
    )
    plotting.plot_surf_stat_map(
        fsaverage.infl_left,
        stat_map=cov_lh_roi,
        hemi='left',
        view='medial',
        bg_map=fsaverage.sulc_left,
        colorbar=False,
        title="Left hemisphere ROI-wise lateral",
        axes=axes[1],
        cmap=cmap
    )
    plt.subplots_adjust(wspace=0.1, left=0.05, right=0.95)

    if threshold : plt.savefig(IMG_PATH_FWD_MAP_PNG_SBM+f"lh_thickness_threshold95thpercentile_ROIwise.png", dpi=300)
    else : plt.savefig(IMG_PATH_FWD_MAP_PNG_SBM+f"lh_thickness_ROIwise.png", dpi=300)
    plt.close()

    # Right hemisphere
    fig_rh, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 5), constrained_layout=True)
    # plot lateral (outside) view
    plotting.plot_surf_stat_map(
        fsaverage.infl_right,
        stat_map=cov_rh_roi,
        hemi='right',
        view='lateral',
        bg_map=fsaverage.sulc_right,
        colorbar=True,
        title="Right hemisphere ROI-wise lateral",
        axes=axes[0],
        cmap=cmap,
    )
    
    # plot medial (inside) view
    plotting.plot_surf_stat_map(
        fsaverage.infl_right,
        stat_map=cov_rh_roi,
        hemi='right',
        view='medial',
        bg_map=fsaverage.sulc_right,
        colorbar=False,
        title="Right hemisphere ROI-wise medial",
        axes=axes[1],
        cmap=cmap
    )
    plt.subplots_adjust(wspace=0.1, left=0.05, right=0.95)

    if threshold : plt.savefig(IMG_PATH_FWD_MAP_PNG_SBM+f"rh_thickness_threshold95thpercentile_ROIwise.png", dpi=300)
    else : plt.savefig(IMG_PATH_FWD_MAP_PNG_SBM+f"rh_thickness_ROIwise.png", dpi=300)
    plt.close()



def get_sbm_vertexwise_plots_and_mean_cov_by_roi(cov_lh, cov_rh, threshold=None):
    """
    cov_lh (numpy.ndarray) : vertexwise covariances for the left hemisphere
    cov_rh (numpy.ndarray) : vertexwise covariances for the left hemisphere

    creates png plots of right and left hemispheres' covariances for thickness, curvature, and sulcation.
    generates df of mean absolute covariances for ROIs of Destrieux from vertexwise covariances

    if threshold is True, save plots with a threshold (95th percentile)
    
    """
    assert np.shape(cov_lh)==(163842, 3)
    assert np.shape(cov_rh)==(163842, 3)
    metrics=["thickness", "curv", "sulc"]
    roi_means = {}
    roi_means_signed = {}

    for i in range(len(metrics)):
        # loop through metrics

        ###============= get the ROI means from vertexwise covariances ==================###
        lh_names, rh_names ,lh_labels, rh_labels = get_destrieux_lh_rn_names_and_labels()

        # checking that they're the same roi as in sbm roi list
        all_rois = lh_names+rh_names
        all_rois=[r for r in all_rois if "Unknown" not in r]
        dfroi, list_rois = get_sbm_roi()
        list_rois=[r for r in list_rois if r.endswith("_area")] # keep only cortical rois for area (could have been thickness, they're the same)
        list_rois = [roi.replace("_area", "") for roi in list_rois] # remove endstring, keep only roi names
        # ['lh_Medial_wall', 'rh_Medial_wall'] are in all_rois but not in list_rois
        all_rois = [r for r in all_rois if "Medial_wall" not in r]
        assert set(all_rois)==set(list_rois) # make sure we have the same cortical roi names as in the SBM ROI dataframe (Destrieux atlas)

        # Left hemisphere
        for idx, roi_name in enumerate(lh_names):
            roi_vertices = (lh_labels == idx)
            if roi_vertices.sum() > 0:
                vals = cov_lh[:,i][roi_vertices]
                mean_abs = np.mean(np.abs(vals))
                roi_means[f"{roi_name}_{metrics[i]}"] = mean_abs

                # signed mean rule: sign determined by majority of positive vs negative values
                count_pos = np.sum(vals > 0)
                count_neg = np.sum(vals < 0)
                mean_signed = mean_abs if count_pos >= count_neg else -mean_abs
                roi_means_signed[f"{roi_name}_{metrics[i]}"] = mean_signed

        # Right hemisphere
        for idx, roi_name in enumerate(rh_names):
            roi_vertices = (rh_labels == idx)
            if roi_vertices.sum() > 0:
                vals = cov_rh[:, i][roi_vertices]
                mean_abs = np.mean(np.abs(vals))
                roi_means[f"{roi_name}_{metrics[i]}"] = mean_abs

                count_pos = np.sum(vals > 0)
                count_neg = np.sum(vals < 0)
                mean_signed = mean_abs if count_pos >= count_neg else -mean_abs
                roi_means_signed[f"{roi_name}_{metrics[i]}"] = mean_signed


    df = pd.DataFrame(list(roi_means.items()), columns=["ROI", "MeanAbsCov"])
    df_signed = pd.DataFrame(list(roi_means_signed.items()), columns=["ROI", "MeanAbsCovSigned"])
    df = pd.merge(df, df_signed, on="ROI", how="inner")
    print(df)
    roi_means_thickness = {k:v for k,v in roi_means.items() if k.endswith("_thickness")}
    roi_means_curv = {k:v for k,v in roi_means.items() if k.endswith("_curv")}
    roi_means_sulc = {k:v for k,v in roi_means.items() if k.endswith("_sulc")}
    
    percentiles={"all":np.quantile(np.abs(list(roi_means.values())), 0.95),\
                 "thickness":np.quantile(np.abs(list(roi_means_thickness.values())), 0.95),\
                    "curv":np.quantile(np.abs(list(roi_means_curv.values())), 0.95),\
                        "sulc": np.quantile(np.abs(list(roi_means_sulc.values())), 0.95)}
   
    print("overall covariances (abs values) 95th percentile ",percentiles["all"])
    print("95th percentile abs cov CT", round(percentiles["thickness"],4)) 
    for k, v in sorted(roi_means_thickness.items(), key=lambda x: x[1], reverse=True):
        if np.abs(v)>=percentiles["all"]:
            sign_roi=np.sign(roi_means_signed[k])
            print(f"{k:<40}: {sign_roi*round(v,4):>10.4f}")

    print("\n95th percentile abs cov curvature", round(percentiles["curv"],4)) 
    for k, v in sorted(roi_means_curv.items(), key=lambda x: x[1], reverse=True):
        if np.abs(v)>=percentiles["all"]:
            sign_roi=np.sign(roi_means_signed[k])
            print(f"{k:<40}: {sign_roi*round(v,4):>10.4f}")

    print("\n95th percentile abs cov sulcation", round(percentiles["sulc"],4)) 
    for k, v in sorted(roi_means_sulc.items(), key=lambda x: x[1], reverse=True):
        if np.abs(v)>=percentiles["all"]:
            sign_roi=np.sign(roi_means_signed[k])
            print(f"{k:<40}: {sign_roi*round(v,4):>10.4f}")

    for i in range(len(metrics)):
    # loop through metrics
        path_img_lh= IMG_PATH_FWD_MAP_PNG_SBM+f"lh_{metrics[i]}_thresholded_95thpercentile_vertexwise.png" if threshold \
            else IMG_PATH_FWD_MAP_PNG_SBM+f"lh_{metrics[i]}_vertexwise.png"
        path_img_rh=IMG_PATH_FWD_MAP_PNG_SBM+f"rh_{metrics[i]}_thresholded_95thpercentile_vertexwise.png" if threshold else \
            IMG_PATH_FWD_MAP_PNG_SBM+f"rh_{metrics[i]}_vertexwise.png"
    
        if os.path.exists(path_img_lh) and  os.path.exists(path_img_rh):
            # create plots for SBM vertexwise data if the images don't already exist

            fsaverage = datasets.fetch_surf_fsaverage('fsaverage7')
            map_lh = cov_lh[:,i]
            map_rh = cov_rh[:,i]
            # threshold for each hemisphere
            if threshold: 
                map_lh[np.abs(map_lh) < percentiles[metrics[i]]] = np.nan
                map_rh[np.abs(map_rh) < percentiles[metrics[i]]] = np.nan

            if np.all(map_lh[np.isfinite(map_lh)] <= 0) and np.all(map_rh[np.isfinite(map_rh)] <= 0):
                cmap='Blues_r'
            else: 
                cmap='coolwarm'

            fig_lh, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 5), constrained_layout=True)
            plotting.plot_surf_stat_map(
                fsaverage.infl_left,
                stat_map=map_lh,
                hemi='left',
                view='lateral',
                bg_map=fsaverage.sulc_left,
                colorbar=True,
                title=f"Left hemisphere vertex-wise lateral {metrics[i]}",
                axes=axes[0],
                cmap=cmap
            )
            plotting.plot_surf_stat_map(
                fsaverage.infl_left,
                stat_map=map_lh,
                hemi='left',
                view='medial',
                bg_map=fsaverage.sulc_left,
                colorbar=False,
                title=f"Left hemisphere vertex-wise medial {metrics[i]}",
                axes=axes[1],
                cmap=cmap
            )
            plt.subplots_adjust(wspace=0.1, left=0.05, right=0.95)
            plt.savefig(path_img_lh, dpi=300)  
            plt.close()

            fig_rh, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 5), constrained_layout=True)
            # Plot right hemisphere thickness
            plotting.plot_surf_stat_map(
                fsaverage.infl_right,
                stat_map=map_rh,
                hemi='right',
                view='lateral',
                bg_map=fsaverage.sulc_right,
                colorbar=True,
                axes=axes[0],
                title=f"Right hemisphere vertex-wise lateral {metrics[i]}",
                cmap=cmap
            )
            plotting.plot_surf_stat_map(
                fsaverage.infl_right,
                stat_map=map_rh,
                hemi='right',
                view='medial',
                bg_map=fsaverage.sulc_right,
                colorbar=False,
                title=f"Right hemisphere vertex-wise medial {metrics[i]}",
                axes=axes[1],
                cmap=cmap
            )
            plt.subplots_adjust(wspace=0.1, left=0.05, right=0.95)
            plt.savefig(path_img_rh, dpi=300)  
            plt.close()


    """
    95th percentile abs cov CT 0.0648
    rh_S_precentral-inf-part_thickness      :    -0.0838
    lh_G_front_sup_thickness                :    -0.0805
    lh_G_front_middle_thickness             :    -0.0765
    rh_S_front_inf_thickness                :    -0.0732
    lh_S_front_sup_thickness                :    -0.0714
    rh_G_front_inf-Orbital_thickness        :    -0.0699
    rh_S_precentral-sup-part_thickness      :    -0.0651
    lh_G_and_S_cingul-Mid-Ant_thickness     :    -0.0649

    95th percentile abs cov curvature 0.0046
    rh_Pole_temporal_curv                   :    -0.0073
    lh_G_cingul-Post-ventral_curv           :     0.0055
    lh_S_orbital_lateral_curv               :    -0.0052
    rh_G_temp_sup-Lateral_curv              :     0.0051
    rh_G_temporal_middle_curv               :    -0.0051
    rh_S_temporal_inf_curv                  :    -0.0048
    rh_G_and_S_transv_frontopol_curv        :     0.0047
    lh_G_and_S_occipital_inf_curv           :    -0.0046

    95th percentile abs cov sulcation 0.1847
    rh_Pole_temporal_sulc                   :    -0.3953
    rh_G_temp_sup-Lateral_sulc              :     0.3612
    rh_S_temporal_transverse_sulc           :     0.2578
    lh_S_front_inf_sulc                     :    -0.2518
    lh_S_intrapariet_and_P_trans_sulc       :     0.2221
    lh_S_collat_transv_post_sulc            :     0.2154
    rh_S_temporal_sup_sulc                  :     0.1899
    lh_S_occipital_ant_sulc                 :    -0.1883

    if we use the overall cov abs value 95th percentile as threshold over all measures, we get only sulcation ROIs:
    rh_Pole_temporal_sulc                   :    -0.3953
    rh_G_temp_sup-Lateral_sulc              :     0.3612
    rh_S_temporal_transverse_sulc           :     0.2578
    lh_S_front_inf_sulc                     :    -0.2518
    lh_S_intrapariet_and_P_trans_sulc       :     0.2221
    lh_S_collat_transv_post_sulc            :     0.2154
    rh_S_temporal_sup_sulc                  :     0.1899
    lh_S_occipital_ant_sulc                 :    -0.1883
    rh_S_front_middle_sulc                  :     0.1780
    rh_G_parietal_sup_sulc                  :     0.1707
    lh_S_postcentral_sulc                   :    -0.1683
    rh_G_front_middle_sulc                  :    -0.1670
    lh_G_parietal_sup_sulc                  :     0.1639
    rh_G_and_S_transv_frontopol_sulc        :     0.1527
    rh_G_temporal_middle_sulc               :    -0.1475
    rh_S_collat_transv_post_sulc            :     0.1449
    lh_G_front_middle_sulc                  :     0.1447
    rh_G_precuneus_sulc                     :    -0.1412
    rh_S_circular_insula_inf_sulc           :    -0.1403
    lh_S_temporal_sup_sulc                  :     0.1387
    rh_S_front_sup_sulc                     :     0.1375
    lh_S_front_sup_sulc                     :    -0.1365
    rh_S_temporal_inf_sulc                  :    -0.1360

    """
        

def get_ROI_Neuromorphometrics_from_voxelwiseVBM(percentile_99):
    """
    Get the ROIs of importance in the Neuromorphometrics atlas  
        for forward maps computed from voxel-wise VBM images. 

    ref_arr corresponds to the atlas image, and covariates_arr to the array of covariates from the forward model.
    
    """
    covariates_img = nib.load(IMAGE_VOXELWISE_VBM_FWDMAP_PATH)
    covariates_arr = covariates_img.get_fdata()

    ref_im = nib.load(VOL_FILE_VBM)
    ref_arr = ref_im.get_fdata()
    labels = sorted(set(np.unique(ref_arr).astype(int)) - {0})  # 136 labels --> 'Left Inf Lat Vent', 'Right vessel', 'Left vessel' missing in data

    # Use ATLAS_DF instead of XML parsing
    labels_to_index_roi = {}
    index_to_labels_roi = {}

    _, list_rois = get_vbm_roi()

    # Extract ROI base IDs and create mappings using ATLAS_DF
    for name_abbrv in list_rois:
        baseids = ATLAS_DF[(ATLAS_DF['ROIabbr'] == name_abbrv)]["ROIbaseid"].values
        name = ATLAS_DF[(ATLAS_DF['ROIabbr'] == name_abbrv)]["ROIname"].values
        roi_name = name[0]
        int_list = list(map(int, re.findall(r'\d+', baseids[0])))
        if "Left" in roi_name: 
            if len(int_list)==2: baseid = int_list[1]
            else : baseid = int_list[0]
        else : 
            baseid = int_list[0]

        # create mappings
        labels_to_index_roi[roi_name] = baseid
        index_to_labels_roi[baseid] = roi_name
    
    # Display data
    # create np array of same shape as nifti data
    ref_arr = np.array(ref_arr)
    covariates_arr = np.array(covariates_arr)

    ref_arr = ref_arr.flatten()
    covariates_arr = covariates_arr.flatten()

    mean_covariances_by_roi = []
    covariances_means, covariances_means_signed = {}, {}
    cpt = 0
    cov_res_dict = {}

    max_abs_val = -np.inf
    roi_with_max_abs = None

    for roi in labels: # integer labels of atlas nifti file (one int label encodes one roi)
        indices_roi = np.where(ref_arr == roi)
        covariates_for_this_roi = covariates_arr[indices_roi]
        # remove covariates equal to 0
        covariates_for_this_roi = covariates_for_this_roi[covariates_for_this_roi != 0]
        count_positive = np.sum(covariates_for_this_roi > 0)
        count_negative = np.sum(covariates_for_this_roi < 0)
        # print("count of positive covariates for current roi : ", count_positive)
        # print("count of negative covariates for current roi: ", count_negative)
        
        if len(covariates_for_this_roi) != 0:
            mean_of_abs_cov_for_this_roi = np.mean(np.abs(covariates_for_this_roi))
            max_abs_for_this_roi = np.max(np.abs(covariates_for_this_roi))
        else:
            mean_of_abs_cov_for_this_roi = 0
            max_abs_for_this_roi = 0
        
        # Use the mapping from ATLAS_DF instead of XML
        if roi in index_to_labels_roi:
            roi_name = index_to_labels_roi[roi]
        else:
            roi_name = f"Unknown_ROI_{roi}"  # Fallback for unmapped ROIs
        
        if max_abs_for_this_roi > max_abs_val:
            max_abs_val = max_abs_for_this_roi
            roi_with_max_abs = roi_name
            
        print("\n",roi_name)
        mean_covariances_by_roi.append(mean_of_abs_cov_for_this_roi)
        covariances_means[roi_name] = mean_covariances_by_roi[cpt]
        
        if count_positive < count_negative:
            covariances_means_signed[roi_name] = -mean_covariances_by_roi[cpt]
        else:
            covariances_means_signed[roi_name] = mean_covariances_by_roi[cpt]
            
        if len(covariates_for_this_roi) == 0:
            mean_covariances_by_roi[cpt] = 0

        cov_res_dict[roi_name]=mean_of_abs_cov_for_this_roi

        print("absolute mean of all covariance values for this ROI : ", round(mean_of_abs_cov_for_this_roi,4))
        cpt += 1
        
    print("\nROI with maximum absolute covariance value:", roi_with_max_abs)
    print("Maximum absolute covariance value:", max_abs_val)    
    percentile_90 = np.quantile(np.abs(list(cov_res_dict.values())), 0.90)

    if not os.path.exists(PATH_FWDMAP_VOXELWISE_PER_ROI_NEUROMORPHOMETRICS):
        with open(PATH_FWDMAP_VOXELWISE_PER_ROI_NEUROMORPHOMETRICS, 'w') as f:
            json.dump(cov_res_dict, f)
    
    def top_n_keys_with_highest_values(d, n=10):
        # Sort the dictionary by values in descending order and get the top n keys
        sorted_keys = sorted(d, key=d.get, reverse=True)[:n]
        return sorted_keys
    def top_n_keys_with_lowest_values(d, n=10):
        # Sort the dictionary by values in descending order and get the top n keys
        sorted_keys = sorted(d, key=d.get)[:n]
        return sorted_keys
    
    print("\nrois with absolute covariances higher than the 90th percentile")
    for k, v in sorted(covariances_means.items(), key=lambda item: abs(item[1]), reverse=True):
        if abs(v) >= percentile_90:
            print(k, round(v, 2), ", covariance sign : ", np.sign(covariances_means_signed[k]))

    
    """
    Left Putamen 0.69  sign :  1.0
    Right Putamen 0.68  sign :  1.0
    Left Parietal Operculum 0.38  sign :  -1.0
    Left Frontal Operculum 0.38  sign :  -1.0
    Right Parietal Operculum 0.37  sign :  -1.0
    Left Anterior Insula 0.36  sign :  -1.0
    Left Temporal Transverse Gyrus 0.35  sign :  -1.0
    Right Anterior Insula 0.34  sign :  -1.0
    Right Frontal Operculum 0.31  sign :  -1.0
    Right Anterior Cingulate Gyrus 0.3  sign :  -1.0
    Left Posterior Insula 0.3  sign :  -1.0
    Right Medial Frontal Cerebrum 0.29  sign :  -1.0
    Left Inferior Frontal Gyrus 0.29  sign :  -1.0
    Left Anterior Cingulate Gyrus 0.29  sign :  -1.0
    """
    
    print("\n\ntop 20 absolute covariances in rois: \n",top_n_keys_with_highest_values(covariances_means, 20),"\n\n\n")
    print("top highest (positive) 10 covariances signed found in rois : \n", top_n_keys_with_highest_values(covariances_means_signed, 10),
          "\ntop lowest (negative) 10 covariances signed found in rois: \n",top_n_keys_with_lowest_values(covariances_means_signed,10))
    """
    top highest (positive) 10 covariances signed found in rois : 
    ['Left Putamen', 'Right Putamen', 'Left Pallidum', 'Right Pallidum', 'Unknown_ROI_64', 'Right Temporal Transverse Gyrus', 
    'Left Precentral Gyrus', 'Right Middle Frontal Gyrus', 'Left Superior Temporal Gyrus', 'Right Thalamus Proper'] 

    top lowest (negative) 10 covariances signed found in rois: 
    ['Left Parietal Operculum', 'Left Frontal Operculum', 'Right Parietal Operculum', 'Left Anterior Insula', 
    'Left Temporal Transverse Gyrus', 'Right Anterior Insula', 'Right Frontal Operculum', 'Right Anterior Cingulate Gyrus', 
    'Left Posterior Insula', 'Right Medial Frontal Cerebrum']

    top 20 absolute covariances in rois:
    ['Left Putamen', 'Right Putamen', 'Left Parietal Operculum', 'Left Frontal Operculum', 
    'Right Parietal Operculum', 'Left Anterior Insula', 'Left Temporal Transverse Gyrus', 
    'Right Anterior Insula', 'Right Frontal Operculum', 'Right Anterior Cingulate Gyrus', 'Left Posterior Insula', 
    'Right Medial Frontal Cerebrum', 'Left Inferior Frontal Gyrus', 'Left Anterior Cingulate Gyrus', 'Left Pallidum', 
    'Left Central Operculum', 'Unknown_ROI_50', 'Right Central Operculum', 'Right Gyrus Rectus', 'Right Posterior Insula']

    """

def forward_maps(preproc="vbm", granularity="roi", formula_res="age+sex+site"):
    """
    Computes forward maps for :
        - GM and CSF volumes of VBM ROI, (but plots only GM volume related covariances)
        - CT, area, and subcortical volumes of SBM ROI (but plots only cortical thickness related covariances)
        - GM VBM voxelwise volume
        - SBM vertex wise measures (plots CT)
    
    preproc (str) : chosen preprocessing, "vbm" or "sbm"
    granularity (str) : "roi" or "wholebrain" , wholebrain referring to voxelwise for VBM measures and vertexwise for SBM measures.
    formula_res (str) : residualization formula, only applied to ROI measures
    """

    assert granularity in ["roi","wholebrain"]
    assert preproc in ["vbm","sbm"]

    if granularity=="roi": # get input data and paths of classification scores
        if preproc=="vbm": 
            dfroi, list_rois = get_vbm_roi()
            path_scores = ROOT+"results_classif/meta_model/scores_tr_te_N861_train_size_N8_vbmroi.pkl"
        if preproc=="sbm": 
            dfroi, list_rois = get_sbm_roi()
            path_scores = ROOT+"results_classif/meta_model/EN_N8_Destrieux_SBM_ROI_N763.pkl"
        # get prediction scores
        scores = read_pkl(path_scores)
        print(scores.keys())
        all_Xim_tests, all_scores_tests = get_fwd_map_roi(scores, dfroi, list_rois, formula_res)
        # compute covariance
        cov = compute_covariance(all_Xim_tests, all_scores_tests)
        print(np.shape(cov), type(cov))

    if granularity =="wholebrain":
        if preproc=="vbm": 
            dl_tl_5de_tr = read_pkl(LEARNING_CRV_TL_DIR+"mean_ypred_252_combinations_df_TL_densenet_Train_set.pkl")
            dl_tl_5de_te = read_pkl(LEARNING_CRV_TL_DIR+"mean_ypred_252_combinations_df_TL_densenet_Test_set.pkl")
            
            if not os.path.exists(COVARIANCES_VBM_VOXELWISE):
                all_Xim_tests, all_scores_tests = get_fwd_map_vbm_voxelwise(dl_tl_5de_tr, dl_tl_5de_te)
                cov = compute_covariance(all_Xim_tests, all_scores_tests) # compute covariance
                np.save(COVARIANCES_VBM_VOXELWISE, cov.squeeze())
            else : cov= np.load(COVARIANCES_VBM_VOXELWISE)
            # reshape cov to shape of an image after transforms were applied (as fed to the CNN) 
            cov = cov.reshape((1,128,128,128))
            # the transforms were : 1. crop 2. pad 3. normalize, therefore here we
            # 1. unpad, 2. uncrop (the normalization wouldn't make sense on covariances, since it was computed
            # with different std and mean for different LOSO test sites + we're looking at the relative importance of voxels
            # , not necessarily the absolute units matching raw images)
            cov = unpad(cov, (1,121,128,121))
            cov = uncrop(cov, (1,121,145,121))
            cov = cov.squeeze()
            assert cov.shape==(121,145,121)
            print(cov.shape)
            mask_img = nib.load(BINARY_BRAIN_MASK)
            new_img = new_img_like(mask_img , cov)
            nib.save(new_img, IMAGE_VOXELWISE_VBM_FWDMAP_PATH)

            fig = plt.figure(figsize=(20, 10))
            percentile_99 = np.quantile(np.abs(cov), 0.99)
            print(" 99th percentile absolute value : ",percentile_99)

            display = plotting.plot_glass_brain(
                new_img,
                display_mode="ortho",
                colorbar=True,
                cmap=plt.cm.coolwarm,
                plot_abs=False ,
                alpha = 0.6 ,
                threshold=percentile_99,
                title="forward map voxelwise VBM",
                figure = fig)

            cbar = display._cbar  # Access the colorbar object
            cbar.ax.tick_params(labelsize=20)  # Set the font size for the color bar labels
            if not os.path.exists(IMG_PATH_FWD_MAP_PNG_VBM): display.savefig(IMG_PATH_FWD_MAP_PNG_VBM, dpi=300)
            get_ROI_Neuromorphometrics_from_voxelwiseVBM(percentile_99)

        if preproc=="sbm":
            if not os.path.exists(COVARIANCES_SBM_VERTEXWISE_RESHAPED_LH) and not os.path.exists(COVARIANCES_SBM_VERTEXWISE_RESHAPED_RH):
                if not os.path.exists(COVARIANCES_SBM_VERTEXWISE):
                    all_Xim_tests, all_scores_tests = get_fwd_map_sbm_vertexwise()
                    cov = compute_covariance(all_Xim_tests, all_scores_tests) # compute covariance
                    print("cov shape and type", type(cov),np.shape(cov))
                    np.save(COVARIANCES_SBM_VERTEXWISE, cov.squeeze())
                else : cov= np.load(COVARIANCES_SBM_VERTEXWISE)
                cov = cov.reshape((3, 20484))
                print("cov shape and type", type(cov),np.shape(cov))

                # undo concatenation of Right and Left hemispheres
                cov_lh = cov[:, :int(20484/2)]
                cov_rh = cov[:, int(20484/2):]
                print("cov left hemi shape and type", type(cov_lh),np.shape(cov_lh))

                """
                Original data shapes (from npy files of 3D volumes): 
                Loading /neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/surface-lh_data.npy...
                Loaded surface-lh_data: (763, 163842, 3)  # Shape: (n_subjects, n_vertices, n_metrics)
                Loading /neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/data/processed/surface-rh_data.npy...
                Loaded surface-rh_data: (763, 163842, 3)  # Shape: (n_subjects, n_vertices, n_metrics)
                """
                
                cov_lh = cov_lh.reshape((1, *cov_lh.shape))
                cov_rh = cov_rh.reshape((1, *cov_rh.shape))

                # apply reverse transforms to get covariances in the space of features before transformations 
                cov_lh = reverse_initial_transform(cov_lh) 
                cov_rh = reverse_initial_transform(cov_rh) 
                print("cov left hemi shape and type", type(cov_lh),np.shape(cov_lh))
                print("cov right hemi shape and type", type(cov_rh),np.shape(cov_rh))
                np.save(COVARIANCES_SBM_VERTEXWISE_RESHAPED_LH,cov_lh)
                np.save(COVARIANCES_SBM_VERTEXWISE_RESHAPED_RH, cov_rh)

            else: 
                cov_lh = np.load(COVARIANCES_SBM_VERTEXWISE_RESHAPED_LH)
                cov_rh = np.load(COVARIANCES_SBM_VERTEXWISE_RESHAPED_RH)
                cov_lh=np.squeeze(cov_lh)
                cov_rh=np.squeeze(cov_rh)
                print("cov left hemi shape and type", type(cov_lh),np.shape(cov_lh)) # <class 'numpy.ndarray'> (163842, 3)
                print("cov right hemi shape and type", type(cov_rh),np.shape(cov_rh)) # <class 'numpy.ndarray'> (163842, 3)
                get_sbm_vertexwise_plots_and_mean_cov_by_roi(cov_lh, cov_rh) #, threshold=True)
            
                
    if granularity=="roi":
        print("covariance array shape and type ", np.shape(cov),type(cov))
        dict_cov = dict(zip(list_rois,cov))
        # print(dict_cov)
        
        if preproc=="vbm":
            assert np.shape(cov)==(280,)
            dict_GM = {k:v for k,v in dict_cov.items() if k.endswith("_GM_Vol")}
            percentile_95 = np.quantile(np.abs(list(dict_GM.values())), 0.95)
            print("95th percentile ", percentile_95)
            for k, v in sorted(dict_GM.items(), key=lambda x: x[1], reverse=True):
                if np.abs(v)>=percentile_95:
                    name = ATLAS_DF[(ATLAS_DF['ROIabbr'] == k)]['ROIname'].values
                    print(f"{name[0]}: {round(v,4)}")
            """
            'Left Pallidum': 0.23791969417582098
            'Right Pallidum': 0.23378581894041026
            'Left Anterior Insula': -0.15275887098746835
            'Left Middle Temporal Gyrus': -0.15423021307341456
            'Right Lingual Gyrus': -0.15907781750228547
            'Right Superior Occipital Gyrus': -0.1654395752461516
            'Left Middle Frontal Gyrus': -0.1682084983123802
            """
            dict_CSF = {k:v for k,v in dict_cov.items() if k.endswith("_CSF_Vol")}
            plot_glassbrain_VBM_ROI(dict_GM)
            # 95th percentile  0.148223492151889

        if preproc=="sbm":
            # print(dict_cov)
            # selecting CT ROIs only, as the only measures present both in vertexwise and ROI-wise SBM features are
            # measures of cortical thickness
            percentile_95 = np.quantile(np.abs(list(dict_cov.values())), 0.95)
            print("95th percentile ", percentile_95)

            for k, v in sorted(dict_cov.items(), key=lambda x: x[1], reverse=True):
                if np.abs(v)>=percentile_95:
                    print(f"{k:<40} {v:>10.4f}")

            """
            # when we compute the threshold and covariance for CT, area, and subcortical ROIs: 
            95th percentile  0.11407346570256681
            rh_S_postcentral_thickness                  -0.1141
            lh_S_intrapariet_and_P_trans_thickness      -0.1146
            lh_S_temporal_sup_thickness                 -0.1147
            rh_S_oc_middle_and_Lunatus_thickness        -0.1148
            lh_G_and_S_cingul-Mid-Post_thickness        -0.1153
            rh_G_and_S_cingul-Mid-Post_thickness        -0.1155
            rh_S_parieto_occipital_thickness            -0.1158
            lh_S_front_inf_thickness                    -0.1162
            lh_S_precentral-sup-part_thickness          -0.1196
            rh_S_temporal_sup_thickness                 -0.1196
            lh_S_postcentral_thickness                  -0.1213
            lh_G_front_middle_thickness                 -0.1216
            lh_S_parieto_occipital_thickness            -0.1217
            lh_G_front_sup_thickness                    -0.1269
            rh_S_oc_sup_and_transversal_thickness       -0.1270
            lh_S_front_sup_thickness                    -0.1314
            rh_S_intrapariet_and_P_trans_thickness      -0.1333

            # when we compute the threshold and covariance for CT ROIs only: 
            95th percentile  0.11962149324465816
            rh_S_temporal_sup_thickness                 -0.1196
            lh_S_postcentral_thickness                  -0.1213
            lh_G_front_middle_thickness                 -0.1216
            lh_S_parieto_occipital_thickness            -0.1217
            lh_G_front_sup_thickness                    -0.1269
            rh_S_oc_sup_and_transversal_thickness       -0.1270
            lh_S_front_sup_thickness                    -0.1314
            rh_S_intrapariet_and_P_trans_thickness      -0.1333
            """

            dict_CT = {k:v for k,v in dict_cov.items() if k.endswith("_thickness")}
            percentile_95_CT = np.quantile(np.abs(list(dict_CT.values())), 0.95)
            print("\n95th percentile ", percentile_95_CT)

            for k, v in sorted(dict_CT.items(), key=lambda x: x[1], reverse=True):
                if np.abs(v)>=percentile_95_CT:
                    print(f"{k:<40} {v:>10.4f}")

            create_SBM_plot_from_dict_roi_cov(dict_CT, threshold=percentile_95_CT)


def main():
    # forward_maps(preproc="sbm")
    forward_maps(granularity="wholebrain", preproc="sbm")


if __name__ == "__main__":
    main()
