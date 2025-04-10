import sys, os, time, scipy
import numpy as np
import nibabel
import pandas as pd
import seaborn as sns
import ast, re
from scipy.cluster.hierarchy import linkage, dendrogram
import shap
import xml.etree.ElementTree as ET
import nilearn.plotting as plotting
from utils import get_predict_sites_list, read_pkl, get_LOSO_CV_splits_N861, get_participants, save_pkl, get_LOSO_CV_splits_N763, \
compute_covariance, get_reshaped_4D, inverse_transform, round_sci
from deep_ensemble import get_mean
from classif_VoxelWiseVBM_ML import get_all_data
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from univariate_stats import get_scaled_data
from sklearn.linear_model import LogisticRegressionCV
from classif_VBMROI import remove_zeros 
from torchvision.transforms.transforms import Compose

from transforms import Crop, Padding, Normalize

sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer

# inputs
ROOT = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
SHAP_DIR_SVMRBF=ROOT+"models/ShapValues/shap_computed_from_all_Xtrain/"
DATAFOLDER=ROOT+"data/processed/"
VOL_FILE_VBM = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"
VBMLOOKUP_FILE = "/drf/local/spm12/tpm/labels_Neuromorphometrics.xml"
MASK_FILE ="/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/mni_cerebrum-gm-mask_1.5mm.nii.gz"

# outputs
RESULTS_FEATIMPTCE_AND_STATS_DIR=ROOT+"results_feat_imptce_and_univ_stats/"
# /neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/results_feat_imptce_and_univ_stats/specific_ROI_SHAP_SVMRBF_VBM.xlsx

def make_shap_df(verbose=False, VBM=False, SBM=False):
    assert not (VBM and SBM),"a feature type has to be chosen between VBM ROI and SBM ROI"

    # SHAP only computed for VBM ROI and age+sex+site residualization
    if VBM:
        VBMdf = pd.read_csv(DATAFOLDER+"VBMROI_Neuromorphometrics.csv")
        exclude_elements = ['participant_id', 'session', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']
        VBMdf = VBMdf.drop(columns=exclude_elements)
        list_roi = list(VBMdf.columns)
        columns_with_only_zeros = VBMdf.columns[(VBMdf == 0).all()]
        list_roi = [ roi for roi in list_roi if roi not in columns_with_only_zeros]
        assert len(list_roi)==280
        columns_with_zeros_indices = [VBMdf.columns.get_loc(col) for col in columns_with_only_zeros]
    if SBM : 
        # always Destrieux and with subcortical regions
        SBMdf = pd.read_csv(DATAFOLDER+"SBMROI_Destrieux_CT_SA_subcortical_N763.csv")
        list_roi = [col for col in list(SBMdf.columns) if col not in ["participant_id","TIV"]]

    df_all_shap = pd.DataFrame(columns=['fold', 'ROI', 'mean_abs_shap','shap_array', 'mean_abs_shap_rdm','shap_array_rdm'])

    shap_all, shap_all_rdm = [], []
    if VBM: str_VBM = "VBM_SVM_RBF"
    if SBM: str_SBM = "SBM_EN"

    for site in get_predict_sites_list():
        print(site)
        
        if VBM : shap_path = SHAP_DIR_SVMRBF + "ShapValues_"+str_VBM+"_"+site+"_background_alltr_parallelized_avril25_run1.pkl"
        if SBM : shap_path = SHAP_DIR_SVMRBF + "ShapValues_"+str_SBM+"_"+site+"_background_alltr_parallelized_Destrieux_run1.pkl"
        all_runs_shap_rdm = []

        for run in range(1, 31): # from 1 to 30 included
            if VBM : shap_path_rdm = SHAP_DIR_SVMRBF + "ShapValues_"+str_VBM+"_"+site+"_background_alltr_parallelized_randomized_labels_run"+str(run)+".pkl"
            if SBM : shap_path_rdm = SHAP_DIR_SVMRBF + "ShapValues_"+str_SBM+"_"+site+"_background_alltr_parallelized_randomized_labels_Destrieux_run"+str(run)+".pkl"
            shap_rdm = read_pkl(shap_path_rdm)
            if VBM : 
                shap_rdm = np.array([exp.values for exp in shap_rdm])
                # the four VBM ROIs with zero values have been removed before computing the shap values 
                shap_rdm = np.delete(shap_rdm, columns_with_zeros_indices, axis=1)
                assert shap_rdm.shape[1]==280 , f"Expected 280 columns (array with permutations), \
                    but got {shap_rdm.shape[1]} columns. Site: {site}"
                # 280 ROI (we removed the ROI that had all values across subjects equal to 0 (their shap values are also zeros))
            if SBM:
                assert shap_rdm.shape[1]==330 , f"Expected 330 columns (array with permutations), \
                    but got {shap_rdm.shape[1]} columns. Site: {site}"
            all_runs_shap_rdm.append(shap_rdm)

        all_runs_shap_rdm = np.concatenate(all_runs_shap_rdm, axis=0)
        if VBM : assert all_runs_shap_rdm.shape[1]==280
        if SBM : assert all_runs_shap_rdm.shape[1]==330
        if verbose : print("all_runs_shap_rdm ",all_runs_shap_rdm.shape)
        shap_all_rdm.append(all_runs_shap_rdm)

        # mean over test set subjects x 30 runs with permuted labels : shape = (nb roi,) = (280,)
        mean_abs_shap_rdm = np.mean(np.abs(all_runs_shap_rdm), axis=0) 
        shap_values = read_pkl(shap_path)

        if VBM : 
            shap_values = np.array([exp.values for exp in shap_values])

        assert shap_values.shape[0]*30==all_runs_shap_rdm.shape[0]
        if VBM : assert shap_values.shape[1]==280, f"Expected 280 columns (array without permutations), \
                but got {shap_values.shape[1]} columns. Site: {site}"
        if SBM : assert shap_values.shape[1]==330, f"Expected 330 columns (array without permutations), \
                but got {shap_rdm.shape[1]} columns. Site: {site}"
            
        shap_all.append(shap_values)
        if verbose : print("shap_values ",shap_values.shape)

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0) # mean over test set subjects : shape = (nb roi,) = (280,)
        std_abs_shap = np.std(np.abs(shap_values), axis=0,  ddof=1) # std over test set subjects : shape = (nb roi,) = (280,)
        n = np.repeat(len(shap_values), len(list_roi)).tolist()  # count of test set subjects

        repeated_foldname = np.repeat(site, len(list_roi)).tolist()
        # mean_abs_shap is the mean absolute shap value for ONE fold (for each ROI)
        rows={"fold" : repeated_foldname, "ROI":list_roi, 'mean_abs_shap':mean_abs_shap.tolist(), 'std_abs_shap':std_abs_shap.tolist(), 'count_test_subjects':n, \
              'shap_array':[], 'mean_abs_shap_rdm':mean_abs_shap_rdm.tolist(), 'shap_array_rdm':[]}
    
        for idx_roi in range(0 , shap_values.shape[1]): # loop through ROIs
            rows["shap_array"].append(shap_values[: , idx_roi]) # shap_values shape (nb of test set subjects, 280)
            rows["shap_array_rdm"].append(all_runs_shap_rdm[:,idx_roi]) # all_runs_shap_rdm shape (30 x nb of test set subjects, 280)
        
        df = pd.DataFrame(rows)
        df_all_shap = pd.concat([df_all_shap , df], axis=0)


    concatenated_array = np.concatenate(shap_all, axis=0)
    concatenated_array_rdm = np.concatenate(shap_all_rdm, axis=0)

    dict_means_by_roi = {list_roi[i]:np.mean(np.abs(concatenated_array),axis=0)[i] for i in range(0,len(list_roi))}
    dict_means_by_roi_rdm = {list_roi[i]:np.mean(np.abs(concatenated_array_rdm),axis=0)[i] for i in range(0,len(list_roi))}

    # dict_means_by_roi contains the mean absolute shap values across ALL folds (test subjects of all folds concatenated) (for each ROI)
    # add mean of absolute shap values by ROI across all LOSO-CV sites
    df_all_shap["abs_mean_shap_by_roi"] = df_all_shap["ROI"].map(dict_means_by_roi)
    df_all_shap["abs_mean_shap_by_roi_rdm"] = df_all_shap["ROI"].map(dict_means_by_roi_rdm)
    print(df_all_shap)
    if VBM: df_all_shap.to_excel(SHAP_DIR_SVMRBF+'SHAP_summary_including_shap_from_30rdm_runs_VBM_ROI_avril25.xlsx', index=False)
    if SBM: df_all_shap.to_excel(SHAP_DIR_SVMRBF+'SHAP_summary_including_shap_from_30rdm_runs_SBM_ROI_avril25.xlsx', index=False)

    return df_all_shap

def get_shared_specific_or_suppressor_ROIs_btw_folds(dict_ROIs, verbose=False):
    # Convert arrays to sets and find the intersection
    shared_strings = set(dict_ROIs[next(iter(dict_ROIs))])  # Start with the first key's set

    for arr in dict_ROIs.values():
        shared_strings.intersection_update(arr)  # Keep only shared elements

    if verbose: print("Shared specific ROIs across all folds:", shared_strings, " len : ", len(shared_strings))

    # Print nb of suppressor or specific ROIs for each fold
    for key, arr in dict_ROIs.items():
        if verbose : print(f"Number of specific ROIs for {key}: {len(arr)}")

    return shared_strings


def get_all_Xte_concatenated(VBM=False, SBM=False):
    # retrieve all test set ROI values for all folds
    if VBM : 
        splits = get_LOSO_CV_splits_N861() 
        Nmax=800  # using maximum training set size for dataset with all VBM-preprocessed subjects, N861 
    if SBM : 
        splits = get_LOSO_CV_splits_N763()
        Nmax = 700 # using maximum training set size for dataset with all SBM-preprocessed subjects, N763

    # read participants dataframe
    participants = get_participants()
    participants_all = list(splits["Baltimore-"+str(Nmax)][0])+list(splits["Baltimore-"+str(Nmax)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_ROI = participants.iloc[msk]   
    participants_ROI = participants_ROI.reset_index(drop=True)

    # prep residualizer
    formula_res, formula_full = "site + age + sex", "site + age + sex + dx"
    residualizer = Residualizer(data=participants_ROI, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(participants_ROI)

    # read ROI df
    if VBM : ROIdf = pd.read_csv(DATAFOLDER+"VBMROI_Neuromorphometrics.csv")
    if SBM : ROIdf = pd.read_csv(DATAFOLDER+"SBMROI_Destrieux_CT_SA_subcortical_N763.csv")
    # reorder VBMdf to have rows in the same order as participants_ROI
    ROIdf = ROIdf.set_index('participant_id').reindex(participants_ROI["participant_id"].values).reset_index()

    # create concatenation of all test sets participant's ROIs in Xte_arr np array
    Xte_list, list_rois, participants = [], [], []

    for site in get_predict_sites_list():
        df_te_ = ROIdf[ROIdf["participant_id"].isin(splits[site+"-"+str(Nmax)][1])]
        df_tr_ = ROIdf[ROIdf["participant_id"].isin(splits[site+"-"+str(Nmax)][0])]
        # find index in participants df of the train and test subjects for the current LOSO CV site and train data size
        train = list(participants_ROI.index[participants_ROI['participant_id'].isin(splits[site+"-"+str(Nmax)][0])])
        test = list(participants_ROI.index[participants_ROI['participant_id'].isin(splits[site+"-"+str(Nmax)][1])])

        if VBM : exclude_elements = ['participant_id', 'session', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']
        if SBM : exclude_elements = ['participant_id', 'TIV']
        participants.append(df_te_["participant_id"].values)
        df_te_ = df_te_.drop(columns=exclude_elements)
        df_tr_ = df_tr_.drop(columns=exclude_elements)
        if VBM : 
            columns_with_only_zeros = ROIdf.columns[(ROIdf == 0).all()]
            columns_with_zeros_indices = [ROIdf.columns.get_loc(col) for col in columns_with_only_zeros]
            df_te_ = remove_zeros(df_te_) # remove ROIs with zero values for all subjects
            df_tr_ = remove_zeros(df_tr_)

        # fit residualizer just like before classification (using the training data for each fold)
        X_train = df_tr_.values
        X_test = df_te_.values
        residualizer.fit(X_train, Zres[train])
        X_test = residualizer.transform(X_test, Zres[test])

        # fit scaler using training data for each fold
        scaler_ = StandardScaler()
        X_train = scaler_.fit_transform(X_train)
        X_test = scaler_.transform(X_test)

        if VBM : assert np.shape(X_test)[1]==280, "wrong nb of ROIs"
        if SBM : assert np.shape(X_test)[1]==330, "wrong nb of ROIs"

        Xte_list.append(X_test)
        if site=="Baltimore": list_rois = list(df_te_.columns)

    all_folds_Xtest_concatenated = np.concatenate(Xte_list, axis=0)
    participants = np.concatenate(participants,axis=0)
    if SBM : return all_folds_Xtest_concatenated, list_rois, participants
    if VBM : return all_folds_Xtest_concatenated, list_rois, columns_with_zeros_indices, participants

def read_bootstrapped_shap(save=False, VBM=False, SBM=False):
    # remider : in this study, SHAP values are only computed for VBM ROI and age+sex+site residualization
    assert not (VBM and SBM),"a feature type has to be chosen between VBM ROI and SBM ROI"

    # get univariate statistics (obtained with univariate_stats.py)
    if VBM : path_univ_statistics = RESULTS_FEATIMPTCE_AND_STATS_DIR+"statsuniv_rois_res_age_sex_site_VBM_avril25.xlsx"
    if SBM : path_univ_statistics = RESULTS_FEATIMPTCE_AND_STATS_DIR+"statsuniv_rois_res_age_sex_site_SBM_avril25.xlsx"

    univ_statistics = pd.read_excel(path_univ_statistics)

    if VBM: path_shap_summary = SHAP_DIR_SVMRBF+'SHAP_summary_including_shap_from_30rdm_runs_VBM_ROI_avril25.xlsx'
    if SBM: path_shap_summary = SHAP_DIR_SVMRBF+'SHAP_summary_including_shap_from_30rdm_runs_SBM_ROI_avril25.xlsx'

    if not os.path.exists(path_shap_summary) :
        df_all_shap = make_shap_df()
    else : df_all_shap = pd.read_excel(path_shap_summary)

    if VBM : all_folds_Xtest_concatenated, list_rois , columns_with_zeros_indices, _ = get_all_Xte_concatenated(VBM=VBM, SBM=SBM)
    if SBM : all_folds_Xtest_concatenated, list_rois , _ = get_all_Xte_concatenated(VBM=VBM, SBM=SBM)

    print(type(all_folds_Xtest_concatenated), np.shape(all_folds_Xtest_concatenated))
        
    # SHAP Statistics mean,std, n, CI 
    # abs_mean_shap_by_roi : mean over all concatenated subjects / shap for all roi
    # mean_abs_shap : mean for each fold separately for all roi --> mean of this : mean over the means of each fold
    shap_df_stats = df_all_shap[['fold', 'ROI', 'mean_abs_shap', 'std_abs_shap', 'count_test_subjects', "mean_abs_shap_rdm"]] 
    print(shap_df_stats)
    
    # Ensure uniqueness in df: check that each ROI has a unique abs_mean_shap_by_roi value
    # df_all_shap["abs_mean_shap_by_roi"] has 3360 (12*280) values, and for each ROI the values
    # are the same between folds as they correspond to the mean abs shap values of all concatenated test sets' subjects.
    if df_all_shap.groupby("ROI")["abs_mean_shap_by_roi"].nunique().max() > 1:
        raise ValueError("Some ROIs have multiple different mean_abs_shap values.")

    m_concatenated = df_all_shap.groupby(['ROI'])["abs_mean_shap_by_roi"].mean()
    top_30_dict = m_concatenated.nlargest(30).to_dict()
    
    print("Top 30 ROIs with the highest mean absolute SHAP values over test subjects from all folds concatenated: ")
    for k,v in top_30_dict.items():
        print(k,  "   ",v)

    specific_ROI={fold:[] for fold in get_predict_sites_list()}
    suppressor_ROI={fold:[] for fold in get_predict_sites_list()}

    concatenated_shap_arrays = []

    # doing the analysis for each fold
    for site in get_predict_sites_list():
        shap_df_stats_current_site = shap_df_stats[shap_df_stats["fold"]==site].copy()
        shap_df_stats_current_site = shap_df_stats_current_site.set_index("ROI")

        if VBM : 
            shap_arr_one_fold = read_pkl(SHAP_DIR_SVMRBF + "ShapValues_VBM_SVM_RBF_"+site+"_background_alltr_parallelized_avril25_run1.pkl")
            shap_arr_one_fold = np.array([exp.values for exp in shap_arr_one_fold])
            # shap_arr_one_fold = np.delete(shap_arr_one_fold, columns_with_zeros_indices, axis=1)
        if SBM : shap_arr_one_fold = read_pkl(SHAP_DIR_SVMRBF + "ShapValues_SBM_EN_"+site+"_background_alltr_parallelized_Destrieux_run1.pkl")

        concatenated_shap_arrays.append(shap_arr_one_fold)

        # compute mean, standard deviation, and count of mean absolute shap values for testing set of current fold
        m = shap_df_stats_current_site["mean_abs_shap"]
        s = shap_df_stats_current_site["std_abs_shap"]
        n = shap_df_stats_current_site["count_test_subjects"]
        if VBM : assert len(m)==len(s) and len(s)==len(n) and len(n)==280
        if SBM : assert len(m)==len(s) and len(s)==len(n) and len(n)==330

        # Critical value for t at alpha / 2:
        t_alpha2 = -scipy.stats.t.ppf(q=0.05/2, df=n-1, loc=0)
        ci_low = m - t_alpha2 * s / np.sqrt(n)
        ci_high = m + t_alpha2 * s / np.sqrt(n)
        # mean_abs_shap_rdm are the means of the absolute shap values for all test set subjects between the 30 runs for current LOSO-CV fold
        shap_rnd_absmean = shap_df_stats_current_site[["mean_abs_shap_rdm"]].copy()

        # Convert to a single-row DataFrame with ROIs as columns
        shap_rnd_absmean = shap_rnd_absmean.T
        m_h0 = shap_rnd_absmean
        m_h0 = m_h0[m.index]

        shap_stat = dict(ROI=m.index.values, mean_abs_shap=m.values, ci_low=ci_low.values, ci_high=ci_high.values,
                  mean_abs_shap_h0=m_h0.iloc[0].values, select = m_h0.iloc[0].values < ci_low.values)
        shap_stat = pd.DataFrame(shap_stat)

        shap_stat.sort_values(by="mean_abs_shap", ascending=False, inplace=True)
        shap_stat.reset_index(drop=True, inplace=True)

        # merge with univariate statistics
        shap_stat = pd.merge(shap_stat.reset_index(drop=True), univ_statistics.reset_index(drop=True),
                        on='ROI', how='left')        

        # Split variable into specifics and suppressors
        # check that there is no "type" column in shap_stat (this column will describe the type of variable (specific or suppressor))
        if not "type" in shap_stat: 
            # Filter Features based on significance of SHAP values
            shap_stat = shap_stat[shap_stat.select == 1]
            if site =="Baltimore" and VBM: assert shap_stat.shape[0] == 140, f"shape is {shap_stat.shape[0]} but should be 140" # check that we have 140 SHAP where H0 is rejected
            if site =="Baltimore" and SBM: assert shap_stat.shape[0] == 112 # check that we have 143 SHAP where H0 is rejected

            shap_stat["type"] = None # initialize empty column
            shap_stat.loc[shap_stat.diag_pcor_bonferroni < 0.05, "type"] = "specific" 
            shap_stat.loc[shap_stat.diag_pcor_bonferroni > 0.05, "type"] = "suppressor"

            if save:
                if VBM : shap_stat.to_excel(RESULTS_FEATIMPTCE_AND_STATS_DIR+"shap_from_SVM_RBF_VBM_"+site+"-univstat_avril25.xlsx", sheet_name='SHAP_roi_univstat_'+site, index=False)
                if SBM : shap_stat.to_excel(RESULTS_FEATIMPTCE_AND_STATS_DIR+"shap_from_EN_SBM_"+site+"-univstat_avril25.xlsx", sheet_name='SHAP_roi_univstat_'+site, index=False)

        specific_ROI[site]=shap_stat[shap_stat["type"]=="specific"]["ROI"].values
        suppressor_ROI[site]=shap_stat[shap_stat["type"]=="suppressor"]["ROI"].values

    shared_strings_spec = get_shared_specific_or_suppressor_ROIs_btw_folds(specific_ROI)
    shared_strings_supp = get_shared_specific_or_suppressor_ROIs_btw_folds(suppressor_ROI)  

    # lists of specific and suppressor ROIs shared across all 12 folds
    shap_spec = list(shared_strings_spec)
    shap_suppr = list(shared_strings_supp)

    ################# SAVE SUMMARY VALUES FOR ANALYSIS OF SHAP VALUES ######################  
    concatenated_shap_arrays=np.concatenate(concatenated_shap_arrays, axis=0)

    assert concatenated_shap_arrays.shape == all_folds_Xtest_concatenated.shape, \
        f"Shape mismatch! SHAP: {concatenated_shap_arrays.shape}, X_test: {all_folds_Xtest_concatenated.shape}"

    indices_specificROI = [list_rois.index(roi) for roi in shap_spec if roi in list_rois]
    indices_suppressorROI = [list_rois.index(roi) for roi in shap_suppr if roi in list_rois]

    if VBM :
        # Negate SHAP values for ROIs containing 'CSF_Vol' --> CSF volume varies inversely from GM volume
        csf_indices = [i for i, roi in enumerate(list_rois) if 'CSF_Vol' in roi]
        concatenated_shap_arrays_negatedCSF = concatenated_shap_arrays.copy()
        concatenated_shap_arrays_negatedCSF[:, csf_indices] *= -1

        # get dictionary of correspondecies of ROI names from abbervations to full ROI names
        atlas_df = pd.read_csv(ROOT+"data/atlases/lobes_Neuromorphometrics.csv", sep=';')
        dict_atlas_roi_names = atlas_df.set_index('ROIabbr')['ROIname'].to_dict()

    print("concatenated_shap_arrays_negatedCSF ",np.shape(concatenated_shap_arrays_negatedCSF))
    # Order specific ROIs by mean absolute SHAP values
    if VBM : mean_abs = np.mean(np.abs(concatenated_shap_arrays_negatedCSF[:,indices_specificROI]),axis=0)
    if SBM : mean_abs = np.mean(np.abs(concatenated_shap_arrays[:,indices_specificROI]),axis=0)

    ordered_indices = np.argsort(mean_abs)[::-1]  # [::-1] reverses the order to get descending
    sorted_indices_specificROI = np.array(indices_specificROI)[ordered_indices]

    # save info necessary for summary plot of SHAP values for specific and suppressor ROI
    dict_summary = {"indices_specific_roi":indices_specificROI,"indices_suppressorROI":indices_suppressorROI,\
        "sorted_indices_specificROI":sorted_indices_specificROI, "all_folds_Xtest_concatenated":all_folds_Xtest_concatenated,"list_rois":list_rois,\
            "concatenated_shap_arrays":concatenated_shap_arrays}
    if VBM : 
        dict_summary["concatenated_shap_arrays_with_negated_CSF"]=concatenated_shap_arrays_negatedCSF
        dict_summary["dict_atlas_roi_names"]=dict_atlas_roi_names
        if save: save_pkl(dict_summary, RESULTS_FEATIMPTCE_AND_STATS_DIR+"ShapSummaryDictionnaryForBeeswarmPlot_VBM.pkl")

    if SBM and save : save_pkl(dict_summary, RESULTS_FEATIMPTCE_AND_STATS_DIR+"ShapSummaryDictionnaryForBeeswarmPlot_SBM.pkl")

def plot_beeswarm(VBM=False, SBM=False):
    """
        Aim: printing the mean absolute shap values and their feature's values for all shap values of all 861 test subjects 
            (concatenation of test subjects' feature values and SHAP values for the 12 folds), only for ROIs/features previously 
            found to be 'specific' (ie. have a (hypothetically) direct impact on BD diagnosis).
    """
    assert not (VBM and SBM),"a feature type has to be chosen between VBM ROI and SBM ROI"

    if VBM : dict_summary= read_pkl(RESULTS_FEATIMPTCE_AND_STATS_DIR+"ShapSummaryDictionnaryForBeeswarmPlot_VBM.pkl")
    if SBM : dict_summary= read_pkl(RESULTS_FEATIMPTCE_AND_STATS_DIR+"ShapSummaryDictionnaryForBeeswarmPlot_SBM.pkl")

    if VBM : concatenated_shap_arrays_with_negated_CSF = dict_summary["concatenated_shap_arrays_with_negated_CSF"]
    if SBM : concatenated_shap_arrays = dict_summary["concatenated_shap_arrays"]
    sorted_indices_specificROI = dict_summary["sorted_indices_specificROI"]
    all_folds_Xtest_concatenated = dict_summary["all_folds_Xtest_concatenated"]
    if VBM : dict_atlas_roi_names = dict_summary["dict_atlas_roi_names"]
    list_rois = dict_summary["list_rois"]

    if SBM : mean_abs_specific = np.mean(np.abs(concatenated_shap_arrays[:,sorted_indices_specificROI]),axis=0)
    if VBM : mean_abs_specific = np.mean(np.abs(concatenated_shap_arrays_with_negated_CSF[:,sorted_indices_specificROI]),axis=0)

    # SHAP summary plot
    # if VBM : shap.summary_plot(concatenated_shap_arrays_with_negated_CSF[:,sorted_indices_specificROI], all_folds_Xtest_concatenated[:,sorted_indices_specificROI], \
    #                   feature_names=[dict_atlas_roi_names[list_rois[i]] for i in sorted_indices_specificROI], max_display=len(sorted_indices_specificROI))
   
    # if SBM : shap.summary_plot(concatenated_shap_arrays[:,sorted_indices_specificROI], all_folds_Xtest_concatenated[:,sorted_indices_specificROI], \
    #                   feature_names=[list_rois[i] for i in sorted_indices_specificROI], max_display=len(sorted_indices_specificROI))
   
    print("Sorted ROI (highest to lowest mean_abs):", [list_rois[i] for i in sorted_indices_specificROI])
    # get univariate statistics (obtained with univariate_stats.py)
    path_univ_statistics = RESULTS_FEATIMPTCE_AND_STATS_DIR+"statsuniv_rois_res_age_sex_site.xlsx"
    univ_statistics = pd.read_excel(path_univ_statistics)
    stats_specific_ROI = univ_statistics[univ_statistics["ROI"].isin([list_rois[i] for i in sorted_indices_specificROI])].copy()
    recap = stats_specific_ROI[stats_specific_ROI["diag_pcor_bonferroni"] < 0.05][["ROI", "diag_pcor_bonferroni"]]
    recap["mean_abs_shap"] = np.round(mean_abs_specific,4)
    recap['diag_pcor_bonferroni'] = recap['diag_pcor_bonferroni'].apply(lambda x: round_sci(x, sig=2))

    if VBM : 
        recap["ROI"] = recap['ROI'].replace(dict_atlas_roi_names)
        save_pkl(recap, RESULTS_FEATIMPTCE_AND_STATS_DIR+"specific_ROI_SHAP_SVMRBF_VBM.pkl")
        recap.to_excel(RESULTS_FEATIMPTCE_AND_STATS_DIR+"specific_ROI_SHAP_SVMRBF_VBM.xlsx", sheet_name='specificROI_SHAP_SVMRBF_VBM_AVRIL25', index=False)
    if SBM : 
        save_pkl(recap, RESULTS_FEATIMPTCE_AND_STATS_DIR+"specific_ROI_SHAP_EN_SBM.pkl")
        recap.to_excel(RESULTS_FEATIMPTCE_AND_STATS_DIR+"specific_ROI_SHAP_EN_SBM.xlsx", sheet_name='specificROI_SHAP_EN_SBM_AVRIL25', index=False)

    print("p-values after bonferroni correction and mean abs shap values for specific ROIs :\n", recap)


def plot_glassbrain(dict_plot=None, title=""): 
    """
        Aim : plot glassbrain of specfic ROI from SHAP values obtained with an SVM-RBF and VBM ROI features
    """
    if dict_plot is None:
        glassbrain_VBM_SHAP_analysis = True
        specific_roi_df = read_pkl(RESULTS_FEATIMPTCE_AND_STATS_DIR+"specific_ROI_SHAP_SVMRBF_VBM.pkl")
        dict_plot = specific_roi_df.set_index('ROI')['mean_abs_shap'].to_dict()
        print(specific_roi_df)
    
    print("dict_plot\n")
    for k,v in dict_plot.items():
        print(k, "  ",v)

    ref_im = nibabel.load(VOL_FILE_VBM)
    ref_arr = ref_im.get_fdata()
    # labels = sorted(set(np.unique(ref_arr).astype(int))- {0}) # 136 labels --> 'Left Inf Lat Vent', 'Right vessel', 'Left vessel' missing in data
    atlas_df = pd.read_csv(ROOT+"data/atlases/lobes_Neuromorphometrics.csv", sep=';')
    texture_arr = np.zeros(ref_arr.shape, dtype=float)
    
    
    for name, val in dict_plot.items():
        # get GM volume (there is one row for GM and another for CSF for each ROI but the ROIbaseid values are the same for both so we picked GM vol)
        # each baseid is the number associated to the ROI in the nifti image
        baseids = atlas_df[(atlas_df['ROIname'] == name) & (atlas_df['volume'] == 'GM')]["ROIbaseid"].values
        int_list = list(map(int, re.findall(r'\d+', baseids[0])))
        if "Left" in name: 
            if len(int_list)==2: baseid = int_list[1]
            else : baseid = int_list[0]
        else : baseid = int_list[0]

        if glassbrain_VBM_SHAP_analysis:
            # we know from the glassbrain plot that in the case of VBM ROI (SHAP value analysis), the ROIs for which
            # a higher GM vol is associated with a higher SHAP value are only the left and right pallidum, so we "paint"
            # them red in the glassbrain (keep the values positive (they're absolute values so all positive in dict_plot)), 
            # while we "paint" the others in blue (multiply the absolute values by -1, making them all negative)
            if name in ["Left Pallidum", "Right Pallidum"]: texture_arr[ref_arr == baseid] = val
            else : texture_arr[ref_arr == baseid] = - val
        else : texture_arr[ref_arr == baseid] = val

    print("nb unique vals :",len(np.unique(texture_arr)), " \n",np.unique(texture_arr))
    print(np.shape(texture_arr))

    cmap = plt.cm.coolwarm
    vmin = np.min(texture_arr)
    vmax = np.max(texture_arr)
    print("vmin vmax texture arr", vmin,"     ",vmax)
    texture_im = nibabel.Nifti1Image(texture_arr, ref_im.affine)
    if title == "": title="specific ROI"
    plotting.plot_glass_brain(
        texture_im,
        display_mode="ortho",
        colorbar=True,
        cmap=cmap,
        plot_abs=False ,
        alpha = 0.95 ,
        threshold=0,
        title=title)
    plotting.show()

def get_list_specific_supp_ROI(VBM=False, SBM=False):
    assert not (VBM and SBM),"a feature type has to be chosen between VBM ROI and SBM ROI"

    specific_ROI={fold:[] for fold in get_predict_sites_list()}
    suppressor_ROI={fold:[] for fold in get_predict_sites_list()}
    if VBM : str_roi = "SVM_RBF_VBM"
    if SBM : str_roi = "EN_SBM"

    for site in get_predict_sites_list():
        shap_stat=pd.read_excel(RESULTS_FEATIMPTCE_AND_STATS_DIR+"shap_from_"+str_roi+"_"+site+"-univstat_avril25.xlsx", sheet_name='SHAP_roi_univstat_'+site)
        # print(shap_stat)
        shap_spec = shap_stat[shap_stat.type=="specific"]
        # print(shap_spec)

        specific_ROI[site]=shap_stat[shap_stat["type"]=="specific"]["ROI"].values
        suppressor_ROI[site]=shap_stat[shap_stat["type"]=="suppressor"]["ROI"].values

    # get list of overall specific and suppressor ROI (ROIs with mean abs SHAP values that are within the CI for all LOSO-CV folds)
    shared_strings_spec = list(get_shared_specific_or_suppressor_ROIs_btw_folds(specific_ROI))
    shared_strings_supp = list(get_shared_specific_or_suppressor_ROIs_btw_folds(suppressor_ROI))
    return shared_strings_spec, shared_strings_supp

def regression_analysis_with_specific_and_suppressor_ROI(VBM=False, SBM=False, plot_and_save_jointplot=False, plot_and_save_kde_plot=False):
    assert not (VBM and SBM),"a feature type has to be chosen between VBM ROI and SBM ROI"
    #Illustrate Suppressor variable with linear model
    data = get_scaled_data(res="res_age_sex_site",VBM=VBM, SBM=SBM)
    print("data ", type(data), np.shape(data))
    print(data)
   
    # get list of overall specific and suppressor ROI (ROIs with mean abs SHAP values that are within the CI for all LOSO-CV folds)
    shared_strings_spec, shared_strings_supp = get_list_specific_supp_ROI(VBM=VBM, SBM=SBM)

    print("number of specific ROI ", len(shared_strings_spec), "number of suppressor ROI ", len(shared_strings_supp))

    X = data[shared_strings_spec + shared_strings_supp]
    print("X ", np.shape(X), type(X))
    y = data.dx

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegressionCV()
    lr.fit(X_scaled, y)
    assert lr.coef_.shape[1] == len(shared_strings_spec) + len(shared_strings_supp)

    # get coefficients
    coef_spec = lr.coef_[0, :len(shared_strings_spec)] # coefficients of specific ROI
    coef_supr = lr.coef_[0, len(shared_strings_spec):] # coefficients of specific ROI

    # compute scores : linear combination of ROI with their respective weights
    score_spec = np.dot(X_scaled[:, :len(shared_strings_spec)], coef_spec)
    score_supr = np.dot(X_scaled[:, len(shared_strings_spec):], coef_supr)  
    score_tot = np.dot(X_scaled, lr.coef_[0, :]) # score tot being the score with both suppressor + specific ROIs, but not all 280 ROIs

    score_spec_auc = roc_auc_score(y, score_spec)
    score_supr_auc = roc_auc_score(y, score_supr)
    score_tot_auc = roc_auc_score(y, score_tot)

    df = pd.DataFrame(dict(diagnosis=y.map({0: "HC", 1: "BD"}),  # mapped for plotting only
                           score_spec=score_spec, score_supr=score_supr, score_tot=score_tot))
    
    print(df)
    
    if plot_and_save_jointplot: 
        # Create a joint plot with scatter and marginal density plots
        plt.figure(figsize=(8, 8))
        g = sns.jointplot(data=df, x="score_supr", y="score_spec", hue="diagnosis")
        g.ax_joint.set_xlabel("Score with Suppressor Features (AUC=%.2f)" % score_supr_auc, fontsize = 20)
        g.ax_joint.set_ylabel("Score with Speficic Features  (AUC=%.2f)" % score_spec_auc, fontsize = 20)
        # Increase font size for legend title and labels and ticks
        legend = g.ax_joint.legend_
        legend.set_title("Diagnosis", prop={'size': 20})  
        for text in legend.get_texts():
            text.set_fontsize(20)  
        g.ax_joint.tick_params(axis='both', labelsize=16)
        g.figure.suptitle("Scatter Plot with Marginal Densities", y=0.98, fontsize=22)
        
        plt.savefig(RESULTS_FEATIMPTCE_AND_STATS_DIR + "plot_suppressor-specific_scatter_all_folds.pdf")  
        plt.show()
        plt.close()

    if plot_and_save_kde_plot:
        # Density (KDE) Plot
        plt.figure(figsize=(6, 2))
        g = sns.kdeplot(data=df, x='score_tot', hue='diagnosis', fill=True, alpha=0.3)
        g.set_xlabel("Density Plot of Score with specific and suppressor features (AUC=%.2f)" % score_tot_auc, fontsize=20)
        g.set_ylabel("")
        # Increase x and y tick size
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # Increase legend font size
        legend = g.legend_
        legend.set_title("Diagnosis", prop={'size': 20})  # Set legend title font size
        for text in legend.get_texts():
            text.set_fontsize(20)  # Set font size of legend labels
        plt.grid(True)
        plt.title("Density Plot of Score with all both specific and suppressor ROI (AUC=%.2f)" % score_tot_auc)
        plt.savefig(RESULTS_FEATIMPTCE_AND_STATS_DIR + "plot_suppressor-specific_density.pdf")  
        plt.show()
        plt.close()

def plot_cluster_abs_corr_mar(corr_matrix):
    # [Improve the figure](https://fr.moonbooks.org/Articles/Les-dendrogrammes-avec-Matplotlib/)
    from scipy.cluster.hierarchy import linkage, dendrogram
    fig = plt.figure(figsize=(8, 8))  # Create an 8x8 inch figure

    # Apply hierarchical clustering to reorder correlation matrix 
    # clustering based on dissimilarity (1 - correlation)
    # if corr= 1, that means identical --> distance = 0
    # if corr = 0, distance = 1
    linkage_matrix = linkage(1 - corr_matrix, method="ward")  # Ward's method for clustering
    
    #dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
    dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=False)
    plt.grid(False)
    sorted_columns = dendro["ivl"]  # Reordered column names

    # Reordering the matrix based on cluster hierarchy (from dendrogram).
    reordered_corr = corr_matrix.loc[sorted_columns, sorted_columns]
    print("reordered_corr ",np.unique(reordered_corr), "\n",type(reordered_corr), np.shape(reordered_corr))

    # Plot the clustered heatmap with a colormap optimized for positive values
    plt.figure(figsize=(8, 6))
    sns.set_theme(font_scale=0.5)
    g = sns.heatmap(reordered_corr, fmt=".2f", cmap="Reds", square=True,
                linewidths=0.5, vmin=0, vmax=1)

    plt.title("Clustered Correlation Matrix (Absolute Values)")
    plt.show()
    sns.set(font_scale=1.0)

    return dendro

def plot_pca(data):
    """
    Aim : performing Principal Component Analysis (PCA) and plotting the cumulative variance explained by the components
    """

    # Standardize the data (important for PCA)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA()
    pca.fit(data_scaled)

    # Compute cumulative explained variance = how much total variance is explained as you add more components.
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot explained variance
    # Plots the cumulative variance curve: tells you how many components you need to explain most of the variance.
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Components')
    plt.grid(True)
    plt.show()

    return pca, data_scaled

def exploratory_analysis(VBM=True, SBM=False): # implemented for VBM ROI only so far
    # get list of overall specific and suppressor ROI (ROIs with mean abs SHAP values that are within the CI for all LOSO-CV folds)
    shared_strings_spec, shared_strings_supp = get_list_specific_supp_ROI(VBM=VBM, SBM=SBM)
    print(shared_strings_spec)
    data = get_scaled_data(res="res_age_sex_site",VBM=VBM, SBM=SBM)
    print("data ", type(data), np.shape(data))
    print(data)
    quit()
    # shap_spec = shap_stat[shap_stat.type == "specific"]
    corr_matrix = data[shared_strings_spec].corr().abs()

    dendro = plot_cluster_abs_corr_mar(data[shared_strings_spec].corr().abs())
    pca, data_scaled = plot_pca(data[shared_strings_spec])

    plot_cluster_abs_corr_mar(data[shared_strings_supp].corr().abs())
    pca, data_scaled = plot_pca(data[shared_strings_supp])




def forward_maps_ROI(VBM=False, SBM=False):
    # Read classification scores
    if VBM : scores_path = ROOT+"results_classif/stacking/SVMRBF_VBMROI/scores_tr_te_N861_train_size_N800.pkl"
    if SBM : scores_path = ROOT+"results_classif/stacking/EN_SBMROI/scores_tr_te_N763_train_size_N700.pkl"
    scores = read_pkl(scores_path)
    concatenated_scores , concatenated_ids = [], []
    for site in get_predict_sites_list():
        concatenated_scores.append(scores[site]["score test"])
        concatenated_ids.append(scores[site]["participant_ids_te"])
    concatenated_ids= np.concatenate(concatenated_ids, axis=0)
    concatenated_scores= np.concatenate(concatenated_scores, axis=0)

    if VBM : all_folds_Xtest_concatenated, list_rois , columns_with_zeros_indices , participants_te = get_all_Xte_concatenated(VBM=VBM, SBM=SBM)
    if SBM : all_folds_Xtest_concatenated, list_rois, participants_te = get_all_Xte_concatenated(VBM=VBM, SBM=SBM)

    print("participants ", type(participants_te), np.shape(participants_te))
    assert np.array_equal(concatenated_ids, participants_te)
    print("concatenated_scores ",np.shape(concatenated_scores))
    print("all_folds_Xtest_concatenated ",np.shape(all_folds_Xtest_concatenated))

    # Compute Covariances between the ROIs of test sets subjects and their corresponding classification scores
    cov = compute_covariance(all_folds_Xtest_concatenated, concatenated_scores) #shape of cov is the nb of ROI (280 for VBM, 330 for SBM)
    dict_cov = dict(zip(list_rois,cov))
    if VBM:
        for roi in list_rois:
            if "CSF_Vol" in roi: dict_cov[roi]*=-1

    indices_rois_GM =[i for i, roiname in enumerate(list_rois) if "GM_Vol" in roiname]
    indices_rois_CSF =[i for i, roiname in enumerate(list_rois) if "CSF_Vol" in roiname]

    list_rois_GM=[roi for roi in list_rois if "GM_Vol" in roi]
    list_rois_CSF=[roi for roi in list_rois if "CSF_Vol" in roi]

    # Compute the 95th percentile threshold for absolute value of covariances
    thresholdGM = np.percentile(abs(cov[indices_rois_GM]), 95)
    thresholdCSF = np.percentile(abs(cov[indices_rois_CSF]), 95)

    print("threshold ",thresholdGM, "  ",thresholdCSF)
    GM_impt, CSF_impt = [],[]
    # Save values >= 95th percentile
    for k,v in dict_cov.items():
        if "GM_Vol" in k and (abs(v)>=thresholdGM): 
            print(k, "  ",v)
            GM_impt.append(k)
        if "CSF_Vol" in k and (abs(v)>=thresholdCSF): 
            print(k, "  ",v)
            CSF_impt.append(k)

    # Create the dictionary of important ROIs depending on covariances
    cov_impt = {k: v for k, v in dict_cov.items() if k in GM_impt+CSF_impt}
    roi_shared = [roi for roi in GM_impt if roi in CSF_impt]
    assert roi_shared==[]

    if VBM:
        # get dictionary of correspondecies of ROI names from abbervations to full ROI names
        atlas_df = pd.read_csv(ROOT+"data/atlases/lobes_Neuromorphometrics.csv", sep=';')
        dict_atlas_roi_names = atlas_df.set_index('ROIabbr')['ROIname'].to_dict()

    new_dict = {dict_atlas_roi_names.get(k, k): v for k, v in cov_impt.items()}

    plot_glassbrain(new_dict, title="VBM ROI with significant covariates obtained with forward maps")


def forward_maps_voxelwise(verbose=False, display_plot=False):
    mean_ypred_by_site_dict,  y_true_for_each_site_dict = get_mean(list_models= [1,3,5,7,8], transfer=True, metric = "roc_auc", \
             verbose=False, test = True, train=False, model ="densenet")
    print(type(mean_ypred_by_site_dict), np.shape(mean_ypred_by_site_dict), mean_ypred_by_site_dict.keys())
    print(type(mean_ypred_by_site_dict["Baltimore"]))

    cov_array_file = RESULTS_FEATIMPTCE_AND_STATS_DIR+"voxelwise_covariances_array.pkl" 
    temporary_brain_mask_file = RESULTS_FEATIMPTCE_AND_STATS_DIR+"brain_mask_np_with_transforms.pkl"

    if not (os.path.exists(cov_array_file) and os.path.exists(temporary_brain_mask_file)):
        total = 0
        scores_concat = []
        for site in get_predict_sites_list():
            total+=len(mean_ypred_by_site_dict[site])
            scores_concat.append(mean_ypred_by_site_dict[site])

        scores_concat = np.concatenate(scores_concat,axis=0)
        assert total==861, "wrong number of total subjects"

        df_Xim_all = get_all_data(verbose=False)

        participants = get_participants()
        print("np.shape participants",np.shape(participants))
        splits = get_LOSO_CV_splits_N861()    
        # select the participants for VBM ROI (train+test participants of any of the 12 splits)
        # it has to be for max training set size, otherwise it won't cover the whole range of subjects
        participants_all = list(splits["Baltimore-"+str(800)][0])+list(splits["Baltimore-"+str(800)][1])
        msk = list(participants[participants['participant_id'].isin(participants_all)].index)
        participants_VBM = participants.iloc[msk]   
        participants_VBM = participants_VBM.reset_index(drop=True)

        # transforms applied to images before training the DL models
        input_transforms = Compose([Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),  Normalize()])
        input_transformsmask = Compose([Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant')])

        # reorder voxelwiseVBMdf to have rows in the same order as participants_VBM
        voxelwiseVBMdf = df_Xim_all.set_index('participant_id').reindex(participants_VBM["participant_id"].values).copy().reset_index()
        concatenated_Xte = []
        # get training and testing ROI dataframes (contains participant_id + TIV in addition to 330 ROIs)
        for site in get_predict_sites_list():
            df_te_ = voxelwiseVBMdf[voxelwiseVBMdf["participant_id"].isin(splits[site+"-"+str(800)][1])]
            y_test = pd.merge(df_te_, participants_VBM, on ="participant_id")["dx"].values
            assert (np.array_equal(y_test, y_true_for_each_site_dict[site])), "error in the data!"
            df_te_ = df_te_.drop(columns="participant_id")
            X_test = np.vstack(df_te_["data"].values) 
            assert np.shape(X_test)[1]==331695
            print("X_test shape ",np.shape(X_test))

            data = get_reshaped_4D(X_test, MASK_FILE)
            data = np.reshape(data, (data.shape[0], 1, *data.shape[1:]))
            print(np.shape(data), type(data))
            
            mask_img = nibabel.load(MASK_FILE)
            mask_data = mask_img.get_fdata() # contains only 0 and 1 (binary mask), shape (121,145,121)
            if verbose:
                print(np.sum(mask_data==0))
                print(np.sum(mask_data==1))
            # apply transforms to mask (for future plotting)
            mask_data = mask_data.reshape(1, *mask_data.shape)
            print("mask_data ",np.shape(mask_data), type(mask_data))
            # apply transforms to mask
            mask_data = input_transformsmask(mask_data).squeeze()
            print("mask_data ",np.shape(mask_data), type(mask_data))
            save_pkl(mask_data, temporary_brain_mask_file)

            list_Ximte=[]
            # loop through test subjects of current fold
            for subject in range(0,len(data)):
                print(site, "  ",subject,"/",len(data))
                if verbose: print("shape subject ",np.shape(data[subject]), type(data[subject]))
                # apply transforms
                sub_3D = input_transforms(data[subject]).squeeze()
                if verbose: print("shape after transforms :",np.shape(sub_3D), type(sub_3D))
                sub_3Dflat = sub_3D.flatten()
                print("sub_3Dflat ",np.shape(sub_3Dflat))
                assert sub_3Dflat.shape == (2097152,)
                list_Ximte.append(sub_3Dflat)

            list_Ximte = np.array(list_Ximte)
            concatenated_Xte.append(list_Ximte)
            
        concatenated_Xte = np.concatenate(concatenated_Xte, axis=0)
        print("concatenated_Xte ",np.shape(concatenated_Xte), type(concatenated_Xte))
        # compute covariances
        cov = compute_covariance(concatenated_Xte, scores_concat)
        print("cov shape and type for all sites ", np.shape(cov),type(cov))
        print("cov not null ",np.sum(cov!=0))
        #save covariances array
        save_pkl(cov, cov_array_file)
    else : 
        image_cov_path = RESULTS_FEATIMPTCE_AND_STATS_DIR+"forward_maps_voxelwise.nii.gz"
        mask_img = nibabel.load(MASK_FILE)
        if not os.path.exists(image_cov_path):
            cov = read_pkl(cov_array_file)
            mask_data = read_pkl(temporary_brain_mask_file)
            mask_data_flat = mask_data.flatten()

            mask_data_flat[mask_data_flat==1] = cov[mask_data_flat==1]
            print(np.shape(mask_data_flat), type(mask_data_flat))
            reshaped_img = mask_data_flat.reshape(*mask_data.shape)
            print(np.shape(reshaped_img), type(reshaped_img))

            new_img = nibabel.Nifti1Image(reshaped_img, mask_img.affine)
            print(np.shape(new_img), type(new_img))
            # invert transforms (except for normalization since we don't need it here)
            inverted_array = inverse_transform(reshaped_img)
            print(np.shape(inverted_array), type(inverted_array))
            new_img = nibabel.Nifti1Image(inverted_array, mask_img.affine)

            print(np.shape(new_img), type(new_img))
            nibabel.save(new_img, image_cov_path)
        else : 
            image = nibabel.load(image_cov_path)
            data = image.get_fdata()
            mask_data = mask_img.get_fdata()
            flatdata = data.flatten()
            maskflat = mask_data.flatten()
            data_masked=flatdata[maskflat==1]
            print("np.percentile(np.abs(data_masked), 90) ",np.percentile(np.abs(data_masked), 90))
            print("np.percentile(np.abs(data_masked), 95) ",np.percentile(np.abs(data_masked), 95))


            if display_plot: 
                plotting.plot_glass_brain(
                    image,
                    display_mode="ortho",
                    colorbar=True,
                    cmap = plt.cm.coolwarm,
                    plot_abs=False ,
                    alpha = 0.95 ,
                    threshold=np.percentile(np.abs(data_masked), 95),
                    title="covariances of forward models for voxel-wise images using 5-DE TL")
                plotting.show()


            # Get the ROI of Neuromorphometrics with highest covariances within the voxel-wise image
            tree = ET.parse(VBMLOOKUP_FILE)
            root = tree.getroot()
            labels_to_index_roi = {}
            index_to_label_roi = {}
            # Find the 'data' section where ROI labels and their indices are stored
            data_section = root.find('data')
            # Iterate through each 'label' element within the 'data' section
            for label in data_section.findall('label'):
                index = label.find('index').text  # Get the text of the 'index' element
                name = label.find('name').text    # Get the text of the 'name' element
                labels_to_index_roi[name] = int(index)         # Add to dictionary
                index_to_label_roi[int(index)]=name
            # print(labels_to_index_roi)
            # neuromorphometrics vol file read
            ref_im = nibabel.load(VOL_FILE_VBM)
            ref_arr = np.array(ref_im.get_fdata())
            img_array = np.array(image.get_fdata())
            print(type(image))
            print(type(img_array), np.shape(img_array))

            # we set to zero all voxels under the threshold
            img_array[np.abs(img_array) < np.percentile(np.abs(data_masked), 95)] = 0
            
            labels = list(set(labels_to_index_roi.values()))
            print("there are ",len(labels), " labels")

            data = {"name": [], "cov": [], "ratio": [], "nb_voxels":[], "percentage_pos": [], "percentage_neg": []}
            for label in labels:
                # Find coordinates of all points in the reference array (MNI space) that match 'label'
                points_ref = np.asarray(np.where(ref_arr == label)).T
                # mean of voxels for current subject within one ROI
                gm_data = np.asarray([img_array[loc[0], loc[1], loc[2]] for loc in points_ref])
                ratio = np.count_nonzero(gm_data) * 100 / len(gm_data)
                gm_data[gm_data == 0] = np.nan
                # do not include ROI that have only 0 as covariates of their composing voxels
                all_nan = np.isnan(gm_data).all()
                if not all_nan: 
                    gm = np.nanmean(np.abs(gm_data))
                    percen_pos = 100*np.sum(gm_data[~np.isnan(gm_data)]>0)/(np.sum(gm_data[~np.isnan(gm_data)]>0)+np.sum(gm_data[~np.isnan(gm_data)]<0))
                    percen_neg = 100*np.sum(gm_data[~np.isnan(gm_data)]<0)/(np.sum(gm_data[~np.isnan(gm_data)]>0)+np.sum(gm_data[~np.isnan(gm_data)]<0))
                    data["name"].append(index_to_label_roi[label])
                    if percen_neg>percen_pos  : gm = -gm
                    data["cov"].append(gm)
                    data["ratio"].append(ratio)
                    data["nb_voxels"].append(len(gm_data[~np.isnan(gm_data)]))
                    data["percentage_pos"].append(percen_pos)
                    data["percentage_neg"].append(percen_neg)

            df = pd.DataFrame.from_dict(data)
            print(df)
            df_sorted = df.loc[df['cov'].abs().sort_values(ascending=False).index]
            print("top 20 absolute values in cov: ")
            print(df_sorted.head(20))

        



"""
SVM-RBF SHAP VALUES ANALYSIS
after having computed the SHAP values with LOSO-CV folds in classif_VBMROI.py or classif_SBMROI.py (for VBM ROI or SBM ROI features, respectively),
 and after having computed the SHAP values with LOSO-CV folds 30 times with permuted labels, we can start analyzing 
 the statistical significance of each feature of importance : 

STEP 1 : analysis of shap values obtained in steps 1 and 2 with confidence intervals thanks to the random permutation estimates 
        read_bootstrapped_shap()

STEP 4 : merge results with univariate statistics --> for the ROI with shap values within the confidence intervals, separate those with 
            a significant p-value related to Li response (from an OLS regression ROI ~ response + age + sex + site) ("specific" ROI) 
            from those with a p-value > 0.05 ("suppressor" ROI)

STEP 5 : fit a linear regression to the specific and suppressor ROI only on Li response, and compute regression scores separately for
            suppressor and specific ROI.

STEP 6 : plot regression results based on labels and type of ROI/feature (suppressor/specific), and plot the distribution of scores with 
            both suppressor and specific ROI (still from the pool of significant shap (defined by the ci computed in step 3), and the pre-selection
            of suppressor and specific ROI dependent on the p-value thresholds chosen)

STEP 7 : plot beeswarm plot of shap values in order of highest to lowest mean absolute shap for the ROI that were selected as specific ROI in the 
            previous steps with plot_beeswarm()

"""

def main():
    # plot_beeswarm(VBM=True)
    # read_bootstrapped_shap(save=True,VBM=True)
    # plot_beeswarm(VBM=True)
    # plot_glassbrain()
    # quit()
    # forward_maps_voxelwise()
    # quit()

    regression_analysis_with_specific_and_suppressor_ROI(VBM=True,plot_and_save_jointplot=True, plot_and_save_kde_plot=True)
    quit()
    plot_beeswarm()
    plot_glassbrain()

    # make_shap_df()
    # read_bootstrapped_shap(save=True,plot_and_save_jointplot=True, plot_and_save_kde_plot=False)
    
   
if __name__ == "__main__":
    main()
