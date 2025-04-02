import sys, os, time, scipy
import numpy as np
import nibabel
import pandas as pd
import seaborn as sns
import nibabel as nib
import ast, re
import shap
import xml.etree.ElementTree as ET
import nilearn.plotting as plotting

from utils import get_predict_sites_list, read_pkl, get_LOSO_CV_splits_N861, get_participants, save_pkl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from univariate_stats import get_scaled_data
from sklearn.linear_model import LogisticRegressionCV
from classif_VBMROI import remove_zeros 

ROOT = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
SHAP_DIR_SVMRBF=ROOT+"/models/ShapValues/shap_computed_from_all_Xtrain/"
DATAFOLDER=ROOT+"data/processed/"
RESULTS_FEATIMPTCE_AND_STATS_DIR=ROOT+"results_feat_imptce_and_univ_stats/"
VOL_FILE_VBM = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"
VBMLOOKUP_FILE = "/drf/local/spm12/tpm/labels_Neuromorphometrics.xml"

def make_shap_df(verbose=False):
    # SHAP only computed for VBM ROI and age+sex+site residualization

    VBMdf = pd.read_csv(DATAFOLDER+"VBMROI_Neuromorphometrics.csv")
    exclude_elements = ['participant_id', 'session', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']
    VBMdf = VBMdf.drop(columns=exclude_elements)
    list_roi = list(VBMdf.columns)
    columns_with_only_zeros = VBMdf.columns[(VBMdf == 0).all()]
    list_roi = [ roi for roi in list_roi if roi not in columns_with_only_zeros]
    assert len(list_roi)==280
    columns_with_zeros_indices = [VBMdf.columns.get_loc(col) for col in columns_with_only_zeros]
    
    df_all_shap = pd.DataFrame(columns=['fold', 'ROI', 'mean_abs_shap','shap_array', 'mean_abs_shap_rdm','shap_array_rdm'])

    shap_all, shap_all_rdm = [], []

    for site in get_predict_sites_list():
        print(site)
        
        shap_path = SHAP_DIR_SVMRBF + "ShapValues_VBM_SVM_RBF_"+site+"_background_alltr_parallelized.pkl"
        all_runs_shap_rdm = []

        for run in range(1, 31): # from 1 to 30 included
            shap_path_rdm = SHAP_DIR_SVMRBF + "ShapValues_VBM_SVM_RBF_"+site+"_background_alltr_parallelized_randomized_labels_run"+str(run)+".pkl"
            shap_rdm = read_pkl(shap_path_rdm)
            shap_rdm = np.array([exp.values for exp in shap_rdm])
            shap_rdm = np.delete(shap_rdm, columns_with_zeros_indices, axis=1)
            assert shap_rdm.shape[1]==280 # 280 ROI (we removed the ROI that had all values across subjects equal to 0 (their shap values are also zeros))
            # print(type(shap_rdm), np.shape(shap_rdm))
            all_runs_shap_rdm.append(shap_rdm)

        all_runs_shap_rdm = np.concatenate(all_runs_shap_rdm, axis=0)
        assert all_runs_shap_rdm.shape[1]==280
        if verbose : print("all_runs_shap_rdm ",all_runs_shap_rdm.shape)
        shap_all_rdm.append(all_runs_shap_rdm)

        mean_abs_shap_rdm = np.mean(np.abs(all_runs_shap_rdm), axis=0) # mean over test set subjects x 30 runs with permuted labels : shape = (nb roi,) = (280,)

        shap_values = read_pkl(shap_path)
        shap_values = np.array([exp.values for exp in shap_values])
        assert shap_values.shape[0]*30==all_runs_shap_rdm.shape[0]
        assert shap_values.shape[1]==280 # the four ROIs with zero values have been removed before computing the shap values 
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
    df_all_shap.to_excel(SHAP_DIR_SVMRBF+'SHAP_summary_including_shap_from_30rdm_runs.xlsx', index=False)

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

def read_bootstrapped_shap(save=False):
    # SHAP only computed for VBM ROI and age+sex+site residualization

    # get univariate statistics (obtained with univariate_stats.py)
    path_univ_statistics = RESULTS_FEATIMPTCE_AND_STATS_DIR+"statsuniv_rois_res_age_sex_site.xlsx"
    univ_statistics = pd.read_excel(path_univ_statistics)

    path_shap_summary = SHAP_DIR_SVMRBF+'SHAP_summary_including_shap_from_30rdm_runs.xlsx'

    if not os.path.exists(path_shap_summary) :
        df_all_shap = make_shap_df()
    else : df_all_shap = pd.read_excel(path_shap_summary)

    # retrieve all test set ROI values for all folds
    splits = get_LOSO_CV_splits_N861()   
    # read participants dataframe
    participants = get_participants()

    Nmax=800 # using maximum training set size for dataset with all VBM-preprocessed subjects, N861 
    VBMdf = pd.read_csv(DATAFOLDER+"VBMROI_Neuromorphometrics.csv")
    participants_all = list(splits["Baltimore-"+str(Nmax)][0])+list(splits["Baltimore-"+str(Nmax)][1])
    msk = list(participants[participants['participant_id'].isin(participants_all)].index)
    participants_VBM = participants.iloc[msk]   
    participants_VBM = participants_VBM.reset_index(drop=True)
    # reorder VBMdf to have rows in the same order as participants_VBM
    VBMdf = VBMdf.set_index('participant_id').reindex(participants_VBM["participant_id"].values).reset_index()

    # create concatenation of all test sets participant's ROIs in Xte_arr np array
    Xte_list, list_rois = [], []

    for site in get_predict_sites_list():
        df_te_ = VBMdf[VBMdf["participant_id"].isin(splits[site+"-"+str(Nmax)][1])]
        exclude_elements = ['participant_id', 'session', 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']
        df_te_ = df_te_.drop(columns=exclude_elements)
        df_te_ = remove_zeros(df_te_) # remove ROIs with zero values for all subjects
        Xte_list.append(df_te_.values)
        assert np.shape(df_te_.values)[1]==280
        if site=="Baltimore": list_rois = list(df_te_.columns)

    all_folds_Xtest_concatenated = np.concatenate(Xte_list,axis=0)
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

        shap_arr_one_fold = read_pkl(SHAP_DIR_SVMRBF + "ShapValues_VBM_SVM_RBF_"+site+"_background_alltr_parallelized.pkl")
        shap_arr_one_fold = np.array([exp.values for exp in shap_arr_one_fold])
        concatenated_shap_arrays.append(shap_arr_one_fold)

        # compute mean, standard deviation, and count of mean absolute shap values for testing set of current fold
        m = shap_df_stats_current_site["mean_abs_shap"]
        s = shap_df_stats_current_site["std_abs_shap"]
        n = shap_df_stats_current_site["count_test_subjects"]
        assert len(m)==len(s) and len(s)==len(n) and len(n)==280

        # Critical value for t at alpha / 2:
        t_alpha2 = -scipy.stats.t.ppf(q=0.05/2, df=n-1, loc=0)
        ci_low = m - t_alpha2 * s / np.sqrt(n)
        ci_high = m + t_alpha2 * s / np.sqrt(n)
        # mean_abs_shap_rdm are the means of the absolute shap values for all test set subjects between the 30 runs for current LOSO-CV fold
        shap_rnd_absmean = shap_df_stats_current_site[["mean_abs_shap_rdm"]].copy()

        # Convert to a single-row DataFrame with ROIs as columns
        shap_rnd_absmean = shap_rnd_absmean.drop_duplicates().T
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
            if site =="Baltimore": assert shap_stat.shape[0] == 140 # check that we have 140 SHAP where H0 is rejected
            shap_stat["type"] = None # initialize empty column
            shap_stat.loc[shap_stat.diag_pcor_bonferroni < 0.05, "type"] = "specific" 
            shap_stat.loc[shap_stat.diag_pcor_bonferroni > 0.05, "type"] = "suppressor"

            if save:
                shap_stat.to_excel(RESULTS_FEATIMPTCE_AND_STATS_DIR+"shap_from_SVM_RBF_"+site+"-univstat.xlsx", sheet_name='SHAP_roi_univstat_'+site, index=False)
        
        specific_ROI[site]=shap_stat[shap_stat["type"]=="specific"]["ROI"].values
        suppressor_ROI[site]=shap_stat[shap_stat["type"]=="suppressor"]["ROI"].values

    shared_strings_spec = get_shared_specific_or_suppressor_ROIs_btw_folds(specific_ROI)
    shared_strings_supp = get_shared_specific_or_suppressor_ROIs_btw_folds(suppressor_ROI)  

    # lists of specific and suppressor ROIs shared across all 12 folds
    shap_spec = list(shared_strings_spec)
    shap_suppr = list(shared_strings_supp)

    ################# SAVE SUMMARY VALUES FOR ANALYSIS OF SHAP VALUES ######################  
    concatenated_shap_arrays=np.concatenate(concatenated_shap_arrays, axis=0)
    assert concatenated_shap_arrays.shape == all_folds_Xtest_concatenated.shape, "Shape mismatch between shap array and testing data array!"
    indices_specificROI = [list_rois.index(roi) for roi in shap_spec if roi in list_rois]
    indices_suppressorROI = [list_rois.index(roi) for roi in shap_suppr if roi in list_rois]

    # Negate SHAP values for ROIs containing 'CSF_Vol' --> CSF volume varies inversely from GM volume
    csf_indices = [i for i, roi in enumerate(list_rois) if 'CSF_Vol' in roi]
    concatenated_shap_arrays_negatedCSF = concatenated_shap_arrays.copy()
    concatenated_shap_arrays_negatedCSF[:, csf_indices] *= -1

    # Order specific ROIs by mean absolute SHAP values
    mean_abs = np.mean(np.abs(concatenated_shap_arrays_negatedCSF[:,indices_specificROI]),axis=0)
    ordered_indices = np.argsort(mean_abs)[::-1]  # [::-1] reverses the order to get descending
    sorted_indices_specificROI = np.array(indices_specificROI)[ordered_indices]

    # get dictionary of correspondecies of ROI names from abbervations to full ROI names
    atlas_df = pd.read_csv(ROOT+"data/atlases/lobes_Neuromorphometrics.csv", sep=';')
    dict_atlas_roi_names = atlas_df.set_index('ROIabbr')['ROIname'].to_dict()

    # save info necessary for summary plot of SHAP values for specific and suppressor ROI
    dict_summary = {"dict_atlas_roi_names":dict_atlas_roi_names, "indices_specific_roi":indices_specificROI,"indices_suppressorROI":indices_suppressorROI,\
        "sorted_indices_specificROI":sorted_indices_specificROI, "all_folds_Xtest_concatenated":all_folds_Xtest_concatenated,"list_rois":list_rois,\
         "concatenated_shap_arrays_with_negated_CSF":concatenated_shap_arrays_negatedCSF, "concatenated_shap_arrays":concatenated_shap_arrays}
    save_pkl(dict_summary, RESULTS_FEATIMPTCE_AND_STATS_DIR+"ShapSummaryDictionnaryForBeeswarmPlot.pkl")

def plot_glassbrain(): 
    """
        Aim : plot glassbrain of specfic ROI from SHAP values obtained with an SVM-RBF and VBM ROI features
    
    """
    specific_roi_df = read_pkl(RESULTS_FEATIMPTCE_AND_STATS_DIR+"specific_ROI_SHAP_SVMRBF_VBM.pkl")
    specific_ROI_dict = specific_roi_df.set_index('ROI')['mean_abs_shap'].to_dict()
    print(specific_roi_df)
    print(specific_ROI_dict)

    ref_im = nibabel.load(VOL_FILE_VBM)
    ref_arr = ref_im.get_fdata()
    labels = sorted(set(np.unique(ref_arr).astype(int))- {0}) # 136 labels --> 'Left Inf Lat Vent', 'Right vessel', 'Left vessel' missing in data
    atlas_df = pd.read_csv(ROOT+"data/atlases/lobes_Neuromorphometrics.csv", sep=';')
    texture_arr = np.zeros(ref_arr.shape, dtype=float)
    
    
    for name, val in specific_ROI_dict.items():
        # get GM volume (there is one row for GM and another for CSF for each ROI but the ROIbaseid values are the same for both so we picked GM vol)
        baseids = atlas_df[(atlas_df['ROIname'] == name) & (atlas_df['volume'] == 'GM')]["ROIbaseid"].values
        int_list = list(map(int, re.findall(r'\d+', baseids[0])))
        if "Left" in name: 
            if len(int_list)==2: baseid = int_list[1]
            else : baseid = int_list[0]
        else : baseid = int_list[0]
        if name in ["Left Pallidum", "Right Pallidum"]: texture_arr[ref_arr == baseid] = val
        else : texture_arr[ref_arr == baseid] = - val

    print("nb unique vals :",len(np.unique(texture_arr)))
    print(np.shape(texture_arr))

    cmap = plt.cm.coolwarm
    vmin = np.min(texture_arr)
    vmax = np.max(texture_arr)
    print("vmin vmax texture arr", vmin,"     ",vmax)
    texture_im = nibabel.Nifti1Image(texture_arr, ref_im.affine)

    plotting.plot_glass_brain(
        texture_im,
        display_mode="ortho",
        colorbar=True,
        cmap=cmap,
        plot_abs=False ,
        alpha = 0.95 ,
        threshold=0,
        title="glassbrain of specific ROI")
    plotting.show()


def plot_beeswarm():
    """
        Aim: printing the mean absolute shap values and their feature's values for all shap values of all 861 test subjects 
            (concatenation of test subjects' feature values and SHAP values for the 12 folds), only for ROIs/features previously 
            found to be 'specific' (ie. have a (hypothetically) direct impact on BD diagnosis).
    """

    dict_summary= read_pkl(RESULTS_FEATIMPTCE_AND_STATS_DIR+"ShapSummaryDictionnaryForBeeswarmPlot.pkl")
    
    concatenated_shap_arrays_negatedCSF = dict_summary["concatenated_shap_arrays_with_negated_CSF"]
    sorted_indices_specificROI = dict_summary["sorted_indices_specificROI"]
    all_folds_Xtest_concatenated = dict_summary["all_folds_Xtest_concatenated"]
    dict_atlas_roi_names = dict_summary["dict_atlas_roi_names"]
    list_rois = dict_summary["list_rois"]
    mean_abs_specific = np.mean(np.abs(concatenated_shap_arrays_negatedCSF[:,sorted_indices_specificROI]),axis=0)

    # SHAP summary plot
    # shap.summary_plot(concatenated_shap_arrays_negatedCSF[:,sorted_indices_specificROI], all_folds_Xtest_concatenated[:,sorted_indices_specificROI], \
    #                   feature_names=[dict_atlas_roi_names[list_rois[i]] for i in sorted_indices_specificROI], max_display=len(sorted_indices_specificROI))

    print("Sorted ROI (highest to lowest mean_abs):", [list_rois[i] for i in sorted_indices_specificROI])
    # get univariate statistics (obtained with univariate_stats.py)
    path_univ_statistics = RESULTS_FEATIMPTCE_AND_STATS_DIR+"statsuniv_rois_res_age_sex_site.xlsx"
    univ_statistics = pd.read_excel(path_univ_statistics)
    stats_specific_ROI = univ_statistics[univ_statistics["ROI"].isin([list_rois[i] for i in sorted_indices_specificROI])].copy()
    recap = stats_specific_ROI[stats_specific_ROI["diag_pcor_bonferroni"] < 0.05][["ROI", "diag_pcor_bonferroni"]]
    recap["mean_abs_shap"] = np.round(mean_abs_specific,4)

    recap["ROI"] = recap['ROI'].replace(dict_atlas_roi_names)
    save_pkl(recap, RESULTS_FEATIMPTCE_AND_STATS_DIR+"specific_ROI_SHAP_SVMRBF_VBM.pkl")

    print("p-values after bonferroni correction and mean abs shap values for specific ROIs :\n", recap)

    # Illustrate Suppressor variable with linear model

        
#     data = get_scaled_data()
#     data["response"] = data["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
#     print(data)

#     print("nb of specific ROI : ", len(list(shap_spec.ROI)), " nb of suppressor ROI : ",len(list(shap_suppr.ROI)))
#     # 26 specific ROIs, 24 suppressor ROIs
#     X = data[list(shap_spec.ROI) + list(shap_suppr.ROI)].values
#     y = data.response
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)


#     lr = LogisticRegressionCV()
#     lr.fit(X_scaled, y)
#     assert lr.coef_.shape[1] == len(shap_spec.ROI) + len(shap_suppr.ROI) #lr.coef_ shape (1,83)

#     coef_spec = lr.coef_[0, :len(shap_spec.ROI)] # coefficients of specific ROI
#     coef_supr = lr.coef_[0, len(shap_spec.ROI):] # coefficients of supressor ROI

#     # compute scores : linear combination of ROI with their respective weights
#     score_spec = np.dot(X_scaled[:, :len(shap_spec.ROI)], coef_spec) 
#     score_supr = np.dot(X_scaled[:, len(shap_spec.ROI):], coef_supr)
#     score_tot = np.dot(X_scaled, lr.coef_[0, :]) # score tot being the score with both suppressor + specific ROIs, but not *all* ROIs

#     score_spec_auc = roc_auc_score(y, score_spec)
#     score_supr_auc = roc_auc_score(y, score_supr)
#     score_tot_auc = roc_auc_score(y, score_tot)

#     df = pd.DataFrame(dict(response=data.y, score_spec=score_spec,
#                         score_supr=score_supr, score_tot=score_tot))
    
#     ####### PRINT INFO ON OUTLIER SUBJECTS (defined as being 3 stds from the distribution mean) #############
#     df_ROI_age_sex_site = pd.read_csv(ROOT+"df_ROI_age_sex_site_fevrier2025_M00_labels_as_strings.csv")
#     df_outliers=df.copy()
#     df_outliers["participant_id"]=df_ROI_age_sex_site["participant_id"]
#     zscores = np.abs(zscore(df_outliers[["score_supr", "score_spec"]]))
#     outliers = np.where(zscores > 3)

#     outlier_indices = outliers[0]
#     df_outliers = df_outliers.iloc[outlier_indices]
#     print("outliers:\n",df_outliers,"\n")
#     #########################################################################################################


#     if plot_and_save_jointplot: 
#         # Create a joint plot with scatter and marginal density plots
#         plt.figure(figsize=(8, 8))
#         g = sns.jointplot(data=df, x="score_supr", y="score_spec", hue="response")
#         g.ax_joint.set_xlabel("Score with Suppressor Features (AUC=%.2f)" % score_supr_auc, fontsize = 20)
#         g.ax_joint.set_ylabel("Score with Speficic Features  (AUC=%.2f)" % score_spec_auc, fontsize = 20)

#         # Increase font size for legend title and labels and ticks
#         legend = g.ax_joint.legend_
#         legend.set_title("Response", prop={'size': 20})  
#         for text in legend.get_texts():
#             text.set_fontsize(20)  
#         g.ax_joint.tick_params(axis='both', labelsize=16)

#         g.figure.suptitle("Scatter Plot with Marginal Densities", y=0.98, fontsize=22)
#         plt.show()
#         plt.savefig(RESULTS_FEATIMPTCE_DIR + "plot_suppressor-specific_scatter_bootstrap"+str(nb_samplings)+"_without_multiple_comp_corr.pdf")  

#     if plot_and_save_kde_plot:
#         # Density (KDE) Plot
#         plt.figure(figsize=(6, 2))
#         g = sns.kdeplot(data=df, x='score_tot', hue='response', fill=True, alpha=0.3)
#         g.set_xlabel("Density Plot of Score with all Features (AUC=%.2f)" % score_tot_auc, fontsize=20)
#         g.set_ylabel("")
#         # Increase x and y tick size
#         plt.xticks(fontsize=16)
#         plt.yticks(fontsize=16)
#         # Increase legend font size
#         legend = g.legend_
#         legend.set_title("Response", prop={'size': 20})  # Set legend title font size
#         for text in legend.get_texts():
#             text.set_fontsize(20)  # Set font size of legend labels
#         plt.grid(True)
#         plt.show()
#         plt.savefig(RESULTS_FEATIMPTCE_DIR + "plot_suppressor-specific_density_bootstrap"+str(nb_samplings)+"_without_multiple_comp_corr.pdf")  



"""
SVM-RBF SHAP VALUES ANALYSIS
after having computed the SHAP values with LOSO-CV folds, as well as 30 times with permuted labels, start analysing the SHAP values and their 
statistical significance : 

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
    plot_beeswarm()

    plot_glassbrain()
    # path_univ_statistics = RESULTS_FEATIMPTCE_AND_STATS_DIR+"statsuniv_rois_res_age_sex_site.xlsx"
    # univ_statistics = pd.read_excel(path_univ_statistics)
    # print(univ_statistics[[univ_statistics["diag_pcor_bonferroni"] < 0.05]])
    # make_shap_df()
    # read_bootstrapped_shap(save=True,plot_and_save_jointplot=True, plot_and_save_kde_plot=False)
    
   
if __name__ == "__main__":
    main()