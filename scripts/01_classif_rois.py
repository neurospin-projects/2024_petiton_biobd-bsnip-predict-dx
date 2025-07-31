import pandas as pd, numpy as np
import sys, os

sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer
from nitk.ml_utils.residualization import get_residualizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


from utils import get_classifier, get_scores_pipeline, get_predict_sites_list, read_pkl, save_pkl, \
    create_folder_if_not_exists


# inputs
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
DATA_DIR=ROOT+"data/processed/"
ROI_VBM_FILE = DATA_DIR+"VBM_ROI_allfolds_280725.csv"
ROI_SBM_FILE_Destrieux = DATA_DIR+"SBM_ROI_Destrieux_allfolds_280725.csv"
ROI_SBM_FILE_Desikan = DATA_DIR+"SBM_ROI_Desikan_allfolds_280725.csv"
ROI_SBM_FILE_Destrieux_TIV_scaled = DATA_DIR+"SBM_ROI_Destrieux_allfolds_TIV_scaled_280725.csv"
ROI_SBM_FILE_Desikan_TIVscaled = DATA_DIR+"SBM_ROI_Desikan_allfolds_TIV_scaled_280725.csv"
seven_subcortical_roi_sbm = ['Left-Thalamus', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum', \
                             'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area', 'Right-Thalamus',\
                                  'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', \
                                    'Right-Amygdala', 'Right-Accumbens-area']

all_subcortical_rois_sbm = seven_subcortical_roi_sbm + ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent',
                              '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-VentralDC', 'Left-vessel',
                               'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-VentralDC',
                                 'Right-vessel', 'Right-choroid-plexus', '5th-Ventricle', 'Optic-Chiasm', 'CC_Posterior', 'CC_Mid_Posterior', 
                                 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior']

# outputs
RESULTSFOLDER=ROOT+"results_classif/"
RESULTSFOLDER_SBM=RESULTSFOLDER+"classifSBM/"
RESULTSFOLDER_VBM=RESULTSFOLDER+"classifVBM/"

def remove_zeros(df, verbose=False):
        columns_with_only_zeros = df.columns[(df == 0).all()]
        if verbose : print("columns with zeros ",columns_with_only_zeros)
        column_indices = [df.columns.get_loc(col) for col in columns_with_only_zeros]
        assert set(columns_with_only_zeros) <= {'lInfLatVen_GM_Vol', 'lOC_GM_Vol', 'lInfLatVen_CSF_Vol', 'lOC_CSF_Vol'}
        df = df.drop(columns=columns_with_only_zeros)
        return df

def cov_diff(X, groups):
    unique_g = np.unique(groups)
    covs = []
    for g in unique_g:
        covs.append(np.cov(X[groups==g].T))
    return covs

def classify(dfroi, list_rois, datasize_idx, classif_name="svm", formula_res = "age+sex+site", N763=False):
    """
        datasize_idx(int): 0 to 8 included, corresponds to the training dataset size index.
                            index 0 corresponds to roughly 100 subjects per train set, index 1 to about 175 subjects, ...
                            and index 8 corresponds to training with a Leave-One-Site-Out CV scheme using all available data
        classif_name (str): classifier name

        formula_res (str): residualization scheme
        N763 (bool): if True, use N=763 subjects (those that have data that has been processed for both VBM and SBM)
    """

    assert formula_res in [None, "age+sex+site","age+sex"]
    assert classif_name in ["svm","EN","MLP","xgboost","L2LR"],"wrong classifier name"

    sites = dfroi["site"].unique()
    
    results_dict = {}
    for site in sites:
        # Test set = all from test site
        """
        # how to test on only test set from N763 when training can be done on N861 train set
        # useful for meta-model
        dfroi_metamodel = dfroi[dfroi["SBMandVBM"]]
        test_df = dfroi_metamodel[dfroi_metamodel["site"]==site].copy()
        """
        test_df = dfroi[dfroi["site"] == site].copy()

        # Training set = subset defined by datasize_idx for this test site
        if datasize_idx != 8: colname = f"lrn_curv_x{datasize_idx}_{site}"
        else : colname = f"lrn_curv_x{datasize_idx}"

        train_df = dfroi[dfroi[colname]].copy()
        train_df = train_df[train_df["site"] != site]

        # combined df for residualization
        combined_df = pd.concat([train_df, test_df], axis=0)

        # Features and labels
        X_train = train_df[list_rois].values
        y_train = train_df["dx"].values
        X_test = test_df[list_rois].values
        y_test = test_df["dx"].values
        participants_train = train_df["participant_id"].values
        participants_test = test_df["participant_id"].values

        if formula_res:
            residualizer = Residualizer(data=combined_df[list_rois+["age","sex","site","dx"]], formula_res=formula_res, formula_full=formula_res+"+dx")
            Zres = residualizer.get_design_mat(combined_df[list_rois+["age","sex","site","dx"]])

            residualizer.fit(X_train, Zres[:len(X_train)])
            X_train = residualizer.transform(X_train, Zres[:len(X_train)])
            X_test = residualizer.transform(X_test, Zres[len(X_train):])

        """
        cov_by_site = cov_diff(X_train, train_df["site"].values)
        cov_by_class = cov_diff(X_train, y_train)
        diff_class = np.linalg.norm(cov_by_class[0] - cov_by_class[1], ord='fro')
        n_sites = len(cov_by_site)
        diffs = []
        import itertools
        for i, j in itertools.combinations(range(n_sites), 2):
            diff = np.linalg.norm(cov_by_site[i] - cov_by_site[j], ord='fro')
            diffs.append(diff)

        mean_diff_site = np.mean(diffs)
        print("mean_diff_site ",mean_diff_site, " diff_class ", diff_class)

        prints: 
        mean_diff_site  36.674042717010956  diff_class  20.1140720118544
        """

        # Pipeline
        pipe = make_pipeline(
            StandardScaler(),
            get_classifier(classif_name)
        )

        # Fit on train, predict on test
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        # get classification scores for current classifier
        score_test = get_scores_pipeline(pipe, X_test)
        score_train = get_scores_pipeline(pipe, X_train)

        roc_auc = roc_auc_score(y_test, score_test)
        bacc = balanced_accuracy_score(y_test, y_pred)

        results_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc,"score test": score_test,"score train": score_train, \
                                    "participant_ids_te":participants_test, "participant_ids_tr":participants_train,\
                                        "y_train": y_train, "y_test":y_test}

    print("AUC per site:", [results_dict[site]["roc_auc"] for site in sites])
    results_dict["mean over all sites"] = {"roc_auc": np.mean([results_dict[site]["roc_auc"] for site in sites]),\
                                            "balanced-accuracy":np.mean([results_dict[site]["balanced-accuracy"] for site in sites])}
    print("Mean AUC:", np.mean([results_dict[site]["roc_auc"] for site in sites]))

    return results_dict

def run_loso_cv_roi(datasize_idx, VBM=False, SBM=False, classif_name="svm", formula_res = "age+sex+site", N763=False,\
                    atlas="Destrieux", all_subcortical_rois=False, seven_subcortical_rois=False, TIVscaled=True, save=False, dataidx=False):
    """
        datasize_idx(int): 0 to 8 included, corresponds to the training dataset size index.
                            index 0 corresponds to roughly 100 subjects per train set, index 1 to about 175 subjects, ...
                            and index 8 corresponds to training with a Leave-One-Site-Out CV scheme using all available data
        classif_name (str): classifier name

        formula_res (str): residualization scheme
        N763 (bool): if True, use N=763 subjects (those that have data that has been processed for both VBM and SBM)
        TIVscaled (bool): applicable to SBM roi only (whether the ROI are scaled with regards to TIV),
                            all VBM roi are scaled with TIV (so that TIV=1500.0)
        save (bool) : save results file or not
        dataidx (bool) : when saving, save the dataidx instead of the approximate data size
    
    subcortical_rois are optional for SBM roi
    ROIs being scaled by TIV is optional for SBM roi
    atlas = "Destrieux" or "Desikan" for SBM roi (always Neuromorphometrics for VBM roi)
    """
    assert atlas in ["Destrieux","Desikan","Neuromorphometrics"], "wrong atlas name"
    assert VBM or SBM,"it is required to choose between VBM or SBM roi"
    assert not (VBM and SBM),"it is required to choose between VBM or SBM roi"

    if VBM :
        dfroi = pd.read_csv(ROI_VBM_FILE)
        dfroi = remove_zeros(dfroi) # remove rois with values equal to zeros across all participants
        list_rois_ = [r for r in dfroi.columns if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]
        assert len(list_rois_)==280, "there should be 280 rois"
        if N763: dfroi = dfroi[dfroi["SBMandVBM"]] # select participants with both preprocessings
        
        results_dict = classify(dfroi, list_rois_, datasize_idx, classif_name=classif_name, formula_res = formula_res, N763=False)

        if not N763: datasizes = [100,175,250,350,400,500,600,700,800]
        else : datasizes = [75, 150, 200, 300, 400, 450, 500, 600, 700]
        if dataidx: 
            """
            # for meta-model (change saving directory and specify "vbm_roi" in file name)
            folder=ROOT+"results_classif/meta_model/"
            results_file = folder+"scores_tr_te_N861_train_size_N"+str(datasize_idx)+"_vbmroi.pkl"
            """
            results_file = RESULTSFOLDER_VBM+"scores_tr_te_N861_train_size_N"+str(datasize_idx)+".pkl"
        else : results_file = RESULTSFOLDER_VBM+"scores_tr_te_N861_train_size_N"+str(datasizes[datasize_idx])+".pkl"
        print(results_file)

    if SBM: 
        N763=True
        assert atlas!="Neuromorphometrics","VBM atlas, not for SBM roi"

        if not TIVscaled:
                filepath = ROI_SBM_FILE_Destrieux if atlas=="Destrieux" else ROI_SBM_FILE_Desikan
        else: 
            filepath = ROI_SBM_FILE_Destrieux_TIV_scaled if atlas=="Destrieux" else ROI_SBM_FILE_Desikan_TIVscaled

        dfroi = pd.read_csv(filepath)

        list_cols = list(dfroi.columns)
        # print(list_cols)

        if not all_subcortical_rois:
            list_cols = list(dfroi.columns)
            # case in which we have no subcortical rois at all
            if not seven_subcortical_rois: dfroi = dfroi[[r for r in list_cols if r not in all_subcortical_rois_sbm]]
            # case in which we have 7 subcortical rois per hemisphere (replication of Nunes et al)
            else: 
                subcortical_rois_not_in_list_of_seven = [r for r in all_subcortical_rois_sbm if r not in seven_subcortical_roi_sbm]
                dfroi = dfroi[[r for r in list_cols if r not in subcortical_rois_not_in_list_of_seven]]
            list_cols = list(dfroi.columns)

        list_lrn_curv_splits = ["lrn_curv_x"+str(i)+"_"+site for i in range(8) for site in get_predict_sites_list()]
        list_lrn_curv_splits = list_lrn_curv_splits+["lrn_curv_x8"]
        other_elements = ["participant_id","session","age","sex","site","TIV","dx"]+list_lrn_curv_splits
        list_rois_ = [r for r in list_cols if r not in other_elements]
        
        added_sub=0

        if all_subcortical_rois: added_sub = 34
        if seven_subcortical_rois: added_sub = 14 
        if atlas=="Destrieux": assert len(list_rois_) == 296 + added_sub # 74 for each ROI type (area and cortical thickness) for both hemispheres
        if atlas=="Desikan": assert len(list_rois_) == 136 + added_sub # 34 for each ROI type (area and cortical thickness) for both hemispheres
              
        results_dict = classify(dfroi, list_rois_, datasize_idx, classif_name=classif_name, formula_res = formula_res, N763=False)
        if not (all_subcortical_rois or seven_subcortical_rois): 
            strsub = "_no_subcortical"
            folder = "no_subcortical/"
        else: 
            if seven_subcortical_rois: 
                strsub = "_7subROI"
                folder = "with7subcorticalROI/"
            else : 
                strsub = ""
                folder = ""
        if dataidx: results_file = RESULTSFOLDER_SBM+folder+str(classif_name)+"_N"+str(datasize_idx)+"_"+atlas+"_SBM_ROI"+strsub+"_N763.pkl"
        else:
            datasizes = [75, 150, 200, 300, 400, 450, 500, 600, 700]
            results_file = RESULTSFOLDER_SBM+folder+str(classif_name)+"_N"+str(datasizes[datasize_idx])+"_"+atlas+"_SBM_ROI"+strsub+"_N763.pkl"

    if save:
        print("\nsaving classification results ...")
        save_pkl(results_dict, results_file)


def main():
    
    for dataidx in [0,1,2,3,4,5,6,7,8]:
        run_loso_cv_roi(dataidx, VBM=True, SBM=False, classif_name="svm", formula_res = "age+sex+site", N763=False,\
                        atlas="Neuromorphometrics",save=True, dataidx=True)
    quit()
    run_loso_cv(8, classif_name="L2LR", N763=True, formula_res="age+sex+site")

    """
    Nmax=861 for SVM-RBF 
    MEAN : roc_auc  72.58 balanced-accuracy  65.98
    STD : roc_auc  8.03 balanced-accuracy  6.72

    Nmax=763 for SVM-RBF 
    MEAN : roc_auc  73.41 balanced-accuracy  68.71
    STD : roc_auc  8.06 balanced-accuracy  7.14
    """

    # GroupKFold with cv8
    # add shap computation in another def with GroupKFold if possible and svm only

if __name__ == "__main__":
    main()
