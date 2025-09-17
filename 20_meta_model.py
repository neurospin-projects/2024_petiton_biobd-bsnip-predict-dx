import numpy as np, pandas as pd
from utils import read_pkl, get_predict_sites_list, get_scores_pipeline, save_pkl

# for linear model (= scikit learn imports)
from sklearn.model_selection import GridSearchCV
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from scipy.stats import ttest_rel

# inputs
ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
LEARNING_CRV_TL_DIR = ROOT+"models/TL/TL_learningcurve/"

# ouputs
RESULTS_DICT = ROOT+"reports/results_classif/meta_model/summary_dict_results.pkl"

def classif_one_site(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr=None, ypred_voxelwise_vbm_te=None,\
    combination_nb=None):

    # check we have the same participant_id lists for train and test sets
    assert np.all (sbm_current_site["participant_ids_tr"] == vbm_current_site["participant_ids_tr"] )
    assert np.all (sbm_current_site["participant_ids_te"] == vbm_current_site["participant_ids_te"])

    sbm_score_tr = sbm_current_site["score train"]
    vbm_score_tr = vbm_current_site["score train"]
    assert len(sbm_score_tr)==len(vbm_score_tr) # more checks just in case

    sbm_score_te = sbm_current_site["score test"]
    vbm_score_te = vbm_current_site["score test"]
    assert len(sbm_score_te)==len(vbm_score_te)

    sbm_y_tr = sbm_current_site["y_train"]
    sbm_y_te = sbm_current_site["y_test"]

    assert np.all(sbm_y_tr == vbm_current_site["y_train"])
    assert np.all(sbm_y_te == vbm_current_site["y_test"])

    X_train = np.column_stack((sbm_score_tr, vbm_score_tr)) 
   
    if ypred_voxelwise_vbm_tr is not None:  X_train = np.column_stack((X_train, ypred_voxelwise_vbm_tr)) 
    y_train = sbm_y_tr

    # print("X_train and ytrain shapes : ", X_train.shape, "  ",y_train.shape) 
    X_test = np.column_stack((sbm_score_te, vbm_score_te))  
    if ypred_voxelwise_vbm_te is not None:  X_test = np.column_stack((X_test, ypred_voxelwise_vbm_te)) 

    y_test = sbm_y_te
    # print("X_test and ytest shapes : ",X_test.shape, "  ", y_test.shape) 

    # param_grid = {'C': [0.01, 0.1, 1., 10.], 'penalty': ['l2'],'class_weight':['balanced']}
    # classifier = GridSearchCV(estimator=lm.LogisticRegression(fit_intercept=False), param_grid=param_grid, cv=3)
    classifier = lm.LogisticRegression(C=0.01, class_weight='balanced', fit_intercept=False)

    # Pipeline
    pipe = make_pipeline(
        StandardScaler(),
        classifier
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # best_params = pipe.best_params_
    # print("Best Parameters:", best_params)

    # get classification scores for current classifier
    score_test = get_scores_pipeline(pipe, X_test)
    score_train = get_scores_pipeline(pipe, X_train)

    roc_auc = roc_auc_score(y_test, score_test)
    bacc = balanced_accuracy_score(y_test, y_pred)

    # print("roc auc for site ",site, ": ",roc_auc)

    results_dict[site] = {"roc_auc": roc_auc, "balanced-accuracy":bacc,"score test": score_test,"score train": score_train, \
                                "participant_ids_te":sbm_current_site["participant_ids_te"], "participant_ids_tr":sbm_current_site["participant_ids_tr"],\
                                    "y_train": y_train, "y_test":y_test}

    if ypred_voxelwise_vbm_tr is not None and ypred_voxelwise_vbm_te is not None and combination_nb:
        results_dict[site]["combination_nb"]=combination_nb

def run_meta_model(include_voxelwise_vbm=True, save=False):
    sites= get_predict_sites_list()

    for n in range(9):
        print("n : ", n)
        path = ROOT+"results_classif/meta_model/EN_N"+str(n)+"_Destrieux_SBM_ROI_subcortical_N763_data_imputation.pkl"
        sbm = read_pkl(path)

        pathvbm = ROOT+"results_classif/meta_model/scores_tr_te_N861_train_size_N"+str(n)+"_vbmroi.pkl"
        vbm = read_pkl(pathvbm)

        if include_voxelwise_vbm:
            dl_tl_5de_tr = read_pkl(LEARNING_CRV_TL_DIR+"reordered_to_fit_metamodel_mean_ypred_252_combinations_df_TL_densenet_Train_set.pkl")
            dl_tl_5de_te = read_pkl(LEARNING_CRV_TL_DIR+"reordered_to_fit_metamodel_mean_ypred_252_combinations_df_TL_densenet_Test_set.pkl")

        results_dict = {}

        if include_voxelwise_vbm:

            mean_auc_over_all_combinations, mean_bacc_over_all_combinations = [], []
            mean_auc_for_each_site = []
            mean_auc_for_each_site_sbm, mean_auc_for_each_site_vbm = [], []

            for combination_nb in range(1,253):
                # print("combination_nb ",combination_nb)
                for site in sites:
                    sbm_current_site = sbm[site]
                    vbm_current_site = vbm[site]
                    mask_tr = (
                        (dl_tl_5de_tr["site"] == site) &
                        (dl_tl_5de_tr["train_idx"] == n) &
                        (dl_tl_5de_tr["combination"] == combination_nb)
                    )

                    mask_te = (
                        (dl_tl_5de_te["site"] == site) &
                        (dl_tl_5de_te["train_idx"] == n) &
                        (dl_tl_5de_te["combination"] == combination_nb)
                    )

                    dl_tl_5de_tr_one_combination = dl_tl_5de_tr[mask_tr].copy()
                    dl_tl_5de_te_one_combination = dl_tl_5de_te[mask_te].copy()

                    ypred_voxelwise_vbm_tr = dl_tl_5de_tr_one_combination["mean_ypred"].values
                    ypred_voxelwise_vbm_te = dl_tl_5de_te_one_combination["mean_ypred"].values

                    classif_one_site(results_dict, site, sbm_current_site, vbm_current_site, ypred_voxelwise_vbm_tr[0], ypred_voxelwise_vbm_te[0], combination_nb)

                    if combination_nb==1:
                        sbm_y_te = sbm_current_site["y_test"]
                        mean_auc_for_each_site_sbm.append(roc_auc_score(sbm_y_te, sbm_current_site["score test"]))
                        mean_auc_for_each_site_vbm.append(roc_auc_score(sbm_y_te, vbm_current_site["score test"]))
                        
                results_dict["mean over all sites"] = {"roc_auc": np.mean([results_dict[site]["roc_auc"] for site in sites]),\
                                                    "balanced-accuracy":np.mean([results_dict[site]["balanced-accuracy"] for site in sites]),\
                                                    "combination_nb":combination_nb}
                
                mean_over_sites_auc = np.mean([results_dict[site]["roc_auc"] for site in sites])
                mean_over_sites_bacc = np.mean([results_dict[site]["balanced-accuracy"] for site in sites])

                mean_auc_for_each_site.append([results_dict[site]["roc_auc"] for site in sites])

                mean_auc_over_all_combinations.append(mean_over_sites_auc)
                mean_bacc_over_all_combinations.append(mean_over_sites_bacc)

                # print("Mean AUC:", mean_over_sites_auc)
                # print("Mean balanced-accuracy:",mean_over_sites_bacc)

            mean_auc_for_each_site = np.array(mean_auc_for_each_site)
            mean_auc_for_each_site = mean_auc_for_each_site.mean(axis=0)

            t_stat, p_value = ttest_rel(mean_auc_for_each_site, mean_auc_for_each_site_sbm)
            print("Paired t-test p-value btw sbm roi and vbm voxelwise :", p_value)

            t_stat, p_value = ttest_rel(mean_auc_for_each_site, mean_auc_for_each_site_vbm)
            print("Paired t-test p-value btw vbm roi and vbm voxelwise :", p_value)

            t_stat, p_value = ttest_rel(mean_auc_for_each_site_vbm, mean_auc_for_each_site_sbm)
            print("Paired t-test p-value btw vbm roi and sbm roi :", p_value)

            print("\nMean AUC over all combinations ",np.mean(mean_auc_over_all_combinations))
            print("Mean balanced-accuracy over all combinations ",np.mean(mean_bacc_over_all_combinations))

        else:
            for site in sites:
                sbm_current_site = sbm[site]
                vbm_current_site = vbm[site]
                classif_one_site(results_dict, site, sbm_current_site, vbm_current_site)
            
            results_dict["mean over all sites"] = {"roc_auc": np.mean([results_dict[site]["roc_auc"] for site in sites]),\
                                                    "balanced-accuracy":np.mean([results_dict[site]["balanced-accuracy"] for site in sites])}
            print("Mean AUC:", np.mean([results_dict[site]["roc_auc"] for site in sites]))
            print("Mean balanced-accuracy:", np.mean([results_dict[site]["balanced-accuracy"] for site in sites]))

    if save: 
        save_pkl(results_dict,RESULTS_DICT)

"""
meta-model results:

with only VBM ROI and SBM ROI stacking, at maximum training set size (n=8):
    Mean AUC: 0.74875428667064
    Mean balanced-accuracy: 0.658532719675945

with only VBM ROI and VBM voxelwise stacking, at maximum training set size:
    with combination nb 1:
        Mean AUC: 0.769608740053422
        Mean balanced-accuracy: 0.6858811192263289

    mean metrics over all combinations :
        for training set size index idx = 0 (smallest training set size): 
            mean over all combinations  0.6711759806335891
            mean bacc over all combinations  0.5763141531412084
        idx = 1
            mean auc over all combinations  0.7056827134747773
            mean bacc over all combinations  0.6097440597004866
        idx = 2
            mean auc over all combinations  0.7162431388328078
            mean bacc over all combinations  0.6349343886686944
        idx = 3
            mean auc over all combinations  0.7177566925853534
            mean bacc over all combinations  0.6280751938746638
        idx = 4
            mean auc over all combinations  0.739901340297979
            mean bacc over all combinations  0.6625178325431275
        idx = 5
            mean auc over all combinations  0.7321172171717693
            mean bacc over all combinations  0.6649171554001799
        idx = 6
            mean auc over all combinations  0.7503944273500022
            mean bacc over all combinations  0.6578826944979262
        idx = 7
            mean auc over all combinations  0.7600494612782448
            mean bacc over all combinations  0.6930110413595514
        idx = 8
            mean auc over all combinations  0.769822208692828
            mean bacc over all combinations  0.6842651589873633
            
        with classifier = lm.LogisticRegression(C=0.01, class_weight='balanced', fit_intercept=False)
        almost exact same results with GridSearch

at maximum training set size:
Paired t-test p-value btw sbm roi and vbm voxelwise : 2.1315486314235543e-06
Paired t-test p-value btw vbm roi and vbm voxelwise : 0.0005742875290524048
Paired t-test p-value btw vbm roi and sbm roi : 0.00019532726155631125

with only SBM ROI and VBM voxelwise stacking, at maximum training set size:

with only VBM ROI, SBM ROI, and VBM voxelwise stacking, at maximum training set size:

"""

# add Cohen's kappa computation here between SBM and VBM test sets predicted labels (VBM from this script, SBM without imputation)

run_meta_model(save=True)