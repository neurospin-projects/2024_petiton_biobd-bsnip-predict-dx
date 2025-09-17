import numpy as np, pandas as pd, os
from utils import read_pkl, get_predict_sites_list, save_pkl
import itertools
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
PATH_DATA_TL = ROOT+"models/SBM_TL"
PATH_DATA_RIDL = ROOT+"models/SBM_RIDL"
ML_RESULTS_PATH = ROOT+"results_classif/classifSBM/"
# dict containing ratio of training set nbs of HC subjects / BD subjects for each LOSO CV site
POSWEIGHT_DICT = {"Baltimore":1.175,
                    "Boston": 1.26,
                    "Dallas": 1.188,
                    "Detroit": 1.19,
                    "Hartford": 1.18,
                    "mannheim": 1.268,
                    "creteil": 1.244,
                    "udine": 1.082,
                    "galway": 1.286,
                    "pittsburgh": 1.379,
                    "grenoble": 1.292,
                    "geneve": 1.24
                    }
# output
PATH_DF_DL=ROOT+"reports/summary_results_sbm_vertexwise_RIDL_TL.pkl"
PATH_DF_ML=ROOT+"reports/summary_results_sbm_vertexwise_ML.pkl"

PATH_DICT_5DE_RIDL=ROOT+"reports/means_RIDL_5DE_ypred_train_and_test_N8.pkl"
PATH_DICT_5DE_TL=ROOT+"reports/means_TL_5DE_ypred_train_and_test_N8.pkl"

def read_results(folder, run_nb, trainset_size, mode):
    """
    Read evaluation results (metrics, predictions, indices) for a given training run.

    params
    ----------
    folder : str
        Path to the checkpoint/results folder where the .pkl result files are stored.
    run_nb : int
        Identifier of the current run (e.g., 1–10).
    trainset_size : int
        Number of sites used for training (0–8 in your setup).
    mode : str
        Training mode, e.g. "RIDL" (random initialization) or "TL" (transfer learning).
        This is checked against the stored args["mode"] in the results file for consistency.

    returns
    -------
    results : list of dict
        A list of dictionaries, one per test site, each containing:
            - "roc_auc_te" : float, ROC-AUC on the test set for this site
            - "bacc_te"    : float, Balanced accuracy on the test set
            - "site"       : str, site identifier
            - "mode"       : str, training mode ("RIDL" or "TL")
            - "run_nb"     : int, run number
            - "trainset_size" : int, number of sites used for training
            - "ypred_tr"   : np.ndarray, predicted scores for training set
            - "ypred_te"   : np.ndarray, predicted scores for test set
            - "ytrue_tr"   : np.ndarray, ground-truth labels for training set
            - "ytrue_te"   : np.ndarray, ground-truth labels for test set
            - "indices_tr" : np.ndarray or NaN, subject indices in training set (if available)
            - "indices_te" : np.ndarray or NaN, subject indices in test set (if available)

    rocaucmean : float
        Mean ROC-AUC across all test sites in this run.
    baccmean : float
        Mean balanced accuracy across all test sites in this run.
    """
    results = []
    mean_roc_auc, mean_bacc = [], []
    for site in get_predict_sites_list():
        name_file=f"model_epoch_300_encoder.pthresults_{mode}_ep200_dr0.2_bs128_wd5e-05_pw{POSWEIGHT_DICT[site]}_lr0.0001_site{site}_gamma0.85_no_val_results.pkl"
        path = folder+"/"+name_file
        data = read_pkl(path)
        assert mode == data["args"]["mode"],"mode mismatch"

        res = {
            "roc_auc_te": data["test_metrics"]["roc_auc"],
            "bacc_te": data["test_metrics"]["balanced_accuracy"],
            "site": site,
            "mode": data["args"]["mode"],
            "run_nb": run_nb,
            "trainset_size": trainset_size,
            "ypred_tr": data["train_predictions"]["y_pred"],
            "ypred_te": data["test_predictions"]["y_pred"],
            "ytrue_tr": data["train_predictions"]["y_true"],
            "ytrue_te": data["test_predictions"]["y_true"],
        }

        # only add indices if they exist (will be NaN in df if they don't exist)
        if "indices" in data["train_predictions"]:
            res["indices_tr"] = data["train_predictions"]["indices"]

        if "indices" in data["test_predictions"]:
            res["indices_te"] = data["test_predictions"]["indices"]
            
        mean_roc_auc.append(data["test_metrics"]["roc_auc"])
        mean_bacc.append(data["test_metrics"]["balanced_accuracy"])
        
        results.append(res)

    # add mean metrics over all 12 LOSO test sites
    res = {
        "roc_auc_te": np.mean(mean_roc_auc),
        "bacc_te": np.mean(mean_bacc),
        "site": "all_sites_mean",
        "mode": data["args"]["mode"],
        "run_nb": run_nb,
        "trainset_size": trainset_size,
        "ypred_tr": "NA",
        "ypred_te": "NA",
        "ytrue_tr": "NA",
        "ytrue_te": "NA",
        "indices_tr": "NA",
        "indices_te": "NA",
        }
    rocaucmean=np.mean(mean_roc_auc)
    baccmean=np.mean(mean_bacc)

    return results, rocaucmean, baccmean


def get_results_max_tr_set_size_RIDL_vs_TL_10runs():
    """
    Get ROC-AUC and Balanced Accuracy mean and standard deviation (between the 10 runs) 
    for 10 runs of randomly initialized deep learning (RIDL) and transfer learning (TL)
    using SBM vertex wise data
    """
    for mode in ["TL","RIDL"]:
        mean_over_runs_rocauc, mean_over_runs_bacc = [], []

        for run_nb in range(1,11):
            path_data = PATH_DATA_TL if mode=="TL" else PATH_DATA_RIDL
            name_folder = path_data+f"/checkpoints_sbm_{mode}_whole_brain_N{8}_run{run_nb}"
            r, rocaucmean, baccmean = read_results(name_folder, run_nb, 8, mode)
            mean_over_runs_rocauc.append(rocaucmean)
            mean_over_runs_bacc.append(baccmean)
            
        if mode=="TL":
            print("Mean ROC-AUC and mean Balanced-Accuracy over 10 runs at maximum training set site for TRANSFER LEARNING -")
        else:
            print("Mean ROC-AUC and mean Balanced-Accuracy over 10 runs at maximum training set site for RANDOMLY INITIALIZED (WEIGHTS) DL -")
        print("ROC-AUC: ",round(np.mean(mean_over_runs_rocauc),4), " +- ",round(np.std(mean_over_runs_rocauc),4))
        print("Balanced-Accuracy: ",round(np.mean(mean_over_runs_bacc),4), " +- ",round(np.std(mean_over_runs_bacc),4))

    """
    Mean ROC-AUC and mean Balanced-Accuracy over 10 runs at maximum training set site for TRANSFER LEARNING -
    ROC-AUC: 0.6365  +-  0.0061
    Balanced-Accuracy: 0.5912  +-  0.011

    Mean ROC-AUC and mean Balanced-Accuracy over 10 runs at maximum training set site for RANDOMLY INITIALIZED (WEIGHTS) DL -
    ROC-AUC: 0.6308  +-  0.009
    Balanced-Accuracy: 0.5929  +-  0.0091
    """

def create_df_summary_of_results(save=False):
    """
    Creates and optionally saves a dataframe of results from runs of RIDL and TL models for SBM vertexwise data
    runs : 
    TL : 10 runs at all training set sizes (described as N0, N1, N2, N3, N4, N5, N6, N7, and N8 
            (N8 is maximum training set size, with about 700 subjects per training set))
    RIDL : 10 runs at maximum training set size, and 1 run for each other training set size
    """

    all_results = []
    for mode in ["RIDL","TL"]:
        print(f"reading {mode} results ...")
        path_data = PATH_DATA_TL if mode=="TL" else PATH_DATA_RIDL
        for trainset_size in range(0,9):
            print(f"for training set size idx {trainset_size} ...")
            if trainset_size == 8:
                for run_nb in range(1,11):
                    name_folder = path_data+f"/checkpoints_sbm_{mode}_whole_brain_N{trainset_size}_run{run_nb}"
                    results, _, _ = read_results(name_folder, run_nb, trainset_size, mode)
                    all_results.extend(results)
            else:  
                if mode=="RIDL": # only one run for train set sizes other than maximum train set size for RIDL
                    name_folder = path_data+f"/checkpoints_sbm_{mode}_whole_brain_N{trainset_size}_run1"
                    results, _, _ = read_results(name_folder, 1, trainset_size, mode)
                    all_results.extend(results)
                else:
                    for run_nb in range(1,11):
                        name_folder = path_data+f"/checkpoints_sbm_{mode}_whole_brain_N{trainset_size}_run{run_nb}"
                        results, _, _ = read_results(name_folder, run_nb, trainset_size, mode)
                        all_results.extend(results)

    df = pd.DataFrame(all_results)
    print(df.head())
    if save: save_pkl(df, PATH_DF_DL)

def compute_mean_predictions_over_combinations(trainset_size=8, mode="TL", DE_grouping_size=5, save=False):
    """
    Compute mean predictions (train + test) over all combinations of runs.

    params
    ----------
    trainset_size : int
        training set size
    mode : str
        default "TL"
        use either models using transfer learning or trained from randomly initialized weights (RIDL).
    DE_grouping_size : int
        Size of Deep Ensemble (DE) combinations to average over (default 5).

    returns
    -------
    results_dict : dict
        Dict where each key is a combination index (1 to 252) (there are 252 possible ways of arranging 10 into groups of 5),
        and each value is a dict: {site: {"ypred_tr": arr, "ypred_te": arr}}
    """
    df = read_pkl(PATH_DF_DL)
    filtered_df = df[
        (df['mode'] == mode) & 
        (df['trainset_size'] == trainset_size) & 
        (df['site'] != 'all_sites_mean')
    ].copy()
    
    print(f"Filtered DataFrame shape: {filtered_df.shape}")
    print(f"Available sites: {filtered_df['site'].unique()}")
    print(f"Available runs: {sorted(filtered_df['run_nb'].unique())}")
    print(filtered_df)
    assert len(filtered_df)==120 # 12 sites and 10 runs 

    results_dict = {}

    # Group results by (site, run_nb)
    site_run_dict = {site: {} for site in get_predict_sites_list()}
    site_shapes = {site: {"ypred_tr": None, "ypred_te": None} for site in get_predict_sites_list()}


    for _, row in filtered_df.iterrows():
        site = row["site"]
        run = row["run_nb"]

        # ### debug
        # tr_shape = np.shape(row["ypred_tr"])
        # te_shape = np.shape(row["ypred_te"])
        
        # if site_shapes[site]["ypred_tr"] is None:
        #     site_shapes[site]["ypred_tr"] = tr_shape
        #     site_shapes[site]["ypred_te"] = te_shape
        #     print(f"Site {site}: Expected shapes - ypred_tr: {tr_shape}, ypred_te: {te_shape}")
        # else:
        #     expected_tr_shape = site_shapes[site]["ypred_tr"]
        #     expected_te_shape = site_shapes[site]["ypred_te"]
            
        #     if tr_shape != expected_tr_shape:
        #         print(f"Site {site}, Run {run} - ypred_tr shape mismatch! expected: {expected_tr_shape}, got: {tr_shape}")
                
        #     if te_shape != expected_te_shape:
        #         print(f"Site {site}, Run {run} - ypred_te shape mismatch! expected: {expected_te_shape}, got: {te_shape}")
        # ### end debug

        site_run_dict[site][run] = {
            "ypred_tr": row["ypred_tr"],
            "ypred_te": row["ypred_te"],
        }

    # Enumerate all combinations of available runs
    combs = list(itertools.combinations(list(range(1,11)), DE_grouping_size))
    print(f"Total combinations: {len(combs)}")
    assert len(combs)==252,f"there should be 252 combinations. {len(combs)} found instead."
    
    for idx, comb in enumerate(combs, 1):  # start index at 1
        comb_dict = {}
        for site, run_dict in site_run_dict.items():
            print(site)
            ypred_trs, ypred_tes = [], []
            for element in comb:
                print(np.shape(run_dict[element]["ypred_tr"])) 
                ypred_trs.append(run_dict[element]["ypred_tr"])
                ypred_tes.append(run_dict[element]["ypred_te"])
            ypred_trs=np.array(ypred_trs)
            ypred_tes=np.array(ypred_tes)

            # Compute mean for current combination
            mean_ypred_tr = np.mean(ypred_trs, axis=0)
            mean_ypred_te = np.mean(ypred_tes, axis=0)

            comb_dict[site] = {
                "ypred_tr": mean_ypred_tr,
                "ypred_te": mean_ypred_te,
            }

        results_dict[idx] = comb_dict

    print(f"Successfully computed {len(results_dict)} combinations")
    # print(f"Sites in first combination: {list(results_dict[1].keys()) if results_dict else 'None'}")
    # print(results_dict.keys())
    # print(results_dict[1].keys())

    if trainset_size==8: path_dict = PATH_DICT_5DE_TL if mode=="TL" else PATH_DICT_5DE_RIDL
    else: 
        if mode=="TL": path_dict=ROOT+f"reports/means_TL_5DE_ypred_train_and_test_N{trainset_size}.pkl"
        else: 
            print("RIDL 5DE not supported for training set sizes other than maximum training set size. More runs would be required.")
            quit()

    if save: save_pkl(results_dict, path_dict)

    return results_dict

def compare_5DE_TL_and_RIDL():
    ridl = read_pkl(PATH_DICT_5DE_RIDL)
    tl = read_pkl(PATH_DICT_5DE_TL)
    df = read_pkl(PATH_DF_DL)
    filtered_df = df[
        (df['mode'] == "TL") & 
        (df['trainset_size'] == 8) & 
        (df['site'] != 'all_sites_mean')
    ].copy()
    list_sites = get_predict_sites_list()
    dict_y_true = {site: {} for site in list_sites}

    for site in list_sites:
        filtered_df_site_tr = filtered_df[filtered_df["site"]==site]["ytrue_tr"].values
        filtered_df_site_te = filtered_df[filtered_df["site"]==site]["ytrue_te"].values
        dict_y_true[site]={"y_true_tr":filtered_df_site_tr[0], "y_true_te":filtered_df_site_te[0]}

    roc_auc_ridl, roc_aucs_tl = {site:{} for site in list_sites}, {site:{} for site in list_sites}
    
    for combination_idx in range(1,253):
        for site in list_sites:
            y_pred_te_ridl = ridl[combination_idx][site]["ypred_te"]
            y_pred_te_tl = tl[combination_idx][site]["ypred_te"]

            roc_auc_te_ridl = roc_auc_score(dict_y_true[site]["y_true_te"], y_pred_te_ridl)
            roc_auc_te_tl = roc_auc_score(dict_y_true[site]["y_true_te"], y_pred_te_tl)
            roc_auc_ridl[site][combination_idx]=roc_auc_te_ridl
            roc_aucs_tl[site][combination_idx] = roc_auc_te_tl

    df_ridl = pd.DataFrame(roc_auc_ridl).T   # shape: sites × combinations
    df_tl   = pd.DataFrame(roc_aucs_tl).T  
    print(df_ridl)  
    print(df_tl)

    def summarize(df):
        overall_mean = df.values.mean()
        std_across_sites = df.mean(axis=1).std() # mean across combinations per site, then std across sites
        std_across_combs = df.mean(axis=0).std() # mean across sites per combination, then std across combs
        return overall_mean, std_across_sites, std_across_combs

    ridl_mean, ridl_std_sites, ridl_std_combs = summarize(df_ridl)
    tl_mean,   tl_std_sites,   tl_std_combs   = summarize(df_tl)

    print(f"RIDL - mean: {ridl_mean:.3f}, std across sites: {ridl_std_sites:.3f}, std across combinations: {ridl_std_combs:.3f}")
    print(f"TL   - mean: {tl_mean:.3f}, std across sites: {tl_std_sites:.3f}, std across combinations: {tl_std_combs:.3f}")

    """
    RIDL - mean: 0.634, std across sites: 0.107, std across combinations: 0.005
    TL   - mean: 0.646, std across sites: 0.095, std across combinations: 0.002
    """

def learning_curve_RIDL_vs_DL():
    roc_aucs = {"TL":{},"RIDL":{},"EN":{},"L2LR":{},"MLP":{},"svm":{},"xgboost":{}}
    list_ML_classifiers = ["svm", "MLP", "EN", "L2LR", "xgboost"] 
    list_DL_classifiers = ["TL","RIDL"]
    df_DL = read_pkl(PATH_DF_DL)  
    df_ML = read_pkl(PATH_DF_ML)

    for mode in list_DL_classifiers:
        for trainset_size in range(1,9):
            filtered_dfDL = df_DL[
                (df_DL['mode'] == mode) & 
                (df_DL['trainset_size'] == trainset_size) & 
                (df_DL['site'] != 'all_sites_mean') &
                (df_DL["run_nb"]==1)
            ].copy()
            
            print(f"Filtered DataFrame:\n {filtered_dfDL}")
            print(filtered_dfDL[["site","roc_auc_te","mode","run_nb"]])
            assert len(filtered_dfDL)==12
            print(filtered_dfDL["roc_auc_te"].mean())
            roc_aucs[mode][trainset_size]=filtered_dfDL["roc_auc_te"].mean()

    for classifier in list_ML_classifiers:
        for trainset_size in range(1,9):
            filtered_dfML = df_ML[
                (df_ML['classifier'] == classifier) & 
                (df_ML['trainset_size'] == trainset_size) & 
                (df_ML['site'] != 'all_sites_mean')
            ].copy()
            
            print(f"Filtered DataFrame:\n {filtered_dfML}")
            print(filtered_dfML[["site","roc_auc_te","classifier"]])
            assert len(filtered_dfML)==12
            print(filtered_dfML["roc_auc_te"].mean())
            roc_aucs[classifier][trainset_size]=filtered_dfML["roc_auc_te"].mean()

    print(roc_aucs)
    x_vals = np.linspace(100, 700, 8)
    ideal_area = np.trapz(np.ones_like(x_vals), x_vals)  # max possible (y=1)

    # get area under curve of ROC AUC
    dict_auc = {}
    for classif in list_DL_classifiers+list_ML_classifiers:
        y = list(roc_aucs[classif].values())
        auc_ml = np.trapz(y, x_vals)
        auc_ml = round((auc_ml / ideal_area) * 100,2)
        dict_auc[classif]=auc_ml

    from scipy.stats import ttest_1samp
    from itertools import combinations

    print("Test signficance of differences in ROC-AUC values at different train set sizes: \n")

    for clf1, clf2 in combinations(list_DL_classifiers+list_ML_classifiers, 2):
        print(f"Comparing {clf1} vs {clf2}")
        aucs1 = np.array(list(roc_aucs[clf1].values()), dtype=float)
        aucs2 = np.array(list(roc_aucs[clf2].values()), dtype=float)
        differences = aucs2-aucs1

        t_stat, p_value = ttest_1samp(differences, popmean=0)
        print(f"One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = {p_value:.4f}\n")
    
    """
    Comparing TL vs RIDL
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.2386

    Comparing TL vs svm
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0003

    Comparing TL vs MLP
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0060

    Comparing TL vs EN
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0702

    Comparing TL vs L2LR
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0004

    Comparing RIDL vs svm
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0004

    Comparing RIDL vs MLP
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0149

    Comparing RIDL vs EN
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0620

    Comparing RIDL vs L2LR
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0006

    Comparing svm vs MLP
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0179

    Comparing svm vs EN
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0838

    Comparing svm vs L2LR
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0653

    Comparing MLP vs EN
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.5945

    Comparing MLP vs L2LR
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0039

    Comparing EN vs L2LR
    One sample t-test where H0: no difference in ROC-AUC values for the two classifiers = 0.0093
    """


    deep_palette = sns.color_palette("deep")
    color_dict={
        "EN" : deep_palette[1],
        "xgboost" : deep_palette[2],
        "L2LR" : deep_palette[3],
        "svm" : deep_palette[4],
        "MLP":"#8B4513",
        "RIDL":"#FFC0CB",
        "TL":deep_palette[0]
    }

    
    for classif in list_DL_classifiers: 
        # plot TL (blue) or RIDL (pink)
        sns.lineplot(x_vals, list(roc_aucs[classif].values()), color=color_dict[classif],  linewidth=5, linestyle="dashed", label=f"{classif} (AUC={dict_auc[classif]:.2f})")

    for classif in list_ML_classifiers:
        if classif=="xgboost": classif_name = "Gradient Boosting" 
        elif classif=="svm": classif_name = "SVM-RBF" 
        else: classif_name = classif
        sns.lineplot(x_vals, list(roc_aucs[classif].values()), color=color_dict[classif],  linewidth=5, linestyle="solid", \
                     label=f"{classif_name} (AUC={dict_auc[classif]:.2f})")


    plt.xlabel("training dataset size",fontsize=25)
    plt.ylabel("Mean ROC-AUC over LOSO test sites",fontsize=25)
    plt.title("DL vs ML performance with SBM vertexwise features",fontsize=30)
    plt.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()

def get_results_ML(save=False):
    list_sites = get_predict_sites_list()
    list_sizes = [75, 150, 200, 300, 400, 450, 500, 600,700]
    list_classifiers = ["svm", "MLP", "EN", "L2LR", "xgboost"]
    results = [] 

    for classifier in list_classifiers:
        print("\nclassifier ", classifier)
        for size in [75, 150, 200, 300, 400, 450, 500, 600, 700]:
            print("train set size ",size)
            res_path = ML_RESULTS_PATH+f"{classifier}_N{size}_SBM_vertexwise_N763.pkl"
            data=read_pkl(res_path)
            roc_auc_all_sites, bacc_all_sites = [], []
            for site in list_sites:
                traindata_size_idx = list_sizes.index(size)
                roc_auc = data[site]["roc_auc"]
                bacc = data[site]["balanced-accuracy"]

                results.append({"site":site,
                                    "classifier":classifier,
                                    "trainset_size":traindata_size_idx,
                                    "roc_auc_te":roc_auc,
                                    "bacc_te":bacc})
                
                roc_auc_all_sites.append(roc_auc)
                bacc_all_sites.append(bacc)
            # print("mean roc auc ", round(np.mean(roc_auc_all_sites),4))
            # print("std roc auc ", round(np.std(roc_auc_all_sites),4))

            print("mean bacc ", round(np.mean(bacc_all_sites),4))
            print("std bacc ", round(np.std(bacc_all_sites),4))

            results.append({
                "site": "all_sites_mean",
                "classifier": classifier,
                "trainset_size": traindata_size_idx,
                "roc_auc_test": np.mean(roc_auc_all_sites),
                "bacc_test": np.mean(bacc_all_sites)
            })

    df_results = pd.DataFrame(results)
    if os.path.exists(PATH_DF_ML) and save: save_pkl(df_results, PATH_DF_ML)
    # print(df_results)
            

def main():
    get_results_ML(save=True)
    learning_curve_RIDL_vs_DL()
    quit()
    # get_results_max_tr_set_size_RIDL_vs_TL_10runs()
    # create_df_summary_of_results(save=True)
    # for trainset_size in range(1,8):
        # compute_mean_predictions_over_combinations(mode="TL", save=True, trainset_size=trainset_size)
    # compare_5DE_TL_and_RIDL()
    
    def get_sbm_vertex_wise_ytrue_dict():
        df = read_pkl(PATH_DF_DL)
        filtered_df = df[
            (df['mode'] == "TL") & 
            (df['trainset_size'] == 8) & 
            (df['site'] != 'all_sites_mean')
        ].copy()
        list_sites = get_predict_sites_list()
        dict_y_true = {site: {} for site in list_sites}

        for site in list_sites:
            filtered_df_site_tr = filtered_df[filtered_df["site"]==site]["ytrue_tr"].values
            filtered_df_site_te = filtered_df[filtered_df["site"]==site]["ytrue_te"].values
            dict_y_true[site]={"y_true_tr":filtered_df_site_tr[0], "y_true_te":filtered_df_site_te[0]}

        return dict_y_true

    def get_y_pred_sbm_vertexwise_by_combination_idx_and_site(ridl, tl, combination_idx,site):
        y_pred_te_ridl = ridl[combination_idx][site]["ypred_te"]
        y_pred_te_tl = tl[combination_idx][site]["ypred_te"]
        return y_pred_te_ridl, y_pred_te_tl
    
    ridl = read_pkl(PATH_DICT_5DE_RIDL)
    tl = read_pkl(PATH_DICT_5DE_TL)
    # for combination_idx in range(1,253):
        


    

    """
    at N8 TL, mean over 10 runs
              site mode  roc_auc_te
    0    Baltimore   TL    0.555220
    1       Boston   TL    0.802469
    2       Dallas   TL    0.547205
    3      Detroit   TL    0.807143
    4     Hartford   TL    0.600561
    5      creteil   TL    0.556115
    6       galway   TL    0.564706
    7       geneve   TL    0.578143
    8     grenoble   TL    0.658937
    9     mannheim   TL    0.668750
    10  pittsburgh   TL    0.634148
    11       udine   TL    0.664321
    """
        


if __name__ == "__main__":
    main()

