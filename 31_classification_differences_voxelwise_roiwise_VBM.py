import numpy as np, pandas as pd
from utils import read_pkl, get_predict_sites_list, get_scores_pipeline, save_pkl
import torch, torch.nn as nn, torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

ROOT="/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/"
LEARNING_CRV_TL_DIR = ROOT+"models/VBM_TL/TL_learningcurve/"

class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return x / self.T
    

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_disagreement_scatter(scores_a, scores_b, y_true,
                              label_a="Classifier A",
                              label_b="Classifier B",
                              color_a=None, color_b=None,
                              figsize=(6,6), savepath=None):
    """
    Scatter plot of scores for subjects where two classifiers disagree.
    
    Parameters
    ----------
    scores_a, scores_b : array-like, shape (n_samples,)
        Raw scores or logits of the classifiers.
    y_true : array-like, shape (n_samples,)
        Ground-truth binary labels (0/1).
    label_a, label_b : str
        Names of the classifiers (for axes & legend).
    color_a, color_b : str
        Colors for points where each classifier is correct.
    figsize : tuple
        Size of the figure.
    savepath : str or None
        If given, saves the figure to this path.
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    y_true = np.asarray(y_true)

    # Predictions from scores: positive if >0
    pred_a = (scores_a > 0).astype(int)
    pred_b = (scores_b > 0).astype(int)

    # Mask where they disagree
    mask_disagree = pred_a != pred_b
    if not np.any(mask_disagree):
        print("No disagreements between classifiers.")
        return

    x = scores_a[mask_disagree]
    y = scores_b[mask_disagree]
    y_true_dis = y_true[mask_disagree]
    pred_a_dis = pred_a[mask_disagree]
    pred_b_dis = pred_b[mask_disagree]

    # Colors
    palette = sns.color_palette("deep")
    if color_a is None:
        color_a = palette[0]  # blue-ish
    if color_b is None:
        color_b = "#FFC0CB"  # pink

    colors = [
        color_a if pa == yt else color_b
        for pa, yt in zip(pred_a_dis, y_true_dis)
    ]


    plt.figure(figsize=figsize)
    plt.scatter(x, y, c=colors, alpha=0.7, edgecolor="k", linewidth=0.4)
    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.axvline(0, color='gray', linestyle='--', lw=1)
    plt.xlabel(f"{label_a} scores")
    plt.ylabel(f"{label_b} scores")
    plt.title("Disagreements between classifiers")
    plt.grid(True, linestyle=":", alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', label=f"{label_a} correct",
               markerfacecolor=color_a, markersize=8),
        Line2D([0],[0], marker='o', color='w', label=f"{label_b} correct",
               markerfacecolor=color_b, markersize=8)
    ]
    plt.legend(handles=legend_elems, loc="best")

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def VBM_roi_VBM_voxelwise(plot=False, calibration=True, verbose=False):
    """
    generates and saves a plot to compare VBM ROI and VBM voxelwise predictions (in terms of probabilities)
    at N861 (maximum training set size) 
    """
    dl_tl_5de_te = read_pkl(LEARNING_CRV_TL_DIR+"mean_ypred_252_combinations_df_TL_densenet_Test_set.pkl")
    dl_tl_5de_tr = read_pkl(LEARNING_CRV_TL_DIR+"mean_ypred_252_combinations_df_TL_densenet_Train_set.pkl")
    pathvbm = ROOT+"results_classif/meta_model/scores_tr_te_N861_train_size_N8_vbmroi.pkl"
    vbm = read_pkl(pathvbm)
    sites= get_predict_sites_list()

    all_voxelwise_ypred, all_roi_ypred = [], []
    all_voxelwise_ypred_scores, all_roi_ypred_scores = [], []

    ytrue_all=[]
    for site in sites:
        vbm_current_site = vbm[site]

        maskvbm_te = (
            (dl_tl_5de_te["site"] == site) &
            (dl_tl_5de_te["train_idx"] == 8) &
            (dl_tl_5de_te["combination"] == 1) # chose combination one (doesn't matter)
        )

        maskvbm_tr = (
            (dl_tl_5de_tr["site"] == site) &
            (dl_tl_5de_tr["train_idx"] == 8) &
            (dl_tl_5de_tr["combination"] == 1) # chose combination one (doesn't matter)
        )

        dl_tl_5de_te_one_combination = dl_tl_5de_te[maskvbm_te].copy()
        dl_tl_5de_tr_one_combination = dl_tl_5de_tr[maskvbm_tr].copy()
        ypred_voxelwise_vbm_te = dl_tl_5de_te_one_combination["mean_ypred"].values
        ypred_voxelwise_vbm_te = np.array(ypred_voxelwise_vbm_te[0])

        ypred_voxelwise_vbm_tr = dl_tl_5de_tr_one_combination["mean_ypred"].values
        ypred_voxelwise_vbm_tr = np.array(ypred_voxelwise_vbm_tr[0])

        # print(vbm_current_site["score test"])
        probas_vbm_roi=vbm_current_site["probas_test"]
        scores_vbm_roi_te=vbm_current_site["score test"]
        scores_vbm_roi_tr=vbm_current_site["score train"]

        y_true_te = vbm_current_site["y_test"] # retrieve labels for each site
        y_true_tr = vbm_current_site["y_train"] # retrieve labels for train set
        ytrue_all.append(np.array(y_true_te))

        roc_auc_roiwise = roc_auc_score(y_true_te, scores_vbm_roi_te)
        if verbose: print(f"roc auc vbm roi TEST SET for site {site}={np.round(roc_auc_roiwise,4)}")
        all_roi_ypred.append(probas_vbm_roi)

        roc_auc_voxelwise_te=roc_auc_score(y_true_te, ypred_voxelwise_vbm_te)
        # roc_auc_voxelwise_tr=roc_auc_score(y_true_tr, ypred_voxelwise_vbm_tr)
        if verbose: print(f"roc auc vbm voxelwise TEST SET for site {site}={np.round(roc_auc_voxelwise_te,4)}")
        # print(f"roc auc vbm voxelwise TRAIN SET for site {site}={np.round(roc_auc_voxelwise_tr,4)}")

        scores_vbm_voxelwise_te = ypred_voxelwise_vbm_te
        scores_vbm_voxelwise_tr = ypred_voxelwise_vbm_tr
        dl_scores_test_tensor = torch.from_numpy(ypred_voxelwise_vbm_te)  # convert to tensor

        scaler = StandardScaler()
        scores_vbm_roi_tr = scaler.fit_transform(scores_vbm_roi_tr.reshape(-1, 1))
        scores_vbm_roi_te = scaler.transform(scores_vbm_roi_te.reshape(-1, 1))
        scaler = StandardScaler()
        scores_vbm_voxelwise_tr = scaler.fit_transform(scores_vbm_voxelwise_tr.reshape(-1, 1))
        scores_vbm_voxelwise_te = scaler.transform(scores_vbm_voxelwise_te.reshape(-1, 1))
        all_roi_ypred_scores.append(scores_vbm_roi_te.ravel())

        # calibration 
        if calibration:
            # ==isotonic calibration ===

            calibrator_dl = IsotonicRegression(out_of_bounds='clip')
            calibrator_dl.fit(ypred_voxelwise_vbm_tr, y_true_tr)
            probas_vbm_voxelwise = calibrator_dl.transform(ypred_voxelwise_vbm_te)

            # ==== Platt scaling ====

            # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
            # platt = LogisticRegression(solver='lbfgs', max_iter=1000)
            # grid = GridSearchCV(platt, param_grid, cv=5, scoring='neg_log_loss')
            # grid.fit(vbm_current_site["score train"].reshape(-1,1), y_true_tr)
        
            # platt = LogisticRegression(solver='lbfgs', C=0.001, max_iter=1000)
            # platt.fit(ypred_voxelwise_vbm_tr.reshape(-1,1), y_true_tr)
            # logits_test_reshaped = ypred_voxelwise_vbm_te.reshape(-1,1)
            # probas_vbm_voxelwise = platt.predict_proba(logits_test_reshaped)[:,1]  # probabilities
            # probas_vbm_voxelwise=probas_vbm_voxelwise.squeeze()

            # ==== Temperature scaling ====
            # dl_scores_train_tensor = torch.from_numpy(ypred_voxelwise_vbm_tr)
            # scaler = TempScaler()
            # optimizer = optim.LBFGS([scaler.T], lr=0.01, max_iter=50)
            # loss_fn = nn.BCEWithLogitsLoss()
            # labels_train = torch.from_numpy(y_true_tr).float()

            # def closure_fn():
            #     optimizer.zero_grad()
            #     loss = loss_fn(scaler(dl_scores_train_tensor).squeeze(), labels_train)
            #     loss.backward()
            #     return loss
            # optimizer.step(closure_fn)
            # T = 1.5 #scaler.T.item()
            # print("Fitted temperature:", T)
            # probas_vbm_voxelwise = torch.sigmoid(dl_scores_test_tensor / T).numpy()


        else: probas_vbm_voxelwise = torch.sigmoid(dl_scores_test_tensor).numpy()  # shape: (n_samples,)

        assert np.shape(probas_vbm_roi)==np.shape(probas_vbm_voxelwise)
        
        all_voxelwise_ypred.append(probas_vbm_voxelwise)
        all_voxelwise_ypred_scores.append(scores_vbm_voxelwise_te.ravel())

    all_voxelwise_ypred = np.concatenate(all_voxelwise_ypred,axis=0)
    all_roi_ypred=np.concatenate(all_roi_ypred,axis=0)
    ytrue_all=np.concatenate(ytrue_all, axis=0)
    all_voxelwise_ypred_scores=np.concatenate(all_voxelwise_ypred_scores,axis=0)
    all_roi_ypred_scores=np.concatenate(all_roi_ypred_scores,axis=0)

    plot_disagreement_scatter(all_voxelwise_ypred_scores, all_roi_ypred_scores, ytrue_all,
                          label_a="DL (logits)", label_b="SVM (decision_function)", savepath=ROOT+"test.png")

    quit()
    print("Brier DL:", brier_score_loss(ytrue_all, all_voxelwise_ypred))
    print("Brier SVM:", brier_score_loss(ytrue_all, all_roi_ypred))

    # Calibration curve (fraction of positives vs mean predicted prob)
    prob_true_dl, prob_pred_dl = calibration_curve(ytrue_all, all_voxelwise_ypred, n_bins=10)
    print("prob_true_dl",prob_true_dl)
    print("prob_pred_dl",prob_pred_dl)
    prob_true_svm, prob_pred_svm = calibration_curve(ytrue_all, all_roi_ypred, n_bins=10)
    print("prob_true_svm",prob_true_svm)
    print("prob_pred_svm",prob_pred_svm )
    

    # subjects' predicted labels for each classifier
    pred_dl  = (all_voxelwise_ypred >= 0.5).astype(int)
    pred_svm = (all_roi_ypred >= 0.5).astype(int)

    conf_dl = np.abs(all_voxelwise_ypred - 0.5)
    conf_svm = np.abs(all_roi_ypred - 0.5)

    conf_correct = []
    conf_wrong = []

    for i in range(len(ytrue_all)):
        dl_corr = pred_dl[i] == ytrue_all[i]
        svm_corr = pred_svm[i] == ytrue_all[i]

        if dl_corr and not svm_corr:
            conf_correct.append(conf_dl[i])
            conf_wrong.append(conf_svm[i])
        elif svm_corr and not dl_corr:
            conf_correct.append(conf_svm[i])
            conf_wrong.append(conf_dl[i])

    mean_ratio = np.mean(conf_correct) / np.mean(conf_wrong)
    print("Mean confidence correct / wrong ratio:", mean_ratio)

    # confidence = probability of predicted class
    conf_dl = np.where(pred_dl == 1, all_voxelwise_ypred, 1 - all_voxelwise_ypred)
    conf_svm = np.where(pred_svm == 1, all_roi_ypred, 1 - all_roi_ypred)

    # masks for which model is correct
    dl_correct = (pred_dl == ytrue_all)
    svm_correct = (pred_svm == ytrue_all)

    # Only keep samples where exactly one is correct (so we compare the correct vs wrong)
    xor_mask = dl_correct ^ svm_correct
    idx = np.where(xor_mask)[0]

    conf_correct = np.where(dl_correct[idx], conf_dl[idx], conf_svm[idx])
    conf_wrong   = np.where(dl_correct[idx], conf_svm[idx], conf_dl[idx])

    # log-likelihood ratio
    llr = np.sum(np.log(conf_correct)) - np.sum(np.log(conf_wrong))
    ratio = np.exp(llr)   # your product ratio, but may be huge/small
    print("LLR (sum log conf correct - sum log conf wrong):", llr)
    print("Ratio (exp(LLR)):", ratio)

    if plot : 
        # colors
        colors = []
        deep_palette = sns.color_palette("deep")

        dl_color = deep_palette[0]       # DL-only correct
        svm_color = "#FFC0CB"           # SVM-only correct
        both_color = "green"
        wrong_color = "gray"

        # compute binary predictions
        pred_dl = (all_voxelwise_ypred >= 0.5).astype(int)
        pred_svm = (all_roi_ypred >= 0.5).astype(int)

        # disagreement mask
        mask_disagree = pred_dl != pred_svm

        # distances from 0.5
        x_vals = np.abs(all_voxelwise_ypred[mask_disagree] - 0.5)
        y_vals = np.abs(all_roi_ypred[mask_disagree] - 0.5)
        y_true_dis = ytrue_all[mask_disagree]
        pred_dl_dis = pred_dl[mask_disagree]
        pred_svm_dis = pred_svm[mask_disagree]

        # colors
        colors = []
        for dl_corr, svm_corr in zip(pred_dl_dis == y_true_dis, pred_svm_dis == y_true_dis):
            if dl_corr:
                colors.append(dl_color)
            else:
                colors.append(svm_color)

        # mask for DL correct points
        mask_dl_correct = (pred_dl_dis == y_true_dis)

        # mask for SVM correct points
        mask_svm_correct = (pred_svm_dis == y_true_dis)

        # count DL correct under the diagonal (x < y)
        count_dl_under = np.sum((x_vals[mask_dl_correct] < y_vals[mask_dl_correct]))
        print("DL correct and under x=y:", count_dl_under, "/total points", len(y_true_dis))

        # count SVM correct over the diagonal (y < x)
        count_svm_over = np.sum((y_vals[mask_svm_correct] < x_vals[mask_svm_correct]))
        print("SVM correct and over x=y:", count_svm_over, "/total points", len(y_true_dis))
        quit()

        plt.figure(figsize=(6,6))
        plt.scatter(x_vals, y_vals, c=colors, alpha=0.7, edgecolor='k')

        # diagonal line x=y
        plt.plot([0, 0.5], [0, 0.5], linestyle='--', color='gray')  # max distance is 0.5
        plt.xlabel("Distance from p=0.5 (DL VBM voxelwise)")
        plt.ylabel("Distance from p=0.5 (SVM VBM ROI)")
        plt.title("Disagreement cases: color = correct classifier")
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlim(0,0.5)
        plt.ylim(0,0.5)
        plt.savefig("probabilities_only_disagreement_VBM_roi_voxelwise.png", dpi=300, bbox_inches='tight')
        quit()

        for dl_corr, svm_corr in zip(pred_dl == ytrue_all, pred_svm == ytrue_all):
            if dl_corr and not svm_corr:
                colors.append(dl_color)   # only DL correct
            elif svm_corr and not dl_corr:
                colors.append(svm_color)    # only SVM correct
            elif dl_corr and svm_corr:
                colors.append(both_color)    # both correct
            else:
                colors.append(wrong_color)    # both wrong

        plt.figure(figsize=(6,6))
        plt.scatter(all_voxelwise_ypred, all_roi_ypred, c=colors, alpha=0.5)

        # dashed lines at 0.5
        plt.axhline(0.5, color=svm_color, linestyle='--', linewidth=1.5)
        plt.axvline(0.5, color=dl_color, linestyle='--', linewidth=1.5)

        plt.xlabel("5-DE TL model probabilities with voxelwise features")
        plt.ylabel("SVM-RBF probabilities with ROI features")
        plt.title("Comparison of probabilities by correctness")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle=':', alpha=0.5)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='VBM voxelwise correct only', markerfacecolor=dl_color, markersize=8),
            Line2D([0], [0], marker='o', color='w', label='VBM ROI correct only', markerfacecolor=svm_color, markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Both correct', markerfacecolor=both_color, markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Both wrong', markerfacecolor=wrong_color, markersize=8),
        ]

        plt.legend(handles=legend_elements, loc='upper left')
        plt.savefig("probabilities_correctness_VBM_roi_voxelwise.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_VBM_roi_SBM_roi():
    """
    generates and saves a plot to compare VBM ROI and SBM ROI predictions (in terms of probabilities)
    at N763 (maximum training set size with both preprocessings available) 
    """
    pathsbm = ROOT+"results_classif/meta_model/EN_N8_Destrieux_SBM_ROI_N763.pkl"
    pathvbm = ROOT+"results_classif/meta_model/scores_tr_te_N763_train_size_N8_vbmroi.pkl"
    vbm = read_pkl(pathvbm)
    sbm = read_pkl(pathsbm)
    sites= get_predict_sites_list()

    all_roi_ypred_sbm, all_roi_ypred_vbm = [], []
    ytrue_all=[]
    for site in sites:
        vbm_current_site = vbm[site]
        sbm_current_site = sbm[site]

        print(sbm_current_site["probas_test"])
        probas_sbm_roi=sbm_current_site["probas_test"]
        probas_vbm_roi=vbm_current_site["probas_test"]
        print(np.shape(probas_vbm_roi))
        quit()

        y_true_te = vbm_current_site["y_test"] # retrieve labels for each site
        ytrue_all.append(np.array(y_true_te))

        roc_auc_roiwise = roc_auc_score(y_true_te, vbm_current_site["score test"])
        print(f"roc auc vbm roi for site {site}={np.round(roc_auc_roiwise,4)}")
        all_roi_ypred_vbm.append(probas_vbm_roi)

        roc_auc_voxelwise=roc_auc_score(y_true_te, sbm_current_site["score test"])
        print(f"roc auc vbm voxelwise for site {site}={np.round(roc_auc_voxelwise,4)}\n")

        assert np.shape(probas_vbm_roi)==np.shape(probas_sbm_roi)
        all_roi_ypred_sbm.append(probas_vbm_voxelwise)

    all_roi_ypred_sbm = np.concatenate(all_roi_ypred_sbm,axis=0)
    all_roi_ypred_vbm=np.concatenate(all_roi_ypred_vbm,axis=0)
    ytrue_all=np.concatenate(ytrue_all, axis=0)
    print("all")
    print(np.shape(all_roi_ypred_sbm))
    print(np.shape(all_roi_ypred_vbm))
    print(np.shape(ytrue_all))

    pred_dl  = (all_roi_ypred_sbm >= 0.5).astype(int)
    pred_svm = (all_roi_ypred_vbm >= 0.5).astype(int)

    # colors
    colors = []
    deep_palette = sns.color_palette("deep")

    for dl_corr, svm_corr in zip(pred_dl == ytrue_all, pred_svm == ytrue_all):
        if dl_corr and not svm_corr:
            colors.append(deep_palette[0])   # only DL correct
        elif svm_corr and not dl_corr:
            colors.append("#FFC0CB")    # only SVM correct
        elif dl_corr and svm_corr:
            colors.append("green")    # both correct
        else:
            colors.append("gray")    # both wrong

    plt.figure(figsize=(6,6))
    plt.scatter(all_roi_ypred_sbm, all_roi_ypred_vbm, c=colors, alpha=0.7)

    # dashed lines at 0.5
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0.5, color='gray', linestyle='--', linewidth=1)

    plt.xlabel("5-DE TL model probabilities with voxelwise features")
    plt.ylabel("SVM-RBF probabilities with ROI features")
    plt.title("Comparison of probabilities by correctness")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=':', alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='VBM voxelwise correct only', markerfacecolor=deep_palette[0], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='VBM ROI correct only', markerfacecolor="#FFC0CB", markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Both correct', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Both wrong', markerfacecolor='gray', markersize=8),
    ]

    plt.legend(handles=legend_elements, loc='upper left')
    plt.savefig("probabilities_correctness_VBM_roi_voxelwise.png", dpi=300, bbox_inches='tight')
    plt.close()

VBM_roi_VBM_voxelwise(plot=True)
quit()
plot_VBM_roi_SBM_roi()