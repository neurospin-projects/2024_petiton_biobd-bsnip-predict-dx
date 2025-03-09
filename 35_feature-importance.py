# %% Imports

import numpy as np
import scipy
import pandas as pd
import pickle
import os.path
import glob

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# ML
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
from sklearn.metrics import roc_auc_score

from nilearn import plotting
from nilearn import image

# %% Path

WD = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx"
OUTPUT_BASENAME = os.path.join(WD, "reports/ROI-SHAP")
OUTPUT_FILENAME = OUTPUT_BASENAME + "_shap-univstat.xlsx"

# %% Read data
# ============

participants = pd.read_csv(os.path.join(WD, "data/processed", "participantsBD.csv"))
participants = participants[["participant_id", "sex", "age", "diagnosis", "site"]]
participants.diagnosis.replace({'bipolar disorder':'BD', 'control':'HC', 'psychotic bipolar disorder':'BD'}, inplace=True)
data = pd.read_csv(os.path.join(WD, "data/processed", "roisBD_residualized_and_scaled.csv"))
data = pd.merge(participants, data)

atlas_df = pd.read_csv(os.path.join(WD, "data/atlases/lobes_Neuromorphometrics.csv"), sep=';')

shap_df = pd.read_excel(os.path.join(WD, "models/ShapValues/shap_computed_from_all_Xtrain/SHAP_summary.xlsx"))

# %% Randomize SHAP values MOVE THIS IN ANOTHER FILE
# ==================================================
#
# Output:  "models/ShapValues/SHAP_randomized_*.csv"

if False:
     
    X = data.loc[:, 'l3thVen_GM_Vol':]
    y = data.diagnosis

    shap_values_list = list()
    for i in range(1, 10): #range(5):
        y = np.random.permutation(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        ss = StandardScaler()
        X_train = pd.DataFrame(ss.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(ss.transform(X_test), columns=X.columns)

        #svc_pipeline = Pipeline([
        #    ('scaler', StandardScaler()), 
        #    ('svc', SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced'))
        #])
        svc_pipeline = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced')
        svc_pipeline.fit(X_train, y_train)

        # Subsample dataset to speed-up computation
        #X_train_sample = X_train.iloc[np.random.choice(X_train.shape[0], 5, replace=False)]
        X_train_sample = X_train
        explainer = shap.KernelExplainer(svc_pipeline.decision_function, X_train_sample) 
        shap_values = explainer.shap_values(X_test)
        #np.save(os.path.join(WD, "models/ShapValues", "SHAP_randomized.npy"), shap_values)
        pd.DataFrame(shap_values, columns=X_train.columns).\
            to_csv(os.path.join(WD, "models/ShapValues/SHAP_randomized_%i.csv" % i), index=False)
        shap_values_list.append(shap_values)

    # print("shap_values =", shap_values)
    # print("base value =", explainer.expected_value)
    # shap_values.shape == (861, 284)
    #shap.plots.waterfall(shap_values[0])

# %% Select relevant features based on significance of SHAP values
# ================================================================
#
# Input: 
# - 'models/ShapValues/SHAP_randomized_*.csv' files
# - "models/statistics_univ/statsuniv_roisBD.xlsx" file
# Output: OUTPUT_FILENAME

# Read randomized SHAP values

files_ = glob.glob(os.path.join(WD, 'models/ShapValues/SHAP_randomized_*.{}'.format("csv")))
shap_rnd_absmean = pd.DataFrame([pd.read_csv(f).abs().mean() for f in files_])

# SHAP Statistics mean,std, n, CI 
shap_df = shap_df[['fold', 'ROI', 'mean_abs_shap']]

m = shap_df.groupby(['ROI'])["mean_abs_shap"].mean()
s = shap_df.groupby(['ROI'])["mean_abs_shap"].std(ddof=1)
n = shap_df.groupby(['ROI'])["mean_abs_shap"].count()
# Critical value for t at alpha / 2:
t_alpha2 = -scipy.stats.t.ppf(q=0.05/2, df=n-1, loc=0)
# One sidded t-stat
# t_alpha2 = -scipy.stats.t.ppf(q=0.05, df=n-1, loc=0)

ci_low = m - t_alpha2 * s / np.sqrt(n)
ci_high = m + t_alpha2 * s / np.sqrt(n)

# Mean under H0
m_h0 = shap_rnd_absmean.mean()
# set(m.index) - set(m_h0.index)
# set(m.index).intersection(set(m_h0.index))


m_h0 = m_h0[m.index]
# np.sum(m_h0 < ci_low)

shap_stat = dict(ROI=m.index, mean_abs_shap=m, ci_low=ci_low, ci_high=ci_high,
                 mean_abs_shap_h0=m_h0, select = m_h0 < ci_low)
shap_stat = pd.DataFrame(shap_stat)
shap_stat.sort_values(by="mean_abs_shap", ascending=False, inplace=True)

# Merge with univ-stats
stat_univ = pd.read_excel(os.path.join(WD, "models/statistics_univ/statsuniv_roisBD.xlsx"),
                          sheet_name='statsuniv_roisBD_residualized_and_scaled')

shap_stat = pd.merge(shap_stat.reset_index(drop=True), stat_univ.reset_index(drop=True),
                     on='ROI', how='left')

# Merge with atlas
shap_stat = pd.merge(shap_stat, atlas_df[['ROIabbr', 'ROIname']], how='left', left_on='ROI', right_on='ROIabbr')
# Write
shap_stat.to_excel(OUTPUT_FILENAME, sheet_name='SHAP_roi_univstat', index=False)


# %% Split variable into specifics and suppressors
# =================================================

# Input: "models/ShapValues/SHAP_roi_univstat.xlsx"
"""
https://www.journals.uchicago.edu/doi/pdf/10.5243/jsswr.2010.2
"A suppressor variable correlates with other independent variables, and accounts for or suppresses some outcome-irrelevant
variation or errors in one or more other predictors, and improves the overall predictive power of the model."
"""

shap_stat = pd.read_excel(OUTPUT_FILENAME,
    sheet_name='SHAP_roi_univstat')

if not "type" in shap_stat:
    # Filter Features based on significance of SHAP values
    shap_stat = shap_stat[shap_stat.select == 1]
    assert shap_stat.shape[0] == 116

    shap_stat["type"] = None
    shap_stat.loc[shap_stat.diag_pcor < 0.05, "type"] = "specific"
    shap_stat.loc[shap_stat.diag_p > 0.05, "type"] = "suppressor"

    # Write
    shap_stat.to_excel(OUTPUT_FILENAME,
        sheet_name='SHAP_roi_univstat', index=False)


# %% Illustrate Suppressor variable with linear model

shap_stat = pd.read_excel(OUTPUT_FILENAME,
    sheet_name='SHAP_roi_univstat')
# Split features in two: specific and suppressor
shap_spec = shap_stat[shap_stat.type=="specific"]
shap_suppr = shap_stat[shap_stat.type=="suppressor"]
    

X = data[list(shap_spec.ROI) + list(shap_suppr.ROI)]
y = data.diagnosis

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lr = LogisticRegressionCV()
lr.fit(X_scaled, y)

assert lr.coef_.shape[1] == len(shap_spec.ROI) + len(shap_suppr.ROI)

coef_spec = lr.coef_[0, :len(shap_spec.ROI)]
coef_supr = lr.coef_[0, len(shap_spec.ROI):]  

score_spec = np.dot(X_scaled[:, :len(shap_spec.ROI)], coef_spec)
score_supr = np.dot(X_scaled[:, len(shap_spec.ROI):], coef_supr)  
score_tot = np.dot(X_scaled, lr.coef_[0, :])

score_spec_auc = roc_auc_score(y, score_spec)
score_supr_auc = roc_auc_score(y, score_supr)
score_tot_auc = roc_auc_score(y, score_tot)

df = pd.DataFrame(dict(diagnosis=data.diagnosis, score_spec=score_spec,
                       score_supr=score_supr, score_tot=score_tot))


# Create a joint plot with scatter and marginal density plots

plt.figure(figsize=(8, 8))
g = sns.jointplot(data=df, x="score_supr", y="score_spec", hue="diagnosis")
g.ax_joint.set_xlabel("Score with Suppressor Features (AUC=%.2f)" % score_supr_auc)
g.ax_joint.set_ylabel("Score with Speficic Features  (AUC=%.2f)" % score_spec_auc)
g.fig.suptitle("Scatter Plot with Marginal Densities", y=1.02)
# plt.show()
plt.savefig(OUTPUT_BASENAME + "plot_suppressor-specific_scatter.pdf")  
plt.close()

# Density (KDE) Plot

plt.figure(figsize=(6, 2))
g = sns.kdeplot(data=df, x='score_tot', hue='diagnosis', fill=True, alpha=0.3)
g.set_xlabel("Density Plot of Score with all Features (AUC=%.2f)" % score_tot_auc)
plt.title("Density Plot of Score with all Features (AUC=%.2f)" % score_tot_auc)
#plt.grid(True)
#plt.show()
plt.savefig(OUTPUT_BASENAME + "plot_suppressor-specific_density.pdf")  
plt.close()

# %% Exploratory analysis (PCA & Correlation matrix)

shap_stat = pd.read_excel(OUTPUT_FILENAME,
    sheet_name='SHAP_roi_univstat')

shap_spec = shap_stat[shap_stat.type == "specific"]

def plot_cluster_abs_corr_mar(corr_matrix):
    # [Improve the figure](https://fr.moonbooks.org/Articles/Les-dendrogrammes-avec-Matplotlib/)
    from scipy.cluster.hierarchy import linkage, dendrogram
    fig = plt.figure(figsize=(8, 8))  # Create an 8x8 inch figure

    # Apply hierarchical clustering to reorder correlation matrix
    linkage_matrix = linkage(1 - corr_matrix, method="ward")  # Ward's method for clustering
    
    #dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
    dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=False)
    plt.grid(False)
    sorted_columns = dendro["ivl"]  # Reordered column names

    # Reorder correlation matrix
    reordered_corr = corr_matrix.loc[sorted_columns, sorted_columns]

    # Plot the clustered heatmap with a colormap optimized for positive values
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=0.5)
    g = sns.heatmap(reordered_corr, fmt=".2f", cmap="Reds", square=True,
                linewidths=0.5, vmin=0, vmax=1)

    plt.title("Clustered Correlation Matrix (Absolute Values)")
    plt.show()
    sns.set(font_scale=1.0)

    return dendro

def plot_pca(data):

    # Standardize the data (important for PCA)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA()
    pca.fit(data_scaled)

    # Compute cumulative explained variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Components')
    plt.grid(True)
    plt.show()

    return pca, data_scaled

corr_matrix = data[shap_spec.ROI].corr().abs()

dendro = plot_cluster_abs_corr_mar(data[shap_spec.ROI].corr().abs())
pca, data_scaled = plot_pca(data[shap_spec.ROI])

plot_cluster_abs_corr_mar(data[shap_suppr.ROI].corr().abs())
pca, data_scaled = plot_pca(data[shap_suppr.ROI])


# %% Feature agglomeration
# ========================

from sklearn.cluster import FeatureAgglomeration

# %% How many clusters ?
# ----------------------
#
# Simplify resulst by feature aglomeration 

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_validate


X = data.loc[:, 'l3thVen_GM_Vol':]
y = data.diagnosis
groups = data.site


def clf_cv(X, y, groups):
    svc_pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('svc', SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced'))
    ])

    logo = LeaveOneGroupOut()

    scores_metrics = ["roc_auc", "balanced_accuracy"]  # You can add all the metrics you want to evaluate 

    # Conducting the cross-validation with shuffling of task order within each participant's group
    results = cross_validate(svc_pipeline, X, y, cv=logo, groups=groups, scoring=scores_metrics, return_train_score=True,
                            return_estimator=True, n_jobs=len(groups.unique()))

    return(results['test_roc_auc'].mean().item(), results['test_balanced_accuracy'].mean().item())


# No preprocessing
clf_cv(X, y, groups)
# (0.739744563882606, 0.6760663888907534)
 
# CSF => -1
csf_cols = [col for col in X.columns if 'CSF' in col]
X.loc[:, csf_cols] = -1 * X[csf_cols]
clf_cv(X, y, groups)
# (0.739744563882606, 0.6760663888907534)

shap_stat = pd.read_excel(OUTPUT_FILENAME, sheet_name='SHAP_roi_univstat')

specfific_colnames = shap_stat[shap_stat.type == "specific"].ROI
others_colnames = shap_stat[shap_stat.type != "specific"].ROI


def fa_clf_cv(X, y, groups, n_clusters, specfific_colnames, others_colnames):

    ss = StandardScaler()
    ss.set_output(transform="pandas")
    Xs = ss.fit_transform(X)


    fa = FeatureAgglomeration(n_clusters=int(n_clusters/2), compute_distances=True)

    # Agglomerate specific features
    fa.set_output(transform="pandas")
    fa.fit(Xs[specfific_colnames])
    X_as = fa.transform(Xs[specfific_colnames])

    # Agglomerate other features
    fa.fit(Xs[others_colnames])
    X_ao = fa.transform(Xs[others_colnames])
    X_ao.columns = [c + '_other' for c in X_ao.columns]

    # Concatenate
    X_a = pd.concat([X_as, X_ao], axis=1)

    # Predict
    return clf_cv(X_a, y, groups)


# 18 clusters => 9 specific clusters
 
fa_clf_cv(X, y, groups, n_clusters=18, specfific_colnames=specfific_colnames, others_colnames=others_colnames)
# (0.7596885366014319, 0.6903067945774235)


results = pd.DataFrame(data=
    [[n_clusters] + list(fa_clf_cv(X, y, groups, n_clusters=n_clusters,
                                   specfific_colnames=specfific_colnames, others_colnames=others_colnames))
     for n_clusters in range(2, 61, 2)],
    columns=["n_clusters", "auc", "bacc"])

print(results.round(3))

#     n_clusters    auc   bacc
# 0            2  0.726  0.676
# 1            4  0.727  0.682
# 2            6  0.715  0.673
# 3            8  0.725  0.665
# 4           10  0.762  0.690
# 5           12  0.758  0.680
# 6           14  0.759  0.694
# 7           16  0.762  0.691
# 8           18  0.760  0.690
# 9           20  0.756  0.685
# 10          22  0.763  0.687
# 11          24  0.763  0.705
# 12          26  0.763  0.707
# 13          28  0.769  0.712
# 14          30  0.769  0.700
# 15          32  0.764  0.702
# 16          34  0.764  0.713
# 17          36  0.762  0.712
# 18          38  0.761  0.710
# 19          40  0.762  0.704
# 20          42  0.762  0.704
# 21          44  0.761  0.702
# 22          46  0.763  0.707
# 23          48  0.759  0.705
# 24          50  0.761  0.700
# 25          52  0.761  0.701
# 26          54  0.763  0.700
# 27          56  0.767  0.704
# 28          58  0.766  0.708
# 29          60  0.769  0.712


results.plot(x='n_clusters', y='auc', title="Feature Aglomeration")
# 10 clusters => 5 specific clusters

plt.savefig(OUTPUT_BASENAME + "plot_FeatureAgglomeration.pdf")  
plt.close()

# %% Feature agglomeration

# ********** #
n_clusters = 5
# ********** #

shap_stat = pd.read_excel(OUTPUT_FILENAME,
    sheet_name='SHAP_roi_univstat')

# Split fature in two: specific and suppressor
shap_spec = shap_stat[shap_stat.type=="specific"]
shap_suppr = shap_stat[shap_stat.type=="suppressor"]


def cluster_features(Xdf, n_clusters):
    # Scale
    ss = StandardScaler()
    ss.set_output(transform="pandas")
    Xdf = ss.fit_transform(Xdf)
    #Xdf = pd.DataFrame(X, columns=colnames)

    # setting distance_threshold=0 ensures we compute the full tree.
    #model = AgglomerativeClustering(distance_threshold=0, n_clusters=4)
    model = FeatureAgglomeration(n_clusters=n_clusters, compute_distances=True)
    model.set_output(transform="pandas")
    model = model.fit(Xdf)
    Xdf_r = model.transform(Xdf)
    
    roi_cluster = pd.DataFrame(dict(ROI=Xdf.columns, label=model.labels_))

    return model, roi_cluster


# Multiply CSF feature by -1
Xdf = data[shap_spec.ROI]
csf_cols = [col for col in shap_spec.ROI if 'CSF' in col]
Xdf.loc[:, csf_cols] = -1 * Xdf[csf_cols]

colnames = shap_spec.ROI

model, roi_cluster = cluster_features(Xdf, n_clusters)


shap_stat = pd.read_excel(OUTPUT_FILENAME,
    sheet_name='SHAP_roi_univstat')

shap_stat = pd.merge(shap_stat, roi_cluster, how='left')

if not "label" in \
    pd.read_excel(OUTPUT_FILENAME,
        sheet_name='SHAP_roi_univstat'):
    shap_stat.to_excel(OUTPUT_FILENAME,
        sheet_name='SHAP_roi_univstat', index=False)
    
# %% Plot dendogram

from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendro = dendrogram(linkage_matrix, **kwargs)
    return dendro


#plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
dendro = plot_dendrogram(model, color_threshold=60)
reorder_idx = np.array([int(idx) for idx in dendro["ivl"]])
reorder_columns = Xdf.columns[reorder_idx]
reorder_columns_label = [name + '_' + str(clust) for name, clust in
  zip(reorder_columns, model.labels_[reorder_idx])]

loc, _ = plt.xticks()
plt.xticks(loc, labels=reorder_columns_label)#, rotation=45)
#plt.show()
plt.savefig(OUTPUT_BASENAME + "plot_specific_FeatureAgglomeration_dendrogram.pdf")  
plt.close()

# Plot Corr Mat
# import matplotlib.patches as patches

corr_matrix = Xdf[reorder_columns].corr()
corr_matrix.columns = corr_matrix.index = reorder_columns_label

# Plot the clustered heatmap with a colormap optimized for positive values
plt.figure(figsize=(8, 6))
sns.set(font_scale=0.5)
sns.heatmap(corr_matrix, fmt=".2f", cmap="Reds", square=True,
            linewidths=0.5, vmin=0, vmax=1)
sns.set(font_scale=1.0)
# ax = plt.gca()
# square = patches.Rectangle((0, 0), 1, 2, edgecolor='purple', facecolor='none')
# ax.add_patch(square)
#plt.show()
plt.savefig(OUTPUT_BASENAME + "plot_specific_FeatureAgglomeration_corrMat.pdf")  
plt.close()

# %% Glass brains
# Input
atlas_nii_filename = "data/atlases/neuromorphometrics.nii"
atlas_csv_filename = "data/atlases/lobes_Neuromorphometrics.csv"

import nibabel
import nilearn
from nilearn import plotting

shap_stat = pd.read_excel(OUTPUT_FILENAME,
    sheet_name='SHAP_roi_univstat')

info = shap_stat[shap_stat.type == "specific"]

atlas_nii = nibabel.load(os.path.join(WD, atlas_nii_filename))
atlas_arr = atlas_nii.get_fdata()
atlas_df = pd.read_csv(os.path.join(WD, atlas_csv_filename), sep=';')
info = pd.merge(info, atlas_df[['ROIabbr', 'ROIname', 'ROIid']], how='left')#, left_on='ROI', right_on='ROIabbr')

vmin=info.mean_abs_shap.min()
vmax=info.mean_abs_shap.max()

for lab in info.label.unique(): # Iterate over cluster
    df = info[info.label == lab]
    print(df)
    
    clust_arr = np.zeros(atlas_arr.shape)
    #clust_name = str(int(lab)) + "_" + " ".join([s.replace("_Vol", '').replace("_CSF", '').replace("_GM", '') for s in list(df.ROI)])
    clust_name = str(int(lab)) + "_" + " ".join([s.replace("_Vol", '') for s in list(df.ROI)])
    
    for i in range(df.shape[0]): # Iterate over regions

        roi = df.iloc[i, :]
        roi_mask = atlas_arr == roi.ROIid

        print(i, roi.ROI, roi_mask.sum(), roi.mean_abs_shap)
        mult = -1 if 'CSF' in roi.ROI else 1
        clust_arr[roi_mask] = mult * np.sign(roi.diag_t) * roi.mean_abs_shap

    clust_img = image.new_img_like(atlas_nii, clust_arr)

    plotting.plot_glass_brain(clust_img, title=clust_name, vmax=vmax, colorbar=True, plot_abs=False, symmetric_cbar=True)
    #plotting.plot_glass_brain(clust_img, title=clust_name, vmin=vmin, vmax=vmax, colorbar=True, plot_abs=False, symmetric_cbar=True)
    #plotting.show()
    plt.savefig(OUTPUT_BASENAME + "plot_specific_FeatureAgglomeration_cluster_shapsum=%.3f__%s.pdf" % (df.mean_abs_shap.sum(), clust_name.replace("_CSF", "")))
    plt.close()
    



# %%
