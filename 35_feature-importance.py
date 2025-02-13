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


# %% Path

WD = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx"

# %% Read data

participants = pd.read_csv(os.path.join(WD, "data/processed", "participantsBD.csv"))
participants = participants[["participant_id", "sex", "age", "diagnosis", "site"]]
participants.diagnosis.replace({'bipolar disorder':'BD', 'control':'HC', 'psychotic bipolar disorder':'BD'}, inplace=True)
data = pd.read_csv(os.path.join(WD, "data/processed", "roisBD_residualized_and_scaled.csv"))
data = pd.merge(participants, data)

atlas = pd.read_csv(os.path.join(WD, "data/atlases/lobes_Neuromorphometrics.csv"), sep=';')

shap_df = pd.read_excel(os.path.join(WD, "models/ShapValues/shap_computed_from_all_Xtrain/SHAP_summary.xlsx"))

# %% Randomize SHAP values
# ========================
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
# Output: "models/ShapValues/SHAP_roi_univstat.xlsx"

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
shap_stat = pd.merge(shap_stat, atlas[['ROIabbr', 'ROIname']], how='left', left_on='ROI', right_on='ROIabbr')

# Write
shap_stat.to_excel(os.path.join(WD, "models/ShapValues/SHAP_roi_univstat.xlsx"),
    sheet_name='SHAP_roi_univstat', index=False)


# %% Split variable into specifics and suppressors
# =================================================

# Input: "models/ShapValues/SHAP_roi_univstat.xlsx"
"""
https://www.journals.uchicago.edu/doi/pdf/10.5243/jsswr.2010.2
"A suppressor variable correlates with other independent variables, and accounts for or suppresses some outcome-irrelevant variation or errors in one or more other predictors, and improves the overall predictive power of the model."
"""

def split_variables_into_specific_suppressor():
    shap_stat = pd.read_excel(os.path.join(WD, "models/ShapValues/SHAP_roi_univstat.xlsx"),
        sheet_name='SHAP_roi_univstat')

    # Filter Features based on significance of SHAP values
    shap_stat = shap_stat[shap_stat.select == 1]
    assert shap_stat.shape[0] == 116

    # Split fature in two: specific and suppressor
    shap_spec = shap_stat[shap_stat.diag_pcor < 0.05]
    shap_suppr = shap_stat[shap_stat.diag_p > 0.05]
    return shap_spec, shap_suppr
    
# %% Illustrate Suppressor variable with linear model

shap_spec, shap_suppr = split_variables_into_specific_suppressor()

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
g.ax_joint.set_xlabel("Score Supr (AUC=%.2f)" % score_supr_auc)
g.ax_joint.set_ylabel("Score Spec (AUC=%.2f)" % score_spec_auc)
g.fig.suptitle("Scatter Plot with Marginal Density Plots", y=1.02)
plt.show()

# Density (KDE) Plot

plt.figure(figsize=(6, 2))
sns.kdeplot(data=df, x='score_tot', hue='diagnosis', fill=True, alpha=0.3)
plt.xlabel("Score Tot (AUC=%.2f)" % score_tot_auc)
plt.title("Density Plot of Score Tot by Diagnosis")
#plt.grid(True)
plt.show()


# %% Exploratory analysis (PCA & Correlation matrix)

shap_spec, shap_suppr = split_variables_into_specific_suppressor()

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

#from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import FeatureAgglomeration

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


X = data[shap_spec.ROI]
csf_cols = [col for col in shap_spec.ROI if 'CSF' in col]
X[csf_cols] = -1 * X[csf_cols]
ss = StandardScaler()
X = ss.fit_transform(X)
Xdf = pd.DataFrame(X, columns=shap_spec.ROI)

# setting distance_threshold=0 ensures we compute the full tree.
#model = AgglomerativeClustering(distance_threshold=0, n_clusters=4)
model = FeatureAgglomeration(n_clusters=9, compute_distances=True)
model = model.fit(X)
Xr = model.transform(X)

#plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
dendro = plot_dendrogram(model, color_threshold=45)
reorder_idx = np.array([int(idx) for idx in dendro["ivl"]])
reorder_columns = Xdf.columns[reorder_idx]
reorder_columns_label = [name + '_' + str(clust) for name, clust in
  zip(reorder_columns, model.labels_[reorder_idx])]

loc, _ = plt.xticks()
plt.xticks(loc, labels=reorder_columns_label)#, rotation=45)
plt.show()

# Plot Corr Mat
import matplotlib.patches as patches

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
plt.show()


# %% SEM


print("TOTO")