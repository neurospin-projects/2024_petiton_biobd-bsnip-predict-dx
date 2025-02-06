# %% Imports

import numpy as np
import pandas as pd
import pickle
import os.path
import glob

import matplotlib.pyplot as plt
import seaborn as sns

# Statmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats

# ML
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap


# %% Path

WD = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx"

# %% Read data

participants = pd.read_csv(os.path.join(WD, "data/processed", "participantsBD.csv"))
participants = participants[["participant_id", "sex", "age", "diagnosis", "site"]]
participants.diagnosis.replace({'bipolar disorder':'BD', 'control':'HC', 'psychotic bipolar disorder':'BD'}, inplace=True)
data = pd.read_csv(os.path.join(WD, "data/processed", "roisBD_residualized_and_scaled.csv"))
data = pd.merge(participants, data)

atlas = pd.read_csv(os.path.join(WD, "data/atlases/lobes_Neuromorphometrics.csv"), sep=';')
                    

# %% Univariate statistics

# Replace +- not alowed in statsmodel formulae
data.columns = data.columns.str.replace('-', '_MINUS_') 
data.columns = data.columns.str.replace('+', '_PLUS_') 

stats = list()

for var_ in data.loc[:, 'TIV':].columns:
    print(var_)
    lm_ = smf.ols('%s ~ diagnosis + sex + age + site' % var_, data).fit()
    #print(lm_.model.data.param_names)
    aov_ = sm.stats.anova_lm(lm_, typ=2)

    stats_ = [var_] +\
    lm_.tvalues[['diagnosis[T.HC]', 'sex', 'age']].tolist() + \
    lm_.pvalues[['diagnosis[T.HC]', 'sex', 'age']].tolist() + \
    [aov_.loc['diagnosis', "F"], aov_.loc['diagnosis', "PR(>F)"]]

    stats.append(stats_)

stats = pd.DataFrame(stats, columns=
    ['ROI', 'diag_t', 'sex_t', 'age_t', 'diag_p', 'sex_p', 'age_p', 'site_f', 'site_p'])

# Change back to original variable names
stats.ROI = stats.ROI.str.replace('_MINUS_', '-')
stats.ROI = stats.ROI.str.replace('_PLUS_', '+')
data.columns = data.columns.str.replace('_MINUS_', '-')
data.columns = data.columns.str.replace('_PLUS_', '+')

stats.to_excel(os.path.join(WD, "models/statistics_univ", "statsuniv_roisBD.xlsx"),
    sheet_name='statsuniv_roisBD_residualized_and_scaled', index=False)

# %% Compute Shap values

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
        to_csv(os.path.join(WD, "models/ShapValues", "SHAP_randomized_%i.csv" % i), index=False)
    shap_values_list.append(shap_values)

# print("shap_values =", shap_values)
# print("base value =", explainer.expected_value)
# shap_values.shape == (861, 284)
#shap.plots.waterfall(shap_values[0])

# %% Select variables from shap values

shap_df = pd.read_excel(os.path.join(WD, "models/ShapValues/SHAP_summary.xlsx"))
shap_df = shap_df[['fold', 'ROI', 'mean_abs_shap']]


# Read randomized SHAP values

# FIX START
cols = pd.read_csv(WD+"/models/ShapValues/SHAP_randomized_0.csv").columns

files_ = glob.glob(os.path.join(WD, "models/ShapValues", 'SHAP_randomized_*.{}'.format("csv")))
for f in files_:
    df_ = pd.read_csv(f)
    df_.columns = cols
    df_.to_csv(f, index=False)
# FIX END

shap_rnd_absmean = pd.DataFrame([pd.read_csv(f).abs().mean() for f in files_])

# SHAP Statistics mean,std, n, CI 
m = shap_df.groupby(['ROI'])["mean_abs_shap"].mean()
s = shap_df.groupby(['ROI'])["mean_abs_shap"].std()
n = shap_df.groupby(['ROI'])["mean_abs_shap"].count()
# Critical value for t at alpha / 2:
t_alpha2 = -scipy.stats.t.ppf(q=0.05/2, df=n-1, loc=0)
ci_low = m - t_alpha2 * s / np.sqrt(n)
ci_high = m + t_alpha2 * s / np.sqrt(n)

# Mean under H0
m_h0 = shap_rnd_absmean.mean()
# set(m.index) - set(m_h0.index)
# set(m.index).intersection(set(m_h0.index))

m_h0 = m_h0[m.index]

shap_stat = dict(ROI=m.index, mean_abs_shap=m, ci_low=ci_low, ci_high=ci_high,
                 mean_abs_shap_h0=m_h0, select = m_h0 < ci_low)
shap_stat = pd.DataFrame(shap_stat)
shap_stat.sort_values(by="mean_abs_shap", ascending=False, inplace=True)

stat_univ = pd.read_excel(os.path.join(WD, "models/statistics_univ", "statsuniv_roisBD.xlsx"),
                          sheet_name='statsuniv_roisBD_residualized_and_scaled')


shap_stat = pd.merge(shap_stat.reset_index(drop=True), stat_univ.reset_index(drop=True),
                     on='ROI', how='left')

shap_stat = pd.merge(shap_stat, atlas[['ROIabbr', 'ROIname']], how='left', left_on='ROI', right_on='ROIabbr')



shap_stat.to_excel(os.path.join(WD, "models/ShapValues", "SHAP_roi_univstat.xlsx"),
    sheet_name='SHAP_roi_univstat', index=False)

# Specific variables

rois_names_spe = ['lSupParLo_CSF_Vol', 'rSupParLo_CSF_Vol', 'lSupTemGy_CSF_Vol']

# %% Split variable into specifics and suppressors

# Suppressor variables

rois_names_sup = ['rSupTemGy_CSF_Vol', 'lTemPo_CSF_Vol', 'rTemPo_CSF_Vol']

# %% SEM


print("TOTO")