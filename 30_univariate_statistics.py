# %% Imports

import numpy as np
import pandas as pd
import pickle
import os.path
import glob

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Statmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats
from statsmodels.stats.multitest import multipletests


# %% Path

WD = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx"
OUTPUT = os.path.join(WD, "models/statistics_univ/statsuniv_roisBD.xlsx")

# %% Read data

participants = pd.read_csv(os.path.join(WD, "data/processed", "participantsBD.csv"))
participants = participants[["participant_id", "sex", "age", "diagnosis", "site"]]
participants.diagnosis.replace({'bipolar disorder':'BD', 'control':'HC', 'psychotic bipolar disorder':'BD'}, inplace=True)
data = pd.read_csv(os.path.join(WD, "data/processed", "roisBD_residualized_and_scaled.csv"))
data = pd.merge(participants, data)

# atlas = pd.read_csv(os.path.join(WD, "data/atlases/lobes_Neuromorphometrics.csv"), sep=';')

# shap_df = pd.read_excel(os.path.join(WD, "models/ShapValues/shap_computed_from_all_Xtrain/SHAP_summary.xlsx"))


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

_, pcor, _, _  = multipletests(stats.diag_p, method='hs')
np.sum(pcor < 0.05) == 130

_, pcor, _, _  = multipletests(stats.diag_p, method='bonferroni')
np.sum(pcor < 0.05) == 122

stats['diag_pcor'] = pcor

stats.to_excel(OUTPUT,
    sheet_name='statsuniv_roisBD_residualized_and_scaled', index=False)
