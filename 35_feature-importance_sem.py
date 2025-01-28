# %% Imports

import numpy as np
import pandas as pd
import pickle
import os.path

# Statmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats

# %% Path

WD = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx"

# %% Read data

participants = pd.read_csv(os.path.join(WD, "data/processed", "participantsBD.csv"))
participants = participants[["participant_id", "sex", "age", "diagnosis", "site"]]
participants.diagnosis.replace({'bipolar disorder':'BD', 'control':'HC', 'psychotic bipolar disorder':'BD'}, inplace=True)
data = pd.read_csv(os.path.join(WD, "data/processed", "roisBD_residualized_and_scaled.csv"))
df = pd.merge(participants, data)
df.columns = df.columns.str.replace('-', '') 
df.columns = df.columns.str.replace('+', '') 


# %% Univariate statistics

stats = list()

for var_ in df.loc[:, 'TIV':].columns:
    print(var_)
    lm_ = smf.ols('%s ~ diagnosis + sex + age + site' % var_, df).fit()
    #print(lm_.model.data.param_names)
    aov_ = sm.stats.anova_lm(lm_, typ=2)

    stats_ = [var_] +\
    lm_.tvalues[['diagnosis[T.HC]', 'sex', 'age']].tolist() + \
    lm_.pvalues[['diagnosis[T.HC]', 'sex', 'age']].tolist() + \
    [aov_.loc['diagnosis', "F"], aov_.loc['diagnosis', "PR(>F)"]]

    stats.append(stats_)

stats = pd.DataFrame(stats, columns=
    ['var', 'diag_t', 'sex_t', 'age_t', 'diag_p', 'sex_p', 'age_p', 'site_f', 'site_p'])


stats.to_excel(os.path.join(WD, "models/statistics_univ", "statsuniv_roisBD.xlsx"),
    sheet_name='statsuniv_roisBD_residualized_and_scaled', index=False)


# %% Select variables from shap values

shap = pd.read_excel(os.path.join(WD, "models/ShapValues/SHAP_summary.xlsx"))
shap = shap[['fold', 'ROI', 'mean_abs_shap']]

shap.ROI = shap.ROI.str.replace('-', '') 
shap.ROI = shap.ROI.str.replace('+', '') 

m = shap.groupby(['ROI'])["mean_abs_shap"].mean()
s = shap.groupby(['ROI'])["mean_abs_shap"].std()
n = shap.groupby(['ROI'])["mean_abs_shap"].count()

# Critical value for t at alpha / 2:
t_alpha2 = -scipy.stats.t.ppf(q=0.05/2, df=n-1, loc=0)
ci_low = m - t_alpha2 * s / np.sqrt(n)
ci_high = m + t_alpha2 * s / np.sqrt(n)



# Specific variables

rois_names_spe = ['lSupParLo_CSF_Vol', 'rSupParLo_CSF_Vol', 'lSupTemGy_CSF_Vol']

# %% Split variable into specifics and suppressors

# Suppressor variables

rois_names_sup = ['rSupTemGy_CSF_Vol', 'lTemPo_CSF_Vol', 'rTemPo_CSF_Vol']

# %% SEM


print("TOTO")