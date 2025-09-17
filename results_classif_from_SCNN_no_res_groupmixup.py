from utils import read_pkl, get_predict_sites_list
import numpy as np
pos_weight_dict = {"Baltimore":1.175,
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
path = "/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/models/SBM_TL/test_pretraining_no_residualization/"
roc_aucs=[]
baccs= []
for site in get_predict_sites_list():
    path_one_site =path + f"model_epoch_100_encoder.pthresults_TL_ep200_dr0.2_bs128_wd5e-05_pw{pos_weight_dict[site]}_lr0.0001_site{site}_gamma0.85_no_val_results.pkl"
    data = read_pkl(path_one_site)
    print(site, "  ROC-AUC : ",data["test_metrics"]["roc_auc"], "  balanced-accuracy : ",data["test_metrics"]["balanced_accuracy"])
    roc_aucs.append(data["test_metrics"]["roc_auc"])
    baccs.append(data["test_metrics"]["balanced_accuracy"])

print("means over the 12 sites:")
print("ROC-AUC: ",np.mean(roc_aucs))
print("balanced-accuracy: ",np.mean(baccs))

"""
ROC-AUC:  0.6177245438711297
balanced-accuracy:  0.5777011007247664
"""