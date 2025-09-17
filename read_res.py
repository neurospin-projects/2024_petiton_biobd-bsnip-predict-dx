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

meanall=[]
for i in [1,2,3,4,10]:
    meanauc=[]
    for site in get_predict_sites_list():
        path = f"/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/models/SBM_TL/test_pretraining_with_in_img_normalization/model_epoch_300_encoder.pthresults_TL_ep200_dr0.2_bs128_wd5e-05_pw{pos_weight_dict[site]}_lr0.0001_site{site}_gamma0.85_no_val_results.pkl"
        
        
        # path=f"/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/models/SBM_TL/test_pretraining_with_age_and_sex_res_no_standardization/encoder.pthresults_TL_ep200_dr0.2_bs128_wd5e-05_pw{pos_weight_dict[site]}_lr0.0001_site{site}_gamma0.85_no_val_results.pkl"
        # path=f"/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/models/SBM_TL/test_pretraining_without_normalization/encoder.pthresults_TL_ep200_dr0.2_bs128_wd5e-05_pw{pos_weight_dict[site]}_lr0.0001_site{site}_gamma0.85_no_val_results.pkl"
        # path=f"/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/models/SBM_TL/test_pretraining_with_age_and_sex_res_standardization/run{i}N_8/encoder.pthresults_TL_ep200_dr0.2_bs128_wd5e-05_pw{pos_weight_dict[site]}_lr0.0001_site{site}_gamma0.85_no_val_results.pkl"
        
        # path=f"/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/models/SBM_TL/test_pretraining_with_age_and_sex_site_mulm_res_standardization/encoder.pthresults_TL_ep200_dr0.2_bs128_wd5e-05_pw{pos_weight_dict[site]}_lr0.0001_site{site}_gamma0.85_no_val_results.pkl"

        # path = f"/neurospin/signatures/2024_petiton_biobd-bsnip-predict-dx/models/SBM_TL/test_pretraining_with_age_and_sex_res_standardization_with_pretraining_scaler/encoder.pthresults_TL_ep200_dr0.2_bs128_wd5e-05_pw{pos_weight_dict[site]}_lr0.0001_site{site}_gamma0.85_no_val_results.pkl"
        data = read_pkl(path)
        # print(site, "  ",data["test_metrics"]["roc_auc"])
        meanauc.append(data["test_metrics"]["roc_auc"])

    print(i,"  ",np.mean(meanauc))
    meanall.append(np.mean(meanauc))

print("mean ",np.mean(meanall))

# test_pretraining_with_in_img_normalization (so age + sex + site residualization, 
# standardization for pretraining, and normalization for finetuning)
# Baltimore    0.5989010989010989
# Boston    0.7762345679012346
# Dallas    0.5300207039337475
# Detroit    0.8333333333333333
# Hartford    0.6225961538461539
# mannheim    0.6355263157894737
# creteil    0.6075851393188854
# udine    0.6608024691358025
# galway    0.607843137254902
# pittsburgh    0.6115288220551377
# grenoble    0.6570048309178744
# geneve    0.6042857142857143
# 0.6454718572227799

# test_pretraining_with_age_and_sex_res_standardization
# Baltimore    0.5446428571428572
# Boston    0.7901234567901234
# Dallas    0.5652173913043479
# Detroit    0.8015873015873016
# Hartford    0.6033653846153847
# mannheim    0.6947368421052631
# creteil    0.6517027863777091
# udine    0.6700617283950617
# galway    0.7058823529411764
# pittsburgh    0.6422305764411027
# grenoble    0.6328502415458936
# geneve    0.5828571428571429
# 0.6571048385086137

# test_pretraining_with_age_and_sex_res_standardization_with_pretraining_scaler
# Baltimore    0.6078296703296704
# Boston    0.6172839506172839
# Dallas    0.6149068322981367
# Detroit    0.6666666666666667
# Hartford    0.5865384615384615
# mannheim    0.6677631578947368
# creteil    0.7097523219814241
# udine    0.7157407407407408
# galway    0.6274509803921569
# pittsburgh    0.6234335839598997
# grenoble    0.5797101449275363
# geneve    0.6671428571428571
# 0.6403516140407975




