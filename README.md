# 2024_petiton_biobd-bsnip-predict-dx
Prediction of diagnosis (Bipolar Disorder (BD) vs Healthy Controls (HC)) using Anatomical MRI.  <br>

Comparison of voxel-based morphometry (VBM) and surface-based morphometry (SBM) features for classification.

## Motivation 
We found a large performance gap between classification using VBM ROI features (with simple ML models as listed bellow) and classification using SBM ROI (Freesurfer-extracted cortical thickness and surface area, and some subcortical volumes).  
We also found that the inclusion of additional subcortical ROI (from 7 per hemisphere to 17 per hemisphere) in Freesurfer extracted features increased classification performance with ML models.  
Additionally, Goto et al. (2022) *Advantages of Using Both Voxel- and Surface-based Morphometry in Cortical Morphology Analysis: A Review of Various Applications.*[DOI:10.2463/mrms.rev.2021-0096](https://10.2463/mrms.rev.2021-0096) suggest that both SBM and VBM measures should be leveraged to study disorders in neuroimaging analyses.

## CV scheme 
In this work, we chose a Leave-One-Site-Out (LOSO) cross-validation scheme, in order to quantify site-effects.  
This type of CV typically hinders performance, but generates more reproducible results.  

## Machine learning (ML) and Deep learning (DL) tests 

ML tests include 5 models (2 linear, 3 non-linear): linear regression, elastic net regularization, multi-layer perceptron, gradient boosting, and support vector machines with radial basis function (RBF) kernels.  <br>

DL tests include RI-DL (randomly initialized weights + CNN with densenet121 backbone), and transfer learning (TL) (same thing with weights initialized using a contrastive learning model trained on healthy controls such that the latent representations of individuals close in age are pulled closer than those with large age gaps). See: Dufumier et al. (2021). *Contrastive Learning with Continuous Proxy Meta-Data for 3D MRI Classification*. [DOI:10.48550/arXiv.2106.08808](https://doi.org/10.48550/arXiv.2106.08808).   <br>

In addition to TL, we also leverage deep ensemble learning (DE), as suggested in Dufumier et al. (2024) *Exploring the potential of representation and transfer learning for anatomical neuroimaging: Application to psychiatry*. [DOI:10.1016/j.neuroimage.2024.120665](https://10.1016/j.neuroimage.2024.120665), by taking the mean predictions of five DL models (5 models were chosen following this study benchmarking the ideal number of such models needed in a deep ensemble: Petiton et al. (2024) *How and why does deep ensemble coupled with transfer learning increase performance in bipolar disorder and schizophrenia classification?* (https://hal.science/hal-04631924)).

## meta-model 
In this work, we proprose a meta-model in the form of a linear regression using stacked training and testing set scores for BD vs HC classification.
We find a significant improvement in performance metrics with 3 features, each corresponding to the scores of the best-performing models for each feature type (an elastic net for SBM ROI, an SVM-RBF for VBM ROI, and 5-DE TL for voxelwise VBM gray matter measures).  
This reinforces the claims made by Goto et al., and suggests that different brain measures (SBM and VBM) at different granularities (ROI and voxelwise) can not only improve classification results, but also encode complementary information describing the brain structure.  






