# 2024_petiton_biobd-bsnip-predict-dx
Prediction of diagnosis (Bipolar Disorder (BD) vs Healthy Controls (HC)) using Anatomical MRI.  
Comparison of voxel-based morphometry (VBM) and surface-based morphometry (SBM) features for classification, at varying granularities (ROI vs. voxel- or vertex-wise).
We find VBM and SBM features to be synergistic, with a combiner model outperforming all other classifiers, yielding a classification ROC-AUC of over 75%...
The paper for this code can be found as a preprint here: https://hal.science/hal-05571683/. <br>


## Motivation
Although model architecture is often thoroughly explored in ML and DL analyses of neuroimaging data, feature representations remain rarely analyzed. 
Preliminary experiments revealed a considerable performance gap between classification performance using VBM ROI features (cat12 derived) vs. SBM ROI features (Freesurfer-extracted cortical thickness and surface area, and some subcortical volumes), with VBM ROI yielding ROC-AUCs about 10% higher than SBM ROI. Additionally, these preliminary experiments showed that the inclusion of subcortical regions in the SBM ROI case (from 7 to 17 per hemisphere) improved classification performance (while still under-performing compared to VBM ROI features).
Goto et al. (2022) *Advantages of Using Both Voxel- and Surface-based Morphometry in Cortical Morphology Analysis: A Review of Various Applications.*[DOI:10.2463/mrms.rev.2021-0096](https://10.2463/mrms.rev.2021-0096) suggested that both SBM and VBM measures should be leveraged to study disorders in neuroimaging analyses.
Our work thus explores how different feature representations of structural neuroanatomical measures, used with their corresponding model architectures, play a role in improving classification performance, potentially revealing different aspects of illness. 

## CV scheme 
In this work, we chose a Leave-One-Site-Out (LOSO) cross-validation scheme, to optimally quantify site-effects.  
This type of CV typically hinders performance, but generates more reproducible results, which we prioritized.  

## Machine learning (ML) and Deep learning (DL) models 

ML tests include 5 models (2 linear, 3 non-linear): 
- linear regression
- elastic net regularization
- multi-layer perceptron
- gradient boosting
- support vector machines with radial basis function (RBF) kernels.  <br>

DL tests include:
- RI-DL: randomly initialized weights + CNN with a densenet121 backbone
- transfer learning (TL): same architecture as RI-DL, with weights initialized using a contrastive learning model, trained to bring individuals close in age closer in the latent space and individuals with distant ages further appart using a large (around 10k) cohort of healthy controls (see: Dufumier et al. (2021). *Contrastive Learning with Continuous Proxy Meta-Data for 3D MRI Classification*. [DOI:10.48550/arXiv.2106.08808](https://doi.org/10.48550/arXiv.2106.08808)).
- 5-DE RI-DL: a deep ensemble (DE) of five randomly initialized models
- 5-DE TL: a deep ensemble (DE) of five transfer learning models (all using the same pretrained weights described above) <br>

Using deep ensembles was suggested in Dufumier et al. (2024) *Exploring the potential of representation and transfer learning for anatomical neuroimaging: Application to psychiatry*. [DOI:10.1016/j.neuroimage.2024.120665](https://10.1016/j.neuroimage.2024.120665), and consists in taking the mean predictions of five DL models.
We chose ensembles of five models following a previous study in which we benchmarked the ideal number of models needed in a deep ensemble in similar applications: Petiton et al. (2024) *How and why does deep ensemble coupled with transfer learning increase performance in bipolar disorder and schizophrenia classification?* (https://hal.science/hal-04631924)).

## Combiner model 
We proprose a new combiner model in the form of a linear regression using stacked training and testing set scores for BD vs HC classification.
We find a significant improvement in performance metrics with 3 features, each corresponding to the scores of the best-performing models for each feature type (an elastic net for SBM ROI, an SVM-RBF for VBM ROI, and 5-DE TL for voxelwise VBM gray matter measures).  
This reinforces the claims made by Goto et al., and suggests that different brain measures (SBM and VBM) at different granularities (ROI and voxelwise) can not only improve classification results, but also encode complementary synergistic information differentiating the brain structure of healthy controls and bipolar disorder patients.






