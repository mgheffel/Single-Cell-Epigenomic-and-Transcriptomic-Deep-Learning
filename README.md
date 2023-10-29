# Single-Cell-Epigenomic-and-Transcriptomic-Deep-Learning
Deep Neural Networks for single cell epigenomic and transcriptomic dataset imputation, integration, and generation. 

# Models:  
**Linear denoising:**  
Variational Autoencoder encoding and reconstructing single cell gene expression data and integrating with original data  
**Split Cross Modality Generation:**  
Variatitional Autoencoder encoding and reconstructing single cell gene expression data and generating associated single cell ATAC peak data

**UCLA_DEEP-LEARNING_COMSCI260C:**  
Varying degrees of complexity of variational autoencoders (VAEs) implemented and evaluated on their ability to impute missing values and classify cell types on single-cell whole genome bisulfite sequenceing gene body methylation fraction features.
Models tested: 
  vanilla VAE 
  VAE with custom loss to not penalize imputed missing values in training
  VAE with custom dropout for missing features in training
  VAE with custom dropout and loss, combining two models above
  
