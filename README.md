# NuGraph3DA: Domain Adaptation for NuGraph

Domain Adaptaiton (DA) is currently implemented in the event decoder (semantic decoder as well, but it is manually tirned off because it should be moved to the latest version of NG3 with true hierachy). Regular event loss uses labels from both datasets (no need to pretend one is unlabeled).
New and/or modified files:
 - **event_da_dann** - Utilizing Domain Adversarial Neural Network (DANN) for DA. Labels are used from both datasets for the regular event decoder loss.
 - **semantic_da_dann** -  DANN with labels used from both datasets for the semantic loss. DA is turned off by default but can be turn on manually.

 - **event_da_mmd** - DA implemented using Maximum Mean Discrepancy (MMD).
 - **semantic_da_mmd** -  DA implemented using Maximum Mean Discrepancy (MMD).
   
 - **event_da_semantic** - DA implemented using Semantic Loss, which places instances with the same loss close to each other in latent space. Because we use labels from both datasets we can use this loss as well. If one dataset is unlabeled, this loss cannot be used.
 - **semantic_da_semantic** - DA implemented using Semantic Loss.

- **event_da_sinkhorn** - DA implemented using Sinhorn Loss as the distance metric.
- **semantic_da_sinkhorn** - DA implemented using Sinhorn Loss as the distance metric.

Other files which are updated:

- **EmbeddingPlotter** - choose between tSNE, Isomap and UMAP to view the eevent or semantic latent space (semantic needs to be moved to the latest NG version)
- **DANNLoss** - Domain adversarial NN, with domain classifier and gradient reversal
- **MMDLoss** - Maximum Mean Discrepancy Loss
- **SemanticLoss** - minimizing distance (Euclidean or Cosine) of samples with the same labels (both datasets need to be labeld)
- **SinkhornLoss**  - using GeomLoss package for sinkhorn implementation of Optimal Transport distance (via Geomloss we can utilize many options and types of distances)
- **H5Dataset** - now includes classes CombinedDataset and CombinedDatasetCycle (to cycle shorter dataset)
- **H5DataModuleDA** - using the new double batch combined data
- **nugraph3_da_event_semantic** - new with DA in event and optionally in semantic (labels used from both datasets)
- **nugraph3_da** - old version with DA just in event decoder (and no target labels)


_______________________________________________
