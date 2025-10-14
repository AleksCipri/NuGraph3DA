# NuGraph3DA: Domain Adaptation for NuGraph

Domain Adaptaiton (DA) is currently implemented in the event decoder (semantic decoder as well, but it is manually tirned off because it should be moved to the latest version of NG3 with true hierachy). Regular event loss uses labels from both datasets (no need to pretend one is unlabeled).
 - event_da_dann - Utilizing Domain Adversarial Neural Network (DANN) for DA. Labels are used from both datasets for the regular event decoder loss.
 - semantic_da_dann -  DANN with labels used from both datasets for the semantic loss. DA is turned off by default but can be turn on manually.

 - event_da_mmd - DA implemented using Maximum Mean Discrepancy (MMD).
 - semantic_da_mmd -  DA implemented using Maximum Mean Discrepancy (MMD).
   
 - event_da_semantic - DA implemented using Semantic Loss, which places instances with the same loss close to each other in latent space. Because we use labels from both datasets we can use this loss as well. If one dataset is unlabeled, this loss cannot be used.
 - semantic_da_semantic - DA implemented using Semantic Loss.

- event_da_sinkhorn - DA implemented using Sinhorn Loss as the distance metric.
- semantic_da_sinkhorn - DA implemented using Sinhorn Loss as the distance metric.

Files which are updated:

- EmbeddingPlotter - choose between tSNE, Isomap and UMAP to view the eevent or semantic latent space (semantic needs to be moved to the latest NG version)
 - DANNLoss - Domain adversarial NN, with domain classifier and gradient reversal
- MMDLoss - Maximum Mean Discrepancy Loss
- SemanticLoss - minimizing distance (Euclidean or Cosine) of samples with the same labels (both datasets need to be labeld)
- SinkhornLoss  - using GeomLoss package for sinkhorn implementation of Optimal Transport distance (via Geomloss we can utilize many options and types of distances)
- H5Dataset - now includes classes CombinedDataset and CombinedDatasetCycle (to cycle shorter dataset)
- H5DataModuleDA - using the new double batch combined data
- nugraph3_da_event_semantic - new with DA in event and optionally in semantic (labels used from both datasets)
- nugraph3_da - old version with DA just in event decoder (and no target labels)


_______________________________________________
# NuGraph: a Graph Neural Network (GNN) for neutrino physics event reconstruction

This repository contains a GNN architecture for reconstructing particle interactions in neutrino physics detector environments. Its primary function is the classification of detector hit particle type through semantic segmentation, with additional secondary functions such as background hit rejection, event classification, clustering and vertex reconstruction.

## Installation

This repository can be installed in Python via `pip`, although using Anaconda to install dependencies is strongly recommended. Detailed instructions on how to easily install all necessary dependencies are available [here](https://pynuml.readthedocs.io/en/latest/install/installation.html).

Once dependencies are installed, you can simply clone this repository and installing it via `pip` – if you intend to carry out any development on the code, installing in editable mode is recommended:

```
git clone git@github.com:exatrkx/NuGraph
pip install --no-deps -e ./NuGraph
```

## Training a model

You can train the model using a processed graph dataset as input by executing the `train.py` script in the `scripts` subdirectory. This script accepts many arguments to configure your training – for a complete summary of all available arguments, you can simply run

```
scripts/train.py --help
```

As an example, to train the network for semantic segmentation on the Heimdall cluster, one might run

```
scripts/train.py --data-path /raid/uboone/NuGraph2/NG2-paper.gnn.h5 \
                 --logdir /raid/$USER/logs --name default --version semantic-filter \
                 --semantic --filter
```

This command would start a network training using the requested input dataset, training with the semantic head enabled, and writing network parameters and metrics to the directory `/raid/$USER/logs/default/semantic-filter`.

### Training on SLURM clusters

If you're working on a cluster that uses the SLURM batch submission system, such as the Wilson cluster at Fermilab, then you'll need to submit training via a batch script instead. An example batch script `train_batch.sh` is included in the `scripts` subdirectory. If you're training on the Wilson cluster, you can submit a training job by running
```
sbatch scripts/train_batch.sh <args>
```
where `<args>` are the same argument you'd pass if you were executing the training script locally.

If you're training on a SLURM environment other than the Wilson cluster, you'll need to edit the SLURM directives in the script appropriately for the cluster you're working on before submitting.

### Metric logging

In the above example, training outputs including logging metrics would be written to a subdirectory of `/raid/$USER/logs`. We can use the Tensorboard interface to visualise these metrics and track the network's training progress. You can start Tensorboard using the following command:

```
tensorboard --port XXXX --bind_all --logdir /raid/$USER/logs --samples_per_plugin 'images=200'
```

In the above example, you should replace `XXXX` with a unique port number of your choosing. Provided you're forwarding that port when working over SSH, you can then access the interface in a local browser at `localhost:XXXX`.


