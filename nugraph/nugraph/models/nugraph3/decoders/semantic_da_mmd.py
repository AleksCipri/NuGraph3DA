"""NuGraph3 semantic decoder with domain adaptation (MMD)"""
from typing import Any
import torch
from torch import nn
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sn
import torchmetrics as tm
from ....util import RecallLoss
from ..types import Data

from ....util.MMDLoss import MMDLoss
from ....util.EmbeddingPlotter import CombinedEmbeddingPlot


class SemanticDecoderDAmmd(nn.Module):
    """
    NuGraph3 semantic decoder module

    Convolve planar node embedding down to a set of categorical scores for
    each semantic class.

    If turned on,  use maximum Mean Discrepancy (MMD) loss on planar node embedding 
    to align them across the source and target dataset and improve classification. Default is no DA for this decoder 
    (but both datasets use labes for semantic loss).

    Args:
        node_features: Number of planar node features
        planes: List of detector planes
        semantic_classes: List of semantic classes
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str],
                 warmup_epochs: int = 0):
        super().__init__()
        self.warmup_epochs = warmup_epochs  # When to start with domain adaptation.
        self.use_domain_adaptation = False  # Will be updated by main model.
        self.semantic_classes = semantic_classes

        # Loss function
        self.loss = RecallLoss()
        self.loss_mmd = MMDLoss()  # Optional Domain Adapttaion - MMD
        mmd = torch.tensor(0.0)   

        # Temperature parameters are now handling scaling of the losses
        self.tempS = nn.Parameter(torch.tensor(0.))
        self.tempT = nn.Parameter(torch.tensor(0.))
        self.temp_DA = nn.Parameter(torch.tensor(0.))

        # Metrics
        metric_args = {
            "task": "multiclass",
            "num_classes": len(semantic_classes),
            "ignore_index": -1
        }
        self.recall_s = tm.Recall(**metric_args)
        self.precision_s = tm.Precision(**metric_args)
        self.cm_recall_s = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision_s = tm.ConfusionMatrix(normalize="pred", **metric_args)

        self.recall_t = tm.Recall(**metric_args)
        self.precision_t = tm.Precision(**metric_args)
        self.cm_recall_t = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision_t = tm.ConfusionMatrix(normalize="pred", **metric_args)
        
        self.embeddings = CombinedEmbeddingPlot(method="umap") # Include latent space plotting 

        # network
        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Linear(node_features, len(semantic_classes))
        self.classes = semantic_classes

        
    def forward(self, dataS: Data, dataT: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 semantic decoder forward pass with both source and target data

        Args:
            dataS: Graph data object (source)
            dataT: Graph data object (target)
            stage: Stage name (train/val/test)
        """

        # Run network and add output to graph object
        # Souce and Target data (dataS and dataT)
        for p, net in self.net.items():
            dataS[p].x_semantic = net(dataS[p].x)
            dataT[p].x_semantic = net(dataT[p].x)
            if isinstance(dataS, Batch):
                dataS._slice_dict[p]["x_semantic"] = dataS[p].ptr
                incS = torch.zeros(dataS.num_graphs, device=dataS[p].x.device)
                dataS._inc_dict[p]["x_semantic"] = incS
            if isinstance(dataT, Batch):
                dataT._slice_dict[p]["x_semantic"] = dataT[p].ptr
                incT = torch.zeros(dataT.num_graphs, device=dataT[p].x.device)
                dataT._inc_dict[p]["x_semantic"] = incT
    
        # Calculate losses
        # Source data
        xS = torch.cat([dataS[p].x_semantic for p in self.net], dim=0)
        yS = torch.cat([dataS[p].y_semantic for p in self.net], dim=0)
        wS = 2 * (-1 * self.tempS).exp()
        lossS = wS * self.loss(xS, yS) + self.tempS
        
        # Target data
        xT = torch.cat([dataT[p].x_semantic for p in self.net], dim=0)
        yT = torch.cat([dataT[p].y_semantic for p in self.net], dim=0)
        wT = 2 * (-1 * self.tempT).exp()
        lossT = wT * self.loss(xT, yT) + self.tempT

        
        if self.use_domain_adaptation:
            """
            Domain Adaptation is optional and is implemented via MMD (lossDA). 
            Both source and target labels are being used to calculate semantic losses (lossS and lossT).
            All losses are being scaled via their own temperatures, which are all trainable parameters.
            DA loss is also capped to be at most 1/4 of the lossS (to be on the safe side). 
            """
            #print("Using semantic DA!")

            # MMD Alignment
            wDA = 2 * (-1 * self.temp_DA).exp()
            raw_lossDA = wDA * self.loss_mmd(dataS[p].x_semantic, dataT[p].x_semantic) + self.temp_DA #should this be xS and xT?

            # Smooth capping of the DA based on the source event loss value (currently to be at most 1/4 of the event loss value)
            sharp = 20.0  # sharpness of transition when source event loss goes from positive to negative
            sig = torch.sigmoid(sharp * lossS)
            max_lossDA = sig * (lossS / 4) + (1 - sig) * (4 * lossS)
            lossDA = torch.min(raw_lossDA, max_lossDA)

            # Total loss
            loss = lossS + lossT + lossDA

        else:
            #print("No DA in semantic!")
            loss = lossS + lossT

        
        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"loss_semantic/{stage}"] = loss
            metrics[f"recall_semantic_source/{stage}"] = self.recall_s(xS, yS)
            metrics[f"precision_semantic_source/{stage}"] = self.precision_s(xS, yS)
            metrics[f"recall_semantic_target/{stage}"] = self.recall_t(xT, yT)
            metrics[f"precision_semantic_target/{stage}"] = self.precision_t(xT, yT)
            metrics[f"loss_semantic_source/{stage}"] = lossS
            metrics[f"loss_semantic_target/{stage}"] = lossT

            if self.use_domain_adaptation:
                metrics[f"DA_loss_capped_semantic/{stage}"] = lossDA
                metrics[f"DA_loss_uncapped_semantic/{stage}"] = raw_lossDA 
                
        if stage == "train":
            metrics["temperature/semantic_source"] = self.tempS
            metrics["temperature/semantic_target"] = self.tempT
            metrics["temperature/semantic_DA"] = self.temp_DA
            
        if stage in ["val", "test"]:
            self.cm_recall_s.update(xS, yS)
            self.cm_precision_s.update(xS, yS)
            self.cm_recall_t.update(xT, yT)
            self.cm_precision_t.update(xT, yT)
            self.embeddings.update(dataS[p].x_semantic, yS, dataT[p].x_semantic, yT)

        # apply softmax to prediction
        for p in self.net:
            dataS[p].x_semantic = dataS[p].x_semantic.softmax(dim=1)
            dataT[p].x_semantic = dataT[p].x_semantic.softmax(dim=1)

        return loss, metrics

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        """
        Draw confusion matrix

        Args:
            cm: Confusion matrix object
        """
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(confusion,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel("Assigned label")
        plt.ylabel("True label")
        return fig

    def on_epoch_end(self,
                     logger: TensorBoardLogger,
                     stage: str,
                     epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
        if not logger:
            return

        logger.experiment.add_figure(f"recall_semantic_matrix_source/{stage}",
                                     self.draw_confusion_matrix(self.cm_recall_s),
                                     global_step=epoch)
        self.cm_recall_s.reset()

        logger.experiment.add_figure(f"recall_semantic_matrix_target/{stage}",
                                     self.draw_confusion_matrix(self.cm_recall_t),
                                     global_step=epoch)
        self.cm_recall_t.reset()

        logger.experiment.add_figure(f"precision_semantic_matrix_source/{stage}",
                                self.draw_confusion_matrix(self.cm_precision_s),
                                global_step=epoch)
        self.cm_precision_s.reset()

        logger.experiment.add_figure(f"precision_semantic_matrix_target/{stage}",
                                self.draw_confusion_matrix(self.cm_precision_t),
                                global_step=epoch)
        self.cm_precision_t.reset()

        # Plot the embedding space 
        dat1, lab1, dat2, lab2 = self.embeddings.compute()
        dat1sub, lab1sub = self.embeddings.subsample(dat1, lab1, max_samples=1000)
        dat2sub, lab2sub = self.embeddings.subsample(dat2, lab2, max_samples=1000)
        
        embeddings_fig = self.embeddings.plot_combined(dat1sub, lab1sub, dat2sub, lab2sub, epoch=epoch, class_names=self.semantic_classes)
        logger.experiment.add_figure(f"Embeddings semantic/{stage}",
                                     embeddings_fig, global_step=epoch)
        self.embeddings.reset()
