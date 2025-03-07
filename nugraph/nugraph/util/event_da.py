"""NuGraph3 event decoder"""
from typing import Any
import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sn
from ....util import RecallLoss
from ..types import Data
from ....util.MMDLoss import MMDLoss
from ....util.SemanticLoss import SemanticAlignmentLoss
from ....util.Isomap import IsomapCombinedPlot

class EventDecoderDA(nn.Module):
    """
    NuGraph3 event decoder module for Domain Adaptation on event-level features

    Convolve interaction node embedding down to a set of categorical scores
    for each event class.

    Use MMD/Sinkhorn loss on interaction node embedding to align them across the source and target dataset and improve classification.

    Args:
        interaction_features: Number of interaction node features
        planes: List of detector planes
        event_classes: List of event classes
    """
    def __init__(self,
                 interaction_features: int,
                 #event_classes: list[str]):  #use this for correct NG3 data
                 event_classes: list['cc_nue', 'cc_numu', 'cc_nutau', 'nc']):
        super().__init__()
        

        # loss function
        self.loss = RecallLoss()
        ##### UPDATED #####
        self.loss_mmd = MMDLoss()
        mmd = torch.tensor(0.0)
        
        #self.loss_semantic = SemanticAlignmentLoss(metric='cosine') #or metric='euclidean'
        #semantic = torch.tensor(0.0)        
        ##### UPDATED #####
        
        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # metrics
        metric_args = {
            "task": "multiclass",
            "num_classes": 4 #len(event_classes)  #use this for proper NG3 data  
        }
        self.recall_s = tm.Recall(**metric_args)
        self.precision_s = tm.Precision(**metric_args)
        self.cm_recall_s = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision_s = tm.ConfusionMatrix(normalize="pred", **metric_args)

        self.recall_t = tm.Recall(**metric_args)
        self.precision_t = tm.Precision(**metric_args)
        self.cm_recall_t = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision_t = tm.ConfusionMatrix(normalize="pred", **metric_args)
        self.isomap = IsomapCombinedPlot()
        
        # network
        self.net = nn.Linear(in_features=interaction_features,
                             out_features=4) #len(event_classes))  #Use 4 for NG3 data
        self.classes = ['cc_nue', 'cc_numu', 'cc_nutau', 'nc']
        #self.classes = event_classes  #use this for proper NG3 data
        
 #### UPDATED #### decoder is getting both source and target data
    def forward(self, dataS: Data, dataT: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 event decoder forward pass

        Args:
            dataS: Graph data object (source)
            dataT: Graph data object (target)
            stage: Stage name (train/val/test)
        """
#### UPDATED ####
        # run network and calculate loss
        #print(dataT)
        xS = self.net(dataS["evt"].x)
        yS = dataS["evt"].y
        #print(xS.size(),yS.size())
        wS = 2 * (-1 * self.temp).exp()
        lossS = wS * self.loss(xS, yS) + self.temp

        xT = self.net(dataT["evt"].x)
        yT = dataT["evt"].y
        wT = 2 * (-1 * self.temp).exp()
        lossT = wT * self.loss(xT, yT) + self.temp
        

        # MMD Alignment
        mmd = self.loss_mmd(dataS["evt"].x, dataT["evt"].x)
        
        w_da = 0.0 #arbitrary mmd loss weight, update if needed or perform scheduling
        loss = lossS + w_da * mmd # +lossT  #it can be removed if we want or if labels are missing
        #loss = lossS  #for noDA
    
        # Semantic loss alignment option
        #semantic = self.loss_semantic(dataS["evt"].x, yS, dataT["evt"].x, yT) 
        #loss = lossS + lossT + w_da * semantic

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"loss_event_source/{stage}"] = lossS
            metrics[f"loss_event_target/{stage}"] = lossT
            #### UPDATED ####
            metrics[f"mmd/{stage}"] = mmd
            #metrics[f"semantic/{stage}"] = semantic
            metrics[f"loss_total/{stage}"] = loss
            #### UPDATED ####
            metrics[f"recall_event_source/{stage}"] = self.recall_s(xS, yS)
            metrics[f"precision_event_source/{stage}"] = self.precision_s(xS, yS)
            metrics[f"recall_event_target/{stage}"] = self.recall_t(xT, yT)
            metrics[f"precision_event_target/{stage}"] = self.precision_t(xT, yT)
        if stage == "train":
            metrics["temperature/event"] = self.temp
        if stage in ["val", "test"]:
            self.cm_recall_s.update(xS, yS)
            self.cm_precision_s.update(xS, yS)
            self.cm_recall_t.update(xT, yT)
            self.cm_precision_t.update(xT, yT)
            #### UPDATED ####
            self.isomap.update(xS, yS, xT, yT)
            #### UPDATED ####

        # add inference output to graph object
        dataS["evt"].e = xS.softmax(dim=1)
        if isinstance(dataS, Batch):
            dataS._slice_dict["evt"]["e"] = dataS["evt"].ptr
            incS = torch.zeros(dataS.num_graphs, device=dataS["evt"].x.device)
            dataS._inc_dict["evt"]["e"] = incS

        dataT["evt"].e = xT.softmax(dim=1)
        if isinstance(dataT, Batch):
            dataT._slice_dict["evt"]["e"] = dataT["evt"].ptr
            incT = torch.zeros(dataS.num_graphs, device=dataT["evt"].x.device)
            dataT._inc_dict["evt"]["e"] = incT

        return loss, metrics
#### UPDATED ####
    
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

        logger.experiment.add_figure(f"recall_event_matrix_source/{stage}",
                                     self.draw_confusion_matrix(self.cm_recall_s),
                                     global_step=epoch)
        self.cm_recall_s.reset()

        logger.experiment.add_figure(f"recall_event_matrix_target/{stage}",
                                     self.draw_confusion_matrix(self.cm_recall_t),
                                     global_step=epoch)
        self.cm_recall_t.reset()

        logger.experiment.add_figure(f"precision_event_matrix_source/{stage}",
                                self.draw_confusion_matrix(self.cm_precision_s),
                                global_step=epoch)
        self.cm_precision_s.reset()

        logger.experiment.add_figure(f"precision_event_matrix_target/{stage}",
                                self.draw_confusion_matrix(self.cm_precision_t),
                                global_step=epoch)
        self.cm_precision_t.reset()
        
        #### UPDATED ####
        dat1, lab1, dat2, lab2 = self.isomap.compute()
        isomap_fig = self.isomap.plot_isomap_combined_concatenated(
        dat1, lab1, dat2, lab2, epoch=epoch, class_names=self.classes)
        logger.experiment.add_figure(f"Isomap source and target/{stage}",
                                     isomap_fig, global_step=epoch)
        self.isomap.reset()
        #### UPDATED ####
