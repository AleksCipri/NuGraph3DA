"""NuGraph3 model architecture"""
import argparse
import warnings
import psutil

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import Batch

from pytorch_lightning import LightningModule

from .types import Data
from .encoder import Encoder
from .core import NuGraphCore
from .decoders import SemanticDecoder, FilterDecoder, EventDecoderDAmmd, EventDecoderDAdann, EventDecoderDASemantic, EventDecoderDASinkhorn, VertexDecoder, InstanceDecoder

from ...data import H5DataModuleDA

class NuGraph3DA(LightningModule):
    """
    NuGraph3 model architecture.

    Args:
        in_features: Number of input node features
        planar_features: Number of planar node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        instance_features: Number of instance features
        planes: Tuple of planes
        semantic_classes: Tuple of semantic classes
        event_classes: Tuple of event classes
        num_iters: Number of message-passing iterations
        event_head: Whether to enable event decoder
        semantic_head: Whether to enable semantic decoder
        filter_head: Whether to enable filter decoder
        vertex_head: Whether to enable vertex decoder
        use_checkpointing: Whether to use checkpointing
        lr: Learning rate
    """
    def __init__(self,
                 in_features: int = 4,
                 planar_features: int = 128,
                 nexus_features: int = 32,
                 interaction_features: int = 32,
                 instance_features: int = 32,
                 planes: tuple[str] = ('u','v','y'),
                 semantic_classes: tuple[str] = ('MIP','HIP','shower','michel','diffuse'),
                 event_classes: tuple[str] = ('numu','nue','nc'),
                 num_iters: int = 5,
                 event_head: bool = True,
                 semantic_head: bool = True,
                 filter_head: bool = True,
                 vertex_head: bool = False,
                 instance_head: bool = False,
                 use_checkpointing: bool = True,
                 lr: float = 0.001,
                 da_loss: str = 'mmd',
                 warmup_epochs: int = 0):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        self.planes = planes
        self.semantic_classes = semantic_classes
        self.event_classes = event_classes
        self.num_iters = num_iters
        self.lr = lr
        self.warmup_epochs = warmup_epochs

        self.encoder = Encoder(in_features, planar_features,
                               nexus_features, interaction_features,
                               planes)

        self.core_net = NuGraphCore(planar_features,
                                    nexus_features,
                                    interaction_features,
                                    planes,
                                    use_checkpointing)

        self.decoders = []

        if event_head:
            if da_loss=='mmd':
                self.event_decoder = EventDecoderDAmmd(
                    interaction_features,
                    event_classes, warmup_epochs=self.warmup_epochs)
            if da_loss=='dann':
                self.event_decoder = EventDecoderDAdann(
                    interaction_features,
                    event_classes, warmup_epochs=self.warmup_epochs)
            if da_loss=='sinkhorn':
                self.event_decoder = EventDecoderDASinkhorn(
                    interaction_features,
                    event_classes, warmup_epochs=self.warmup_epochs)
            if da_loss=='semantic':
                self.event_decoder = EventDecoderDASemantic(
                    interaction_features,
                    event_classes, warmup_epochs=self.warmup_epochs)
            self.decoders.append(self.event_decoder)
            
        
        if semantic_head:
            self.semantic_decoder = SemanticDecoder(
                planar_features,
                planes,
                semantic_classes)
            self.decoders.append(self.semantic_decoder)

        if filter_head:
            self.filter_decoder = FilterDecoder(
                planar_features,
                planes,
            )
            self.decoders.append(self.filter_decoder)

        if vertex_head:
            self.vertex_decoder = VertexDecoder(interaction_features)
            self.decoders.append(self.vertex_decoder)

        if instance_head:
            self.instance_decoder = InstanceDecoder(
                planar_features,
                instance_features,
                planes,
            )
            self.decoders.append(self.instance_decoder)

        if not self.decoders:
            raise RuntimeError('At least one decoder head must be enabled!')

        # metrics
        self.max_mem_cpu = 0.
        self.max_mem_gpu = 0.

    def forward(self, data: Data,
                stage: str = None):
        """
        NuGraph3 forward function

        This function runs the forward pass of the NuGraph3 architecture,
        and then loops over each decoder to compute the loss and calculate
        and log any performance metrics.

        Args:
            data: Graph data object
            stage: String tag defining the step type
        """
        # Check if the input is a list of two batches
        if isinstance(data, list) and len(data) == 2:
            batchA, batchB = data
        else:
            raise ValueError("Expected input data to be a list of two batches.")

        
        self.encoder(batchA)
        self.encoder(batchB)
        
        for _ in range(self.num_iters):
            self.core_net(batchA)
            self.core_net(batchB)
        total_loss = 0.
        total_metrics = {}
        for decoder in self.decoders:
            if decoder == self.event_decoder: 
                # Pass both batches to the event_decoder
                loss, metrics = decoder(batchA, batchB, stage)
            
            #if hasattr(self, "event_decoder"):
            #    loss, metrics = decoder(batchA, batchB, stage)
            
            else: #ignore the second dataset and don't perform DA for this decoder
                loss, metrics = decoder(batchA, stage)
            total_loss += loss
            total_metrics.update(metrics)

        #if hasattr(self, instance_decoder) and self.global_step > 1000:
        if hasattr(self, "instance_decoder") and self.global_step > 1000:
            if isinstance(batchA, Batch):
                batchA = Batch([self.instance_decoder.materialize(b) for b in batchA.to_data_list()])
            else:
                self.instance_decoder.materialize(batchA)

        return total_loss, total_metrics

    def on_train_start(self):
        hpmetrics = { 'max_lr': self.hparams.lr }
        self.logger.log_hyperparams(self.hparams, metrics=hpmetrics)
        self.max_mem_cpu = 0.
        self.max_mem_gpu = 0.

        scalars = {
            'loss': {'loss': [ 'Multiline', [ 'loss/train', 'loss/val' ]]},
            'acc': {}
        }
        for c in self.semantic_classes:
            scalars['acc'][c] = [ 'Multiline', [
                f'semantic_accuracy_class_train/{c}',
                f'semantic_accuracy_class_val/{c}'
            ]]
        self.logger.experiment.add_custom_scalars(scalars)

        """Enable domain adaptation loss after the warmup phase."""
        if self.current_epoch >= self.warmup_epochs:
            self.event_decoder.use_domain_adaptation = True

    def training_step(self,
                      batch: [Data, Data],
                      batch_idx: int) -> float:
        loss, metrics = self(batch, 'train')
        
        self.log('loss/train', loss, batch_size=batch[0].num_graphs, prog_bar=True)
        self.log_dict(metrics, batch_size=batch[0].num_graphs)
        self.log_memory(batch, 'train')
        return loss

    def validation_step(self,
                        batch: [Data, Data],
                        batch_idx: int) -> None:
        loss, metrics = self(batch, 'val')
        self.log('loss/val', loss, batch_size=batch[0].num_graphs)
        self.log_dict(metrics, batch_size=batch[0].num_graphs)

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)

    def test_step(self,
                  batch: [Data, Data],
                  batch_idx: int) -> None: #int = 0
        loss, metrics = self(batch, 'test', True)
        self.log('loss/test', loss, batch_size=batch[0].num_graphs)
        self.log_dict(metrics, batch_size=batch[0].num_graphs)
        self.log_memory(batch, 'test')

    def on_test_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'test', epoch)

    def predict_step(self,
                     batch: [Data, Data],
                     batch_idx: int) -> [Data, Data]:  #int = 0
        self(batch) 
        return batch

    def configure_optimizers(self) -> tuple:
        optimizer = AdamW(self.parameters(),
                          lr=self.lr)
        onecycle = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    def on_after_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Clamps eta_s, eta_t, eta_da after each optimizer step."""
        self.event_decoder.eta_s.data.clamp_(min=1e-3)
        self.event_decoder.eta_t.data.clamp_(min=1e-3)
        self.event_decoder.eta_da.data.clamp_(min=1e-3)
   

    def log_memory(self, batch: Data, stage: str) -> None:
        """
        Log CPU and GPU memory usage

        Args:
            batch: Data object to step over
            stage: String tag defining the step type
        """
        # log CPU memory
        if not hasattr(self, 'max_mem_cpu'):
            self.max_mem_cpu = 0.
        cpu_mem = psutil.Process().memory_info().rss / float(1073741824)
        self.max_mem_cpu = max(self.max_mem_cpu, cpu_mem)
        self.log(f'memory_cpu/{stage}', self.max_mem_cpu,
                 batch_size=batch[0].num_graphs, reduce_fx=torch.max)

        # log GPU memory
        if not hasattr(self, 'max_mem_gpu'):
            self.max_mem_gpu = 0.
        if self.device != torch.device('cpu'):
            gpu_mem = torch.cuda.memory_reserved(self.device)
            gpu_mem = float(gpu_mem) / float(1073741824)
            self.max_mem_gpu = max(self.max_mem_gpu, gpu_mem)
            self.log(f'memory_gpu/{stage}', self.max_mem_gpu,
                     batch_size=batch[0].num_graphs, reduce_fx=torch.max)

    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add argparse argument group for NuGraph3 model

        Args:
            parser: Argument parser to append argument group to
        """
        model = parser.add_argument_group('model', 'NuGraph3 model configuration')
        model.add_argument('--num-iters', type=int, default=5,
                           help='Number of message-passing iterations')
        model.add_argument('--in-feats', type=int, default=4,
                           help='Number of input node features')
        model.add_argument('--planar-feats', type=int, default=128,
                           help='Hidden dimensionality of planar convolutions')
        model.add_argument('--nexus-feats', type=int, default=32,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--interaction-feats', type=int, default=32,
                           help='Hidden dimensionality of interaction layer')
        model.add_argument('--instance-feats', type=int, default=32,
                           help='Hidden dimensionality of object condensation')
        model.add_argument('--event', action='store_true',
                           help='Enable event classification head')
        model.add_argument('--semantic', action='store_true',
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true',
                           help='Enable background filter head')
        model.add_argument('--instance', action='store_true',
                           help='Enable instance segmentation head')
        model.add_argument('--vertex', action='store_true',
                           help='Enable vertex regression head')
        model.add_argument('--no-checkpointing', action='store_false',
                           dest="use_checkpointing",
                           help='Disable checkpointing during training')
        model.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        model.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        model.add_argument('--da-loss', type=str, choices=['dann', 'mmd', 'sinkhorn', 'semantic'], default='MMD',
                           help='Which DA loss to use (options are: DANN, MMD, Sinkhorn)')
        model.add_argument('--warmup', type=int, default=0,
                           help='Include the warmup phase before domain adaptation loss turns on. Default is 0 i.e. no warmup.')
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, nudata: H5DataModuleDA) -> 'NuGraph3':
        """
        Construct model from arguments

        Args:
            args: Namespace containing parsed arguments
            nudata: Data module
        """
        return cls(
            in_features=args.in_feats,
            planar_features=args.planar_feats,
            nexus_features=args.nexus_feats,
            interaction_features=args.interaction_feats,
            instance_features=args.instance_feats,
            planes=nudata.planes,
            semantic_classes=nudata.semantic_classes,
            event_classes=nudata.event_classes,
            num_iters=args.num_iters,
            event_head=args.event,
            semantic_head=args.semantic,
            filter_head=args.filter,
            vertex_head=args.vertex,
            instance_head=args.instance,
            use_checkpointing=args.use_checkpointing,
            lr=args.learning_rate,
            da_loss=args.da_loss,
            warmup_epochs=args.warmup)
