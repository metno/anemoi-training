import logging
import math
import os
from collections import defaultdict
from collections.abc import Generator
from collections.abc import Mapping
from typing import Optional
from typing import Union

from anemoi.training.train.forecaster import GraphForecaster
import pytorch_lightning as pl
import torch
from anemoi.models.interface import FuserModelInterface
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint
from anemoi.training.losses.zip import ZipLoss

from anemoi.training.losses.weightedloss import BaseWeightedLoss
from anemoi.training.losses.utils import grad_scaler
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.utils.masks import Boolean1DMask
from anemoi.training.utils.masks import NoOutputMask

LOGGER = logging.getLogger(__name__)

class NetatmoGraphForecaster(pl.LightningModule):

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: list,
        metadata: dict,
    ) -> None:
        super().__init__()

        graph_data = graph_data.to(self.device)

        self.model = FuserModelInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=graph_data,
            config=DotDict(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )
        self.data_indices = data_indices

        self.save_hyperparameters()

        self.latlons_data = [graph_data[mesh].x for mesh in config.graph.input_nodes.values()]
        self.node_weights = self.get_node_weights(config, graph_data) 

        #TODO
        if config.model.get("output_mask", None) is not None:
            raise NotImplementedError("output mask not supported in NetatmoGraphForecaster")
            self.output_mask = Boolean1DMask(graph_data[config.graph.data][config.model.output_mask])
        else:
            self.output_mask = NoOutputMask()
        self.node_weights = self.output_mask.apply(self.node_weights, dim=0, fill_value=0.0)

        self.dset_weights = config.training.dataset_loss_scaling

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        variable_scaling = self.get_variable_scaling(config, data_indices)

        self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices) #TODO

        loss_kwargs = [{"node_weights": node_weights} for node_weights in self.node_weights]

        scalars = [{"variable": (-1, scaling)} for scaling in variable_scaling]

        zip_loss = [
            GraphForecaster.get_loss_function(
                loss_config, 
                scalars=scalars[dset],
                **loss_kwargs[dset], 
            )
            for dset, loss_config in enumerate(config.training.training_loss)
            ]
        self.loss = ZipLoss(zip_loss)

        zip_metrics = [
            GraphForecaster.get_loss_function(
                metrics_config, 
                scalars=scalars[dset],
                **loss_kwargs[dset], 
            )
            for dset, metrics_config in enumerate(config.training.validation_metrics)
            ]

        self.metrics = ZipLoss(zip_metrics)

        if config.training.loss_gradient_scaling:
            raise NotImplementedError("Loss gradient scaling not available for NetatmoGraphForecaster")
#            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max                

        self.use_zero_optimizer = config.training.zero_optimizer

        self.model_comm_group = None

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // config.hardware.num_gpus_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % config.hardware.num_gpus_per_model
        self.model_comm_num_groups = math.ceil(
            config.hardware.num_gpus_per_node * config.hardware.num_nodes / config.hardware.num_gpus_per_model,
        )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.model(x, self.model_comm_group)

    @staticmethod
    def get_val_metric_ranges(
        config: DictConfig, 
        data_indices: list
    ) -> list[dict, dict]:
        return [GraphForecaster.get_val_metric_ranges(config, data_index)[1] for data_index in data_indices]

    @staticmethod
    def get_variable_scaling(
        config: DictConfig,
        data_indices: list
    ) -> list:
        return [GraphForecaster.get_variable_scaling(config, data_index) for data_index in data_indices]

    @staticmethod
    def get_node_weights(
        config: DictConfig,
        graph_data: HeteroData
    ) -> list:
        node_weighters = [instantiate(node_loss_weights) for node_loss_weights in config.training.node_loss_weights]

        return [node_weighting.weights(graph_data) for node_weighting in node_weighters]

    def set_model_comm_group(self, model_comm_group: ProcessGroup) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group

    def advance_input(
        self,
        x: list,
        y_pred: list,
        batch: list,
        rollout_step: int,
    ) -> list:
        for dset_idx, (x_elem, y_pred_elem, batch_elem) in enumerate(zip(x, y_pred, batch)):
            x_elem = x_elem.roll(-1, dims=1)

            x_elem[:, -1, :, :, self.data_indices[dset_idx].internal_model.input.prognostic] = y_pred_elem[
                ...,
                self.data_indices[dset_idx].internal_model.output.prognostic,
            ]

            x_elem[:, -1] = self.output_mask.rollout_boundary(x_elem[:, -1], batch_elem[:, -1], self.data_indices[dset_idx])

            x_elem[:, -1, :, :, self.data_indices[dset_idx].internal_model.input.forcing] = batch_elem[
                :,
                self.multi_step +  rollout_step,
                :,
                :,
                self.data_indices[dset_idx].internal_data.input.forcing,
            ]
        return x

    def rollout_step(
        self,
        batch: list,
        rollout: Optional[int] = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator:
        num_dsets = len(batch)
        batch = self.model.pre_processors(batch, in_place=not validation_mode)
        x = [None]*num_dsets
        for batch_idx, batch_elem in enumerate(batch):
            x[batch_idx] = batch_elem[
                :,
                0 : self.multi_step,
                ...,
                self.data_indices[batch_idx].internal_data.input.full,
            ]
            msg = (
                "Batch length not sufficient for requested multi_step length!"
                f", {batch[batch_idx].shape[1]} !>= {rollout + self.multi_step}"
            )
            assert batch[batch_idx].shape[1] >= rollout + self.multi_step, msg

        for rollout_step in range(rollout or self.rollout):
            y_pred = self(x)

            y = [None]*num_dsets
            for batch_idx, batch_elem in enumerate(batch):
                y[batch_idx] = batch_elem[
                    :,
                    self.multi_step + rollout_step,
                    ...,
                    self.data_indices[batch_idx].internal_data.output.full,
                ]
            loss = checkpoint(self.loss, y_pred, y, use_reentrant=False) if training_mode else None

            x = self.advance_input(x, y_pred, batch, rollout_step)

            metrics_next = [{} for _ in range(len(batch))]
            if validation_mode:
                metrics_next = self.calculate_val_metrics( #TODO: implement this to return tuple
                    y_pred,
                    y,
                    rollout_step,
                )
            yield loss, metrics_next, y_pred

    def _step(
        self, 
        batch: list,
        batch_idx: int,
        validation_mode: bool = False  
    ) -> list:
        del batch_idx
        num_dsets = len(batch)
        loss = [torch.zeros(1, dtype=batch[i].dtype, device=self.device, requires_grad=False)
                                 for i in range(num_dsets)]
        metrics = [{} for _ in range(num_dsets)]
        y_preds = [[] for _ in range(num_dsets)]

        for loss_next, metrics_next, y_preds_next in self.rollout_step(
            batch,
            rollout = self.rollout,
            training_mode=True,
            validation_mode=validation_mode
        ):
            for dset in range(num_dsets):
                loss[dset] += loss_next[dset]
                metrics[dset].update(metrics_next[dset])
                y_preds[dset].extend(y_preds_next[dset])
        
        for dset in range(num_dsets):
            loss[dset] *= 1.0 / self.rollout

        return loss, metrics, y_preds
    
    def calculate_val_metrics(
        self,
        y_pred: list,
        y: list,
        rollout_step: int
    ) -> list:
        num_dsets = len(y_pred)
        metrics = [{} for _ in range(num_dsets)]
        y_postprocessed = self.model.post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False)

        for dset, metric in enumerate(self.metrics.losses):
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())

            if not isinstance(metric, BaseWeightedLoss):
                metrics[dset][f"{metric_name}/{rollout_step + 1}"] = metric(
                y_pred_postprocessed[dset],
                y_postprocessed[dset],
                )
                continue
            for mkey, indices in self.val_metric_ranges[dset].items():
                metrics[dset][f"{metric_name}/{mkey}/{rollout_step + 1}"] = metric(
                    y_pred_postprocessed[dset][..., indices],
                    y_postprocessed[dset][..., indices],
                    scalar_indices=[..., indices],
                )

        
        return metrics
    
    def training_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        train_loss, _, _, = self._step(batch, batch_idx)
        for i in range(len(train_loss)):
            train_loss[i] = train_loss[i]*self.dset_weights[i]
        combined_loss = sum(train_loss)
        self.log(
            f"train_wmse",
            combined_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        for dset, loss in enumerate(train_loss):
            self.log(
                f"train_{getattr(self.loss.losses[dset], 'name', self.loss.losses[dset].__class__.__name__.lower())}_{dset}",
                loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=self.logger_enabled,
                batch_size=batch[0].shape[0],
                sync_dist=True
            )
        return combined_loss      
    
    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: list, batch_idx: int) -> None:
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)
        #Scale the loss contributions
        for i in range(len(val_loss)):
            val_loss[i] = val_loss[i]*self.dset_weights[i]
        combined_loss = sum(val_loss)
        self.log(
            f"val_wmse",
            combined_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
            sync_dist=True,
        )
        for dset, loss in enumerate(val_loss):
            self.log(
                f"val_{getattr(self.loss.losses[dset], 'name', self.loss.losses[dset].__class__.__name__.lower())}_dset{dset}",
                loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=self.logger_enabled,
                batch_size=batch[0].shape[0],
                sync_dist=True
            )
            for mname, mvalue in metrics[dset].items():
                self.log(
                    f"val_{mname}_dset{dset}",
                    mvalue,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=batch[0].shape[0],
                    sync_dist=True,
                )
        return combined_loss, y_preds
    
    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        if self.use_zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(),
                optimizer_class=torch.optim.AdamW,
                betas=(0.9, 0.95),
                lr=self.lr,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.trainer.model.parameters(),
                betas=(0.9, 0.95),
                lr=self.lr,
            )  # , fused=True)

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]   


    