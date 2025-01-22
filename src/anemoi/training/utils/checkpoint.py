# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from anemoi.utils.checkpoints import save_metadata

from anemoi.training.train.forecaster import GraphForecaster
from pytorch_lightning.utilities.rank_zero import rank_zero_only


LOGGER = logging.getLogger(__name__)


def load_and_prepare_model(lightning_checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    """Load the lightning checkpoint and extract the pytorch model and its metadata.

    Parameters
    ----------
    lightning_checkpoint_path : str
        path to lightning checkpoint

    Returns
    -------
    tuple[torch.nn.Module, dict]
        pytorch model, metadata

    """
    module = GraphForecaster.load_from_checkpoint(lightning_checkpoint_path)
    model = module.model

    metadata = dict(**model.metadata)
    model.metadata = None
    model.config = None

    return model, metadata


def save_inference_checkpoint(model: torch.nn.Module, metadata: dict, save_path: Path | str) -> Path:
    """Save a pytorch checkpoint for inference with the model metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model
    metadata : dict
        Anemoi Metadata to inject into checkpoint
    save_path : Path | str
        Directory to save anemoi checkpoint

    Returns
    -------
    Path
        Path to saved checkpoint
    """
    save_path = Path(save_path)
    inference_filepath = save_path.parent / f"inference-{save_path.name}"

    torch.save(model, inference_filepath)
    save_metadata(inference_filepath, metadata)
    return inference_filepath

def transfer_learning_loading(model: torch.nn.Module, ckpt_path: Path | str) -> nn.Module:
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location=model.device)
    # Filter out layers with size mismatch
    state_dict = checkpoint["state_dict"]
    model_state_dict = model.state_dict()
    for key in state_dict.copy():
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            LOGGER.info("Skipping loading parameter: %s", key)
            LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
            LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))
            del state_dict[key]  # Remove the mismatched key
    # Load the filtered st-ate_dict into the model
    model.load_state_dict(state_dict, strict=False)
    return model

def transfer_learning_loading_with_layer_dict(
        model: torch.nn.Module, 
        ckpt_path: Path | str, 
        layer_dict: dict
        ) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location = model.device)
    state_dict = checkpoint["state_dict"]

    for mname, mparameter in model.named_parameters():
        cname = mname
        for old_name, new_name in layer_dict.items():
            if new_name in cname:
                cname = mname.replace(new_name, old_name)
        

        if rank_zero_only.rank == 0:
            print(f"cname: {cname}")
            print(f"mname: {mname}")
        if cname in state_dict:
            #LOGGER.info("Replacing layer %s in model with layer %s from old checkpoint", mname, cname)
            if rank_zero_only.rank == 0:
                print(f"Replacing layer {mname} in model with layer {cname} from old checkpoint")
            mparameter.data = state_dict[cname].data

    return model

def freeze_weights(model: torch.nn.Module, layers: list[str]) -> torch.nn.Module:
    for mname, mparameter in model.named_parameters():
        for layer_name in layers:
            if layer_name in mname:
                if rank_zero_only.rank == 0:
                    print(f"freezing layer {mname}")
                mparameter.requires_grad = False

    return model