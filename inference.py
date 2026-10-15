"""
inference.py — Checkpoint loading and SMILES-to-prediction pipeline.

Handles:
  - Loading best_gnn_model.pt with full compatibility validation
  - Converting a SMILES string into a PyG Data object via OGB's smiles2graph
  - Running the GNN in eval mode and returning a probability + label
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.data import Data

from model import GNN

# ── Constants ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path(__file__).parent / "best_gnn_model.pt"

# Thresholds chosen to match binary classification convention
ACTIVITY_THRESHOLD = 0.5  # sigmoid(logit) > 0.5 → "Active"

# Expected number of keys in a correct checkpoint
_EXPECTED_KEY_COUNT = 76


# ── Lazy singleton ────────────────────────────────────────────────────────────
_model: GNN | None = None
_load_error: str | None = None


def load_model(checkpoint_path: str | Path = CHECKPOINT_PATH) -> Tuple[GNN, str]:
    """
    Load the trained GNN from checkpoint.  Caches the model instance so
    subsequent calls are free.

    Returns
    -------
    model : GNN  (in eval mode, on CPU)
    info  : human-readable string describing what was loaded
    """
    global _model, _load_error

    if _model is not None:
        return _model, "Model already loaded (cached)."

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        _load_error = f"Checkpoint not found at: {checkpoint_path}"
        raise FileNotFoundError(_load_error)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if not isinstance(state_dict, dict):
        _load_error = "Checkpoint is not a state_dict (OrderedDict expected)."
        raise ValueError(_load_error)

    n_keys = len(state_dict)
    if n_keys != _EXPECTED_KEY_COUNT:
        _load_error = (
            f"Checkpoint has {n_keys} keys; expected {_EXPECTED_KEY_COUNT}. "
            "The checkpoint may belong to a different architecture."
        )
        raise ValueError(_load_error)

    model = GNN()
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _model = model
    size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
    info = (
        f"Loaded '{checkpoint_path.name}' ({size_mb:.1f} MB) — "
        f"5-layer GINEConv, hidden=128, {n_keys} parameter tensors."
    )
    return _model, info


def smiles_to_pyg(smiles: str) -> Data:
    """
    Convert a SMILES string to a PyTorch Geometric Data object using OGB's
    canonical smiles2graph pipeline — identical to what the training set used.

    Parameters
    ----------
    smiles : str
        A valid SMILES string.

    Returns
    -------
    torch_geometric.data.Data with:
        x          — LongTensor (N, 9) atom features
        edge_index — LongTensor (2, E)
        edge_attr  — LongTensor (E, 3) bond features
        batch      — LongTensor (N,) all-zeros (single graph)
    """
    from ogb.utils.mol import smiles2graph  # import here to keep module light

    graph = smiles2graph(smiles)

    if graph["num_nodes"] == 0:
        raise ValueError(f"SMILES produced an empty graph: '{smiles}'")

    data = Data(
        x=torch.tensor(graph["node_feat"], dtype=torch.long),
        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(graph["edge_feat"], dtype=torch.long),
        batch=torch.zeros(graph["num_nodes"], dtype=torch.long),
    )
    return data


def predict(smiles: str, checkpoint_path: str | Path = CHECKPOINT_PATH) -> dict:
    """
    Run HIV-activity prediction for a single SMILES string.

    Parameters
    ----------
    smiles : str
        A valid SMILES string.
    checkpoint_path : str | Path
        Path to the .pt checkpoint (defaults to best_gnn_model.pt).

    Returns
    -------
    dict with keys:
        probability  : float  — sigmoid output in [0, 1]
        label        : str    — "Active" or "Inactive"
        confidence   : str    — "High" / "Moderate" / "Low"
        logit        : float  — raw model output
        load_info    : str    — description of loaded checkpoint
    """
    model, load_info = load_model(checkpoint_path)
    data = smiles_to_pyg(smiles)

    with torch.no_grad():
        logit = model(data).item()

    probability = torch.sigmoid(torch.tensor(logit)).item()
    label = "Active" if probability >= ACTIVITY_THRESHOLD else "Inactive"

    # Confidence bands
    distance = abs(probability - 0.5)
    if distance >= 0.35:
        confidence = "High"
    elif distance >= 0.15:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return {
        "probability": probability,
        "label": label,
        "confidence": confidence,
        "logit": logit,
        "load_info": load_info,
    }
