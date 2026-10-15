"""
model.py — GNN architecture definition.

This module reproduces the *exact* architecture used during training (train.py)
so that best_gnn_model.pt can be loaded without any key mismatches.

Do NOT modify this file unless you retrain the model with the same changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

# ── Hyper-parameters (must match train.py exactly) ──────────────────────────
HIDDEN = 128
LAYERS = 5

# OGB categorical feature vocabularies
_ATOM_DIMS = get_atom_feature_dims()   # [119, 5, 12, 12, 10, 6, 6, 2, 2]
_BOND_DIMS = get_bond_feature_dims()   # [5, 6, 2]


class AtomEncoder(nn.Module):
    """Sums per-feature embeddings to produce a single hidden-dim atom vector."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, hidden) for dim in _ATOM_DIMS]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor of shape (N, 9) — OGB atom feature matrix.
        Returns:
            FloatTensor of shape (N, hidden).
        """
        result = 0
        for i, embedding in enumerate(self.embeddings):
            result = result + embedding(x[:, i])
        return result


class BondEncoder(nn.Module):
    """Sums per-feature embeddings to produce a single hidden-dim bond vector."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, hidden) for dim in _BOND_DIMS]
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: LongTensor of shape (E, 3) — OGB bond feature matrix.
        Returns:
            FloatTensor of shape (E, hidden).
        """
        result = 0
        for i, embedding in enumerate(self.embeddings):
            result = result + embedding(edge_attr[:, i])
        return result


class GNN(nn.Module):
    """
    5-layer GINE (Graph Isomorphism Network with Edge features) classifier.

    Architecture:
        AtomEncoder  →  5 × (GINEConv → BatchNorm1d → ReLU → Dropout(0.3))
        → global_add_pool  →  Linear(128→64) → ReLU → Dropout(0.5) → Linear(64→1)

    Output: raw logit (apply sigmoid for HIV-activity probability).
    """

    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden)
        self.bond_encoder = BondEncoder(hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        graph_features = global_add_pool(x, data.batch)
        return self.classifier(graph_features)
