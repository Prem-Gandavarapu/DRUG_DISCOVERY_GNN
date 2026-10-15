"""
molecule_utils.py — RDKit utilities for SMILES validation, 2D drawing,
molecular property computation, and Plotly graph construction.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Draw,
    rdMolDescriptors,
)
from rdkit.Chem.Draw import rdMolDraw2D

# ── OGB graph conversion ──────────────────────────────────────────────────────
from ogb.utils.mol import smiles2graph

# ── Visualization ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import networkx as nx


# ────────────────────────────────────────────────────────────────────────────
# Bond type labels (OGB bond-feature index 0 encodes bond type)
# OGB bond type vocab: SINGLE=0, DOUBLE=1, TRIPLE=2, AROMATIC=3, others=4
# ────────────────────────────────────────────────────────────────────────────
_BOND_TYPE_LABELS = {0: "SINGLE", 1: "DOUBLE", 2: "TRIPLE", 3: "AROMATIC", 4: "OTHER"}

# Atom colors for visualization — bright CPK palette that reads on dark backgrounds
_ATOM_COLORS: dict[str, str] = {
    "C": "#94A3B8",   # slate — visible on dark canvas
    "N": "#60A5FA",   # blue
    "O": "#F87171",   # red
    "S": "#FBBF24",   # amber
    "F": "#34D399",   # emerald
    "Cl": "#4ADE80",  # green
    "Br": "#FB923C",  # orange
    "I": "#C084FC",   # purple
    "P": "#F97316",   # orange-red
    "H": "#CBD5E1",   # light slate
}
_DEFAULT_ATOM_COLOR = "#64748B"

# Bond width per type
_BOND_WIDTHS = {0: 1.8, 1: 3.2, 2: 4.8, 3: 2.6, 4: 1.2}


def validate_smiles(smiles: str) -> tuple[bool, str]:
    """
    Validate a SMILES string with RDKit.

    Returns
    -------
    (is_valid, message)
    """
    if not smiles or not smiles.strip():
        return False, "Empty input — please enter a SMILES string."

    smiles = smiles.strip()

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        return False, f"RDKit parse error: {exc}"

    if mol is None:
        return False, "Invalid SMILES: RDKit could not parse this structure."

    if mol.GetNumAtoms() == 0:
        return False, "SMILES produced a molecule with no atoms."

    return True, f"Valid SMILES — {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds."


def mol_to_image_b64(smiles: str, width: int = 450, height: int = 300) -> str:
    """
    Render a 2D structure image from SMILES and return a base64-encoded PNG.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot render invalid SMILES: {smiles}")

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Return SVG directly as a data URI
    svg_b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{svg_b64}"


def compute_properties(smiles: str) -> dict[str, Any]:
    """
    Compute a panel of drug-likeness and structural properties via RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot compute properties for invalid SMILES: {smiles}")

    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    formula = rdMolDescriptors.CalcMolFormula(mol)

    # Lipinski Rule-of-Five
    ro5 = {
        "MW ≤ 500": mw <= 500,
        "LogP ≤ 5": logp <= 5,
        "HBD ≤ 5": hbd <= 5,
        "HBA ≤ 10": hba <= 10,
    }
    ro5_pass = sum(ro5.values())

    return {
        "Molecular Formula": formula,
        "Exact MW (Da)": round(mw, 3),
        "LogP": round(logp, 3),
        "H-Bond Donors": hbd,
        "H-Bond Acceptors": hba,
        "TPSA (Å²)": round(tpsa, 2),
        "Rotatable Bonds": rot_bonds,
        "Rings": rings,
        "Aromatic Rings": arom_rings,
        "Heavy Atoms": heavy_atoms,
        "Lipinski Ro5": f"{ro5_pass}/4 rules satisfied",
        "_ro5_detail": ro5,
    }


def build_graph_figure(smiles: str, dark: bool = True) -> tuple[go.Figure, dict]:
    """
    Build a Plotly figure of the molecular graph and return accompanying stats.

    Nodes = atoms (coloured by element, sized by degree).
    Edges = bonds (width and colour by type).
    Atom indices displayed as hover text.

    Parameters
    ----------
    smiles : str
        A valid SMILES string.
    dark : bool
        When True (default) render on a dark canvas matching the dashboard theme.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot build graph for invalid SMILES: {smiles}")

    # ── Build NetworkX graph ──────────────────────────────────────────────────
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            symbol=atom.GetSymbol(),
            atomic_num=atom.GetAtomicNum(),
            degree=atom.GetDegree(),
            formal_charge=atom.GetFormalCharge(),
            is_aromatic=atom.GetIsAromatic(),
            num_hs=atom.GetTotalNumHs(),
        )

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = bond.GetBondTypeAsDouble()
        btype_str = str(bond.GetBondType()).split(".")[-1]
        G.add_edge(u, v, bond_type=btype, bond_type_str=btype_str)

    # ── RDKit 2D layout ───────────────────────────────────────────────────────
    from rdkit.Chem import rdDepictor
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = {
        atom.GetIdx(): (
            conf.GetAtomPosition(atom.GetIdx()).x,
            conf.GetAtomPosition(atom.GetIdx()).y,
        )
        for atom in mol.GetAtoms()
    }

    # ── OGB features ─────────────────────────────────────────────────────────
    ogb_graph = smiles2graph(smiles)
    node_feats = ogb_graph["node_feat"]  # (N, 9)
    edge_feats = ogb_graph["edge_feat"]  # (E, 3)

    # ── Theme colours ─────────────────────────────────────────────────────────
    bg_color   = "#0D1B2A" if dark else "#FFFFFF"
    paper_bg   = "#111C2F" if dark else "#FFFFFF"
    grid_color = "#1E2D42" if dark else "#E5E7EB"
    text_color = "#CBD5E1" if dark else "#1F2937"
    idx_color  = "#64748B" if dark else "#6B7280"

    bond_colors = {
        "SINGLE":   "#475569" if dark else "#9CA3AF",
        "DOUBLE":   "#38BDF8" if dark else "#2563EB",
        "TRIPLE":   "#F472B6" if dark else "#BE185D",
        "AROMATIC": "#2DD4BF" if dark else "#0D9488",
        "OTHER":    "#334155" if dark else "#D1D5DB",
    }

    # ── Edge traces ───────────────────────────────────────────────────────────
    bond_traces = []
    edge_groups: dict[str, list] = {k: [] for k in bond_colors}

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype_str = str(bond.GetBondType()).split(".")[-1]
        if "DOUBLE" in btype_str:
            group = "DOUBLE"
        elif "TRIPLE" in btype_str:
            group = "TRIPLE"
        elif "AROMATIC" in btype_str:
            group = "AROMATIC"
        elif "SINGLE" in btype_str:
            group = "SINGLE"
        else:
            group = "OTHER"
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_groups[group].append((x0, y0, x1, y1))

    bond_width_map = {"SINGLE": 1.8, "DOUBLE": 3.2, "TRIPLE": 4.8, "AROMATIC": 2.6, "OTHER": 1.2}
    for group, edges in edge_groups.items():
        if not edges:
            continue
        ex, ey = [], []
        for x0, y0, x1, y1 in edges:
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        bond_traces.append(
            go.Scatter(
                x=ex, y=ey,
                mode="lines",
                line=dict(color=bond_colors[group], width=bond_width_map[group]),
                name=f"{group}",
                hoverinfo="none",
                showlegend=True,
            )
        )

    # ── Node trace ────────────────────────────────────────────────────────────
    node_x, node_y, node_colors, node_sizes = [], [], [], []
    hover_texts, node_labels = [], []

    for idx in G.nodes():
        x, y = pos[idx]
        node_x.append(x)
        node_y.append(y)

        sym = G.nodes[idx]["symbol"]
        node_colors.append(_ATOM_COLORS.get(sym, _DEFAULT_ATOM_COLOR))
        degree = G.degree(idx)
        node_sizes.append(max(26, 16 + degree * 5))
        node_labels.append(sym)

        feat_row = node_feats[idx] if idx < len(node_feats) else []
        hover_texts.append(
            f"<b>Atom {idx} — {sym}</b><br>"
            f"Atomic number: {G.nodes[idx]['atomic_num']}<br>"
            f"Degree: {degree}<br>"
            f"Formal charge: {G.nodes[idx]['formal_charge']}<br>"
            f"Aromatic: {G.nodes[idx]['is_aromatic']}<br>"
            f"Total H: {G.nodes[idx]['num_hs']}<br>"
            f"<i>OGB features: {list(feat_row)}</i>"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="middle center",
        textfont=dict(size=10, color="white", family="Inter, Arial, sans-serif"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="#1E293B" if dark else "white"),
            opacity=0.95,
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        name="Atoms",
        showlegend=True,
    )

    # ── Atom index annotations ────────────────────────────────────────────────
    annotations = []
    for idx in G.nodes():
        x, y = pos[idx]
        annotations.append(dict(
            x=x, y=y - 0.30,
            text=f"<span style='font-size:7px;color:{idx_color}'>{idx}</span>",
            showarrow=False,
            xref="x", yref="y",
        ))

    # ── Figure ────────────────────────────────────────────────────────────────
    smiles_label = smiles[:60] + ("…" if len(smiles) > 60 else "")
    fig = go.Figure(
        data=bond_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text=smiles_label,
                font=dict(size=12, color=text_color, family="Inter, monospace"),
                x=0.0,
                xref="paper",
            ),
            showlegend=True,
            hovermode="closest",
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                showline=False,
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                showline=False, scaleanchor="x",
            ),
            plot_bgcolor=bg_color,
            paper_bgcolor=paper_bg,
            margin=dict(l=16, r=16, t=40, b=16),
            height=480,
            annotations=annotations,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.01,
                xanchor="right", x=1,
                font=dict(size=11, color=text_color),
                bgcolor="rgba(0,0,0,0)",
            ),
            hoverlabel=dict(
                bgcolor="#1E293B" if dark else "#F8FAFC",
                bordercolor="#2DD4BF" if dark else "#0D9488",
                font=dict(color="#F1F5F9" if dark else "#1F2937", size=12),
            ),
        ),
    )

    # ── Graph statistics ──────────────────────────────────────────────────────
    bond_type_counts: dict[str, int] = {}
    for bond in mol.GetBonds():
        bt = str(bond.GetBondType()).split(".")[-1]
        bond_type_counts[bt] = bond_type_counts.get(bt, 0) + 1

    element_counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        element_counts[sym] = element_counts.get(sym, 0) + 1

    stats = {
        "num_nodes": mol.GetNumAtoms(),
        "num_edges": mol.GetNumBonds(),
        "num_directed_edges": len(ogb_graph["edge_index"][0]),
        "node_feature_dim": node_feats.shape[1] if len(node_feats) > 0 else 0,
        "edge_feature_dim": edge_feats.shape[1] if len(edge_feats) > 0 else 0,
        "element_counts": element_counts,
        "bond_type_counts": bond_type_counts,
        "avg_degree": round(
            sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1), 2
        ),
        "is_connected": nx.is_connected(G),
    }

    return fig, stats
