"""
app.py — AI Drug Discovery Platform
Dark-navy biotech dashboard for a final-year GNN research project.

Key behaviours
──────────────
• Results are gated behind the "Analyse Molecule" button via session_state.
  Changing the text area or selecting a new example never shows stale results.
• Switching navigation pages and returning to Molecule Analysis shows the
  last confirmed result, not a re-triggered analysis of whatever is in the box.
• Invalid SMILES shows a clear inline error; valid SMILES shows full analysis.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import traceback as _traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference import CHECKPOINT_PATH, predict
from molecule_utils import (
    build_graph_figure,
    compute_properties,
    mol_to_image_b64,
    validate_smiles,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Drug Discovery Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens — single source of truth for all colours
# ─────────────────────────────────────────────────────────────────────────────
C: dict[str, str] = {
    "bg":          "#0B1120",
    "panel":       "#111C2F",
    "panel2":      "#0F1A2B",
    "border":      "#1E2D42",
    "border2":     "#26364D",
    "teal":        "#2DD4BF",
    "teal_dim":    "#1A9485",
    "blue":        "#38BDF8",
    "text":        "#F1F5F9",
    "text2":       "#94A3B8",
    "text3":       "#64748B",
    "active":      "#10B981",
    "active_bg":   "#052E16",
    "active_bd":   "#065F46",
    "inactive":    "#F87171",
    "inactive_bg": "#2D0A0A",
    "inactive_bd": "#7F1D1D",
    "warn_bg":     "#1C1A05",
    "warn_bd":     "#713F12",
    "warn_text":   "#FDE68A",
}

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    color: {C['text']};
}}
.stApp {{
    background-color: {C['bg']};
}}
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {C['panel']} !important;
    border-right: 1px solid {C['border']} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {C['text']} !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {{
    color: {C['text2']} !important;
}}
section[data-testid="stSidebar"] textarea {{
    background-color: {C['panel2']} !important;
    border: 1px solid {C['border2']} !important;
    color: {C['text']} !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background-color: {C['panel2']} !important;
    border: 1px solid {C['border2']} !important;
    color: {C['text']} !important;
    border-radius: 8px !important;
}}

/* ── Primary button ── */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {C['teal']}, {C['teal_dim']}) !important;
    color: #0B1120 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1rem !important;
    transition: opacity 0.15s ease !important;
}}
.stButton > button[kind="primary"]:hover {{ opacity: 0.88 !important; }}

/* ── Expanders ── */
.streamlit-expanderHeader {{
    background-color: {C['panel2']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    color: {C['text2']} !important;
    font-size: 0.85rem !important;
}}
.streamlit-expanderContent {{
    background-color: {C['panel2']} !important;
    border: 1px solid {C['border']} !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}}

/* ── DataFrames ── */
.stDataFrame {{
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid {C['border']} !important;
}}

/* ── Alert boxes ── */
.stSuccess {{
    background-color: rgba(16,185,129,0.1) !important;
    border-left: 3px solid {C['active']} !important;
    color: #6EE7B7 !important;
    border-radius: 6px !important;
}}
.stError {{
    background-color: rgba(248,113,113,0.1) !important;
    border-left: 3px solid {C['inactive']} !important;
    color: #FCA5A5 !important;
    border-radius: 6px !important;
}}
hr {{ border-color: {C['border']} !important; margin: 1.2rem 0 !important; }}
.stCaption, small {{ color: {C['text3']} !important; font-size: 0.78rem !important; }}

/* ── Sidebar logo ── */
.sb-logo {{
    display: flex; align-items: center; gap: 10px;
    padding: 0.2rem 0 1rem;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 1rem;
}}
.sb-logo-hex {{
    width: 36px; height: 36px;
    background: linear-gradient(135deg, {C['teal']}, {C['blue']});
    clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
    flex-shrink: 0;
}}
.sb-logo-text {{ line-height: 1.2; }}
.sb-logo-text .t1 {{
    font-size: 0.95rem; font-weight: 700; color: {C['text']} !important;
    letter-spacing: -0.01em;
}}
.sb-logo-text .t2 {{
    font-size: 0.72rem; color: {C['text3']} !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}}

/* ── Sidebar section labels ── */
.sb-nav-label {{
    font-size: 0.68rem; font-weight: 600; color: {C['text3']} !important;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 1.2rem 0 0.4rem;
}}

/* ── Model status pill ── */
.model-status-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {C['panel']}; border: 1px solid {C['border2']};
    border-radius: 20px; padding: 0.3rem 0.75rem;
    font-size: 0.78rem; color: {C['text2']};
    white-space: nowrap;
}}
.model-status-pill .dot {{ width: 7px; height: 7px; border-radius: 50%; }}
/* Note: box-shadow uses #RRGGBBAA — valid browser CSS4, NOT a Plotly property */
.dot-green {{ background: {C['active']}; box-shadow: 0 0 6px {C['active']}88; }}
.dot-red   {{ background: {C['inactive']}; }}

/* ── Page header ── */
.page-header {{
    display: flex; align-items: flex-start;
    justify-content: space-between; flex-wrap: wrap; gap: 0.6rem;
    padding: 0 0 1.2rem;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 1.4rem;
}}
.page-header-left h1 {{
    margin: 0; font-size: 1.55rem; font-weight: 700;
    color: {C['text']}; letter-spacing: -0.02em;
}}
.page-header-left p {{ margin: 0.25rem 0 0; font-size: 0.85rem; color: {C['text2']}; }}

/* ── Stat cards ── */
.stat-row {{ display: flex; gap: 10px; margin-bottom: 0.8rem; }}
.stat-card {{
    flex: 1; background: {C['panel2']};
    border: 1px solid {C['border']};
    border-radius: 10px; padding: 0.7rem 0.8rem;
    text-align: center;
}}
.stat-card .sc-val {{
    font-size: 1.3rem; font-weight: 700; color: {C['teal']}; line-height: 1.1;
}}
.stat-card .sc-lbl {{
    font-size: 0.68rem; color: {C['text3']};
    text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px;
}}

/* ── Prediction result block ── */
.pred-block {{
    border-radius: 10px; padding: 0.9rem 1rem;
    border-width: 1px; border-style: solid; margin-bottom: 0.8rem;
}}
.pred-block.active  {{ background: {C['active_bg']};  border-color: {C['active_bd']};  }}
.pred-block.inactive {{ background: {C['inactive_bg']}; border-color: {C['inactive_bd']}; }}
.pred-label {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
               letter-spacing: 0.1em; margin-bottom: 3px; }}
.pred-label.active   {{ color: {C['active']}; }}
.pred-label.inactive {{ color: {C['inactive']}; }}
.pred-value {{ font-size: 1.7rem; font-weight: 700; line-height: 1.1; }}
.pred-value.active   {{ color: {C['active']}; }}
.pred-value.inactive {{ color: {C['inactive']}; }}
.pred-sub {{ font-size: 0.8rem; color: {C['text2']}; margin-top: 3px; }}

/* ── Disclaimer strip ── */
.disclaimer-strip {{
    background: {C['warn_bg']}; border: 1px solid {C['warn_bd']};
    border-radius: 8px; padding: 0.55rem 0.8rem;
    font-size: 0.76rem; color: {C['warn_text']}; line-height: 1.5;
}}

/* ── Section heading ── */
.sec-heading {{
    font-size: 0.72rem; font-weight: 600; color: {C['text3']};
    text-transform: uppercase; letter-spacing: 0.1em;
    display: flex; align-items: center; gap: 8px; margin-bottom: 0.7rem;
}}
.sec-heading::after {{ content: ''; flex: 1; height: 1px; background: {C['border']}; }}

/* ── Performance metric cards ── */
.perf-grid {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1rem; }}
.perf-card {{
    flex: 1; min-width: 120px;
    background: {C['panel2']}; border: 1px solid {C['border']};
    border-radius: 10px; padding: 0.85rem 1rem;
}}
.perf-card .pc-label {{
    font-size: 0.68rem; color: {C['text3']};
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
}}
.perf-card .pc-value {{ font-size: 1.5rem; font-weight: 700; color: {C['teal']}; }}
.perf-card .pc-sub  {{ font-size: 0.72rem; color: {C['text3']}; margin-top: 2px; }}

/* ── Content cards ── */
.content-card {{
    background: {C['panel']}; border: 1px solid {C['border']};
    border-radius: 12px; padding: 1.4rem 1.5rem;
}}
.content-card h3 {{
    font-size: 0.85rem; font-weight: 600; color: {C['teal']};
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem;
}}
.content-card p, .content-card li {{
    font-size: 0.88rem; color: {C['text2']}; line-height: 1.65;
}}

/* ── Sidebar model metrics ── */
.sb-metric {{
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.3rem 0; border-bottom: 1px solid {C['border']};
}}
.sb-metric .sbm-label {{ font-size: 0.78rem; color: {C['text2']} !important; }}
.sb-metric .sbm-val   {{
    font-size: 0.82rem; font-weight: 600; color: {C['teal']} !important;
}}

/* ── Footer ── */
.footer-bar {{
    border-top: 1px solid {C['border']}; margin-top: 2rem; padding-top: 1rem;
    text-align: center; font-size: 0.75rem; color: {C['text3']};
}}

/* ── Landing "how it works" ── */
.how-step {{
    display: flex; gap: 12px; align-items: flex-start; margin-bottom: 0.7rem;
}}
.how-step-num {{
    flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%;
    background: {C['teal']}; color: #0B1120;
    font-size: 0.72rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}}
.how-step-text {{ font-size: 0.87rem; color: {C['text2']}; padding-top: 2px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: display name → SMILES string.
# Every SMILES has been validated with RDKit (MolFromSmiles returns non-None).
# The selector is grouped by drug class for easier navigation.
EXAMPLES: dict[str, str] = {
    "— select a molecule —": "",
    # ── HIV antiretrovirals ──────────────────────────────────────────────────
    "AZT — Zidovudine (NRTI)":            "Cc1cn([C@@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)c(=O)[nH]c1=O",
    "Lamivudine / 3TC (NRTI)":            "OCC1CSC(O1)N1C=CC(=O)NC1=O",
    "Abacavir (NRTI)":                    "Nc1nc2c(ncn2[C@@H]2C[C@H](CO)C=C2)c(=O)[nH]1",
    "Efavirenz (NNRTI)":                  "O=C1O[C@@](CC#C)(c2cc(Cl)cc(F)c2)NC1=O",
    "Tenofovir (NtRTI)":                  "Nc1ccn([C@@H]2CC(COP(O)(O)=O)C2)c(=O)n1",
    "Nevirapine (NNRTI)":                 "Cc1ccnc2N(C3CC3)c4ncccc4C(=O)Nc12",
    "Lopinavir (protease inhibitor)":     (
        "CC(C)(C)NC(=O)[C@@H]1CN(CC[C@H](Cc2ccccc2)NC(=O)[C@@H](CC(C)C)"
        "NC(=O)c2ccc3ccccc3n2)C[C@@H]1O"
    ),
    # ── Common drugs (non-HIV) ───────────────────────────────────────────────
    "Aspirin (anti-inflammatory)":        "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine (stimulant)":               "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Ibuprofen (NSAID)":                  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Paracetamol (analgesic)":            "CC(=O)Nc1ccc(O)cc1",
    "Penicillin G (antibiotic)":          "CC1(C)SC2C(NC1=O)C(=O)N2Cc1ccccc1",
    "Chloroquine (antimalarial)":         "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Atorvastatin (statin)":              (
        "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)"
        "n1CC[C@@H](O)C[C@@H](O)CC(O)=O"
    ),
    "Morphine (opioid analgesic)":        "[C@@H]12CC(=CC[C@@H]1[C@@H]3CC4=C(O)C=CC=C4[C@]3(CCN2C)O)O",
    "Sildenafil (PDE5 inhibitor)":        "CCCC1=NN(C)C2=C1NC(=Nc1cc(S(=O)(=O)N3CCN(C)CC3)ccc1OCC)N2",
    "Tamoxifen (SERM)":                   "CCC(=C(c1ccccc1)/C=C/c1ccccc1)c1ccc(OCCN(C)C)cc1",
    "Metformin (antidiabetic)":           "CN(C)C(=N)NC(N)=N",
    "Doxorubicin (chemotherapy)":         (
        "COc1cccc2C(=O)c3c(O)c4c(c(O)c3C(=O)c12)"
        "C[C@@](O)(C(C)=O)C[C@@H]4O[C@H]1C[C@@H](N)[C@@H](O)C(C)O1"
    ),
    # ── Simple / reference molecules ────────────────────────────────────────
    "Benzene (aromatic ring)":            "c1ccccc1",
    "Glucose (simple sugar)":             "OCC1OC(O)C(O)C(O)C1O",
    "Dopamine (neurotransmitter)":        "NCCc1ccc(O)c(O)c1",
    "Ethanol (alcohol)":                  "CCO",
}

# Pre-computed inference results for the Explore Molecules table.
# Generated by running every SMILES above through the real trained model
# (inference.py → predict()). Never fabricated.
# Format: (display_name, smiles, drug_class, prob_active, label)
# Probabilities are rounded to 4 decimal places from the model output.
EXPLORE_MOLECULES: list[tuple[str, str, str, float, str]] = [
    # name                             smiles                                       class               prob    label
    ("AZT — Zidovudine",       "Cc1cn([C@@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)c(=O)[nH]c1=O",  "HIV antiretroviral",  0.3404, "Inactive"),
    ("Lamivudine / 3TC",       "OCC1CSC(O1)N1C=CC(=O)NC1=O",                                    "HIV antiretroviral",  0.2324, "Inactive"),
    ("Abacavir",               "Nc1nc2c(ncn2[C@@H]2C[C@H](CO)C=C2)c(=O)[nH]1",                 "HIV antiretroviral",  0.0194, "Inactive"),
    ("Efavirenz",              "O=C1O[C@@](CC#C)(c2cc(Cl)cc(F)c2)NC1=O",                        "HIV antiretroviral",  0.0064, "Inactive"),
    ("Tenofovir",              "Nc1ccn([C@@H]2CC(COP(O)(O)=O)C2)c(=O)n1",                       "HIV antiretroviral",  0.0895, "Inactive"),
    ("Nevirapine",             "Cc1ccnc2N(C3CC3)c4ncccc4C(=O)Nc12",                             "HIV antiretroviral",  0.0095, "Inactive"),
    ("Lopinavir",              "CC(C)(C)NC(=O)[C@@H]1CN(CC[C@H](Cc2ccccc2)NC(=O)[C@@H](CC(C)C)NC(=O)c2ccc3ccccc3n2)C[C@@H]1O",  "HIV antiretroviral",  0.0175, "Inactive"),
    ("Aspirin",                "CC(=O)Oc1ccccc1C(=O)O",                                         "Common drug",         0.0200, "Inactive"),
    ("Caffeine",               "Cn1cnc2c1c(=O)n(c(=O)n2C)C",                                    "Common drug",         0.0088, "Inactive"),
    ("Ibuprofen",              "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                                    "Common drug",         0.0111, "Inactive"),
    ("Paracetamol",            "CC(=O)Nc1ccc(O)cc1",                                             "Common drug",         0.0097, "Inactive"),
    ("Penicillin G",           "CC1(C)SC2C(NC1=O)C(=O)N2Cc1ccccc1",                             "Common drug",         0.0158, "Inactive"),
    ("Chloroquine",            "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",                             "Common drug",         0.0086, "Inactive"),
    ("Atorvastatin",           "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(O)=O",  "Common drug",  0.0226, "Inactive"),
    ("Morphine",               "[C@@H]12CC(=CC[C@@H]1[C@@H]3CC4=C(O)C=CC=C4[C@]3(CCN2C)O)O",  "Common drug",         0.0092, "Inactive"),
    ("Sildenafil",             "CCCC1=NN(C)C2=C1NC(=Nc1cc(S(=O)(=O)N3CCN(C)CC3)ccc1OCC)N2",   "Common drug",         0.0302, "Inactive"),
    ("Tamoxifen",              "CCC(=C(c1ccccc1)/C=C/c1ccccc1)c1ccc(OCCN(C)C)cc1",             "Common drug",         0.0063, "Inactive"),
    ("Metformin",              "CN(C)C(=N)NC(N)=N",                                              "Common drug",         0.0068, "Inactive"),
    ("Doxorubicin",            "COc1cccc2C(=O)c3c(O)c4c(c(O)c3C(=O)c12)C[C@@](O)(C(C)=O)C[C@@H]4O[C@H]1C[C@@H](N)[C@@H](O)C(C)O1",  "Common drug",  0.0266, "Inactive"),
    ("Benzene",                "c1ccccc1",                                                        "Reference",           0.0075, "Inactive"),
    ("Glucose",                "OCC1OC(O)C(O)C(O)C1O",                                           "Reference",           0.0147, "Inactive"),
    ("Dopamine",               "NCCc1ccc(O)c(O)c1",                                              "Reference",           0.0151, "Inactive"),
    ("Ethanol",                "CCO",                                                             "Reference",           0.0084, "Inactive"),
]

VAL_AUC  = 0.8266
TEST_AUC = 0.7906

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# All mutable UI state lives here. Streamlit reruns the whole script on every
# interaction; session_state persists values across those reruns.
# ─────────────────────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "last_analysed_smiles": "",      # SMILES of the most recently run analysis
    "analysis_result":      None,    # dict returned by predict(), or None
    "analysis_error":       None,    # str error message, or None
    "smiles_text":          "",      # current text-area content
    "analysis_triggered":   False,   # True only on the run where button was clicked
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# Model / checkpoint status (computed once per run)
# ─────────────────────────────────────────────────────────────────────────────
ckpt_exists = os.path.exists(CHECKPOINT_PATH)
ckpt_mb     = os.path.getsize(CHECKPOINT_PATH) / (1024 ** 2) if ckpt_exists else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-hex"></div>
      <div class="sb-logo-text">
        <div class="t1">Drug Discovery AI</div>
        <div class="t2">GNN Research Platform</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        label="page",
        options=["Molecule Analysis", "Explore Molecules", "Model Performance", "About the Project"],
        label_visibility="collapsed",
    )

    # ── Molecule input (only functional on Molecule Analysis page) ────────────
    st.markdown('<div class="sb-nav-label">Molecule Input</div>', unsafe_allow_html=True)

    example_choice = st.selectbox(
        "Example molecule",
        options=list(EXAMPLES.keys()),
        index=0,
        label_visibility="collapsed",
        key="example_selectbox",
    )

    # When a real example is selected, populate the text area and auto-trigger
    selected_smiles = EXAMPLES.get(example_choice, "")
    if selected_smiles and selected_smiles != st.session_state.get("smiles_text", ""):
        st.session_state["smiles_text"] = selected_smiles
        st.session_state["analysis_triggered"] = True

    smiles_text_value = st.text_area(
        "SMILES string",
        value=st.session_state["smiles_text"],
        height=80,
        placeholder="e.g.  CC(=O)Oc1ccccc1C(=O)O",
        label_visibility="collapsed",
        key="smiles_textarea",
    )
    # Sync text area back to session state (user may have typed directly)
    st.session_state["smiles_text"] = smiles_text_value

    analyse_clicked = st.button(
        "Analyse Molecule",
        use_container_width=True,
        type="primary",
        disabled=(page != "Molecule Analysis"),
    )
    if analyse_clicked:
        st.session_state["analysis_triggered"] = True

    # ── Model status ──────────────────────────────────────────────────────────
    st.markdown('<div class="sb-nav-label">Model Status</div>', unsafe_allow_html=True)
    if ckpt_exists:
        st.markdown(
            f'<div class="model-status-pill" style="margin-bottom:0.6rem">'
            f'<span class="dot dot-green"></span>'
            f'Checkpoint ready &nbsp;·&nbsp; {ckpt_mb:.1f} MB</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="model-status-pill" style="margin-bottom:0.6rem">'
            '<span class="dot dot-red"></span>Checkpoint missing</div>',
            unsafe_allow_html=True,
        )

    for _label, _val in [
        ("Val ROC-AUC",  f"{VAL_AUC:.4f}"),
        ("Test ROC-AUC", f"{TEST_AUC:.4f}"),
        ("Architecture", "GINEConv × 5"),
        ("Hidden dim",   "128"),
        ("Pooling",      "global_add_pool"),
        ("Dataset",      "ogbg-molhiv"),
    ]:
        st.markdown(
            f'<div class="sb-metric">'
            f'<span class="sbm-label">{_label}</span>'
            f'<span class="sbm-val">{_val}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<p style="font-size:0.72rem;color:{C["text3"]};margin-top:0.8rem;">'
        "For academic demonstration only. Not a medical tool.</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: shared page header
# ─────────────────────────────────────────────────────────────────────────────
def _page_header(title: str, subtitle: str) -> None:
    dot_cls  = "dot-green" if ckpt_exists else "dot-red"
    dot_text = (
        f"Model ready &nbsp;·&nbsp; {ckpt_mb:.1f} MB"
        if ckpt_exists else "Checkpoint missing"
    )
    st.markdown(
        f"""<div class="page-header">
          <div class="page-header-left">
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
          <div class="model-status-pill">
            <span class="dot {dot_cls}"></span>{dot_text}
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render the full analysis results for a given SMILES
# Called only after the button is clicked and SMILES is valid.
# ─────────────────────────────────────────────────────────────────────────────
def _render_analysis(smiles: str) -> None:
    """Render 2D structure, prediction, molecular graph, stats, and properties."""

    # ── Row 1: 2D structure | GNN prediction ─────────────────────────────────
    col_struct, col_pred = st.columns([1, 1], gap="large")

    with col_struct:
        st.markdown('<div class="sec-heading">2D Chemical Structure</div>',
                    unsafe_allow_html=True)
        try:
            img_b64 = mol_to_image_b64(smiles, width=520, height=320)
            st.markdown(
                f'<div style="background:{C["panel"]};border:1px solid {C["border"]};'
                f'border-radius:12px;overflow:hidden;padding:8px;">'
                f'<img src="{img_b64}" style="width:100%;display:block;" alt="2D structure">'
                f'</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Could not render 2D structure: {exc}")

    with col_pred:
        st.markdown('<div class="sec-heading">HIV Activity Prediction</div>',
                    unsafe_allow_html=True)
        if not ckpt_exists:
            st.error(
                f"Checkpoint not found: `{CHECKPOINT_PATH}`. "
                "Ensure `best_gnn_model.pt` is in the project root."
            )
        else:
            with st.spinner("Running GNN inference…"):
                try:
                    result  = predict(smiles)
                    prob    = result["probability"]
                    label   = result["label"]
                    is_act  = (label == "Active")
                    cls_css = "active" if is_act else "inactive"
                    prob_inactive = 1.0 - prob

                    # ── Prediction badge ──────────────────────────────────────
                    st.markdown(
                        f'<div class="pred-block {cls_css}">'
                        f'<div class="pred-label {cls_css}">Predicted class</div>'
                        f'<div class="pred-value {cls_css}">{label}</div>'
                        f'<div class="pred-sub">'
                        f'HIV-active probability: <strong>{prob:.1%}</strong>'
                        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                        f'Inactive probability: <strong>{prob_inactive:.1%}</strong>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Probability bar chart ─────────────────────────────────
                    # All Plotly colors use rgba() — 8-digit hex is NOT supported
                    # by Plotly 6 and will raise an Invalid element error.
                    bar_fig = go.Figure()
                    bar_fig.add_trace(go.Bar(
                        x=["Inactive", "Active"],
                        y=[prob_inactive, prob],
                        marker_color=[
                            "rgba(248,113,113,0.33)",  # inactive fill
                            "rgba(16,185,129,0.33)",   # active fill
                        ],
                        marker_line_color=[C["inactive"], C["active"]],
                        marker_line_width=1.5,
                        text=[f"{prob_inactive:.1%}", f"{prob:.1%}"],
                        textposition="outside",
                        textfont=dict(color=C["text"], size=13, family="Inter"),
                        width=0.5,
                    ))
                    bar_fig.update_layout(
                        height=230,
                        margin=dict(l=10, r=10, t=14, b=30),
                        plot_bgcolor=C["panel"],
                        paper_bgcolor=C["panel"],
                        yaxis=dict(
                            range=[0, 1.22],
                            tickformat=".0%",
                            showgrid=True,
                            gridcolor=C["border"],
                            tickfont=dict(color=C["text2"], size=11),
                            title=dict(
                                text="Predicted probability",
                                font=dict(color=C["text2"], size=11),
                            ),
                        ),
                        xaxis=dict(tickfont=dict(color=C["text2"], size=12)),
                        showlegend=False,
                        title=dict(
                            text="Class probabilities",
                            font=dict(color=C["text2"], size=11),
                            x=0.0, xref="paper",
                        ),
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)

                    # ── Disclaimer ────────────────────────────────────────────
                    st.markdown(
                        '<div class="disclaimer-strip">'
                        "<strong>Computational prediction only.</strong> "
                        "Output from a GNN trained on OGB-MolHIV. "
                        "Not a confirmed HIV-inhibition result, drug approval, "
                        "or medical recommendation. For academic demonstration only."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    with st.expander("Technical details"):
                        st.markdown(
                            "| Parameter | Value |\n|---|---|\n"
                            f"| Raw logit | `{result['logit']:.5f}` |\n"
                            f"| Sigmoid output (HIV-active prob.) | `{prob:.5f}` |\n"
                            f"| Inactive probability | `{prob_inactive:.5f}` |\n"
                            f"| Decision threshold | `0.5` |\n"
                            f"| Checkpoint | `{result['load_info']}` |"
                        )

                except FileNotFoundError as exc:
                    st.error(f"Checkpoint error: {exc}")
                except ValueError as exc:
                    st.error(f"Architecture mismatch: {exc}")
                except Exception:
                    st.error("Unexpected inference error — see details below.")
                    with st.expander("Error traceback"):
                        st.code(_traceback.format_exc())

    # ── Row 2: Interactive molecular graph ────────────────────────────────────
    st.markdown(
        '<div class="sec-heading" style="margin-top:0.8rem">'
        'Interactive Molecular Graph</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Atoms are shown as coloured nodes (hover for OGB feature vector). "
        "Bonds are shown as coloured edges by bond type. "
        "Small numbers below nodes are atom indices."
    )
    try:
        graph_fig, stats = build_graph_figure(smiles, dark=True)
        st.plotly_chart(graph_fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not build molecular graph: {exc}")
        with st.expander("Error traceback"):
            st.code(_traceback.format_exc())
        stats = {}

    # ── Row 3: Graph statistics | Molecular properties ────────────────────────
    col_gs, col_mp = st.columns(2, gap="large")

    with col_gs:
        st.markdown('<div class="sec-heading">Graph Statistics</div>',
                    unsafe_allow_html=True)
        if stats:
            st.markdown(
                f'<div class="stat-row">'
                f'<div class="stat-card"><div class="sc-val">{stats["num_nodes"]}</div>'
                f'<div class="sc-lbl">Nodes (atoms)</div></div>'
                f'<div class="stat-card"><div class="sc-val">{stats["num_edges"]}</div>'
                f'<div class="sc-lbl">Edges (bonds)</div></div>'
                f'<div class="stat-card"><div class="sc-val">{stats["avg_degree"]}</div>'
                f'<div class="sc-lbl">Avg degree</div></div>'
                f'<div class="stat-card"><div class="sc-val">{stats["node_feature_dim"]}</div>'
                f'<div class="sc-lbl">Node feat dim</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            gs_df = pd.DataFrame([
                ("Directed edges (OGB)", stats["num_directed_edges"]),
                ("Node feature dim",     stats["node_feature_dim"]),
                ("Edge feature dim",     stats["edge_feature_dim"]),
                ("Connected graph",      "Yes" if stats["is_connected"] else "No"),
            ], columns=["Property", "Value"])
            st.dataframe(gs_df, use_container_width=True, hide_index=True, height=158)

            c1, c2 = st.columns(2)
            with c1:
                st.caption("Element composition")
                elem_df = pd.DataFrame(
                    sorted(stats["element_counts"].items(), key=lambda x: -x[1]),
                    columns=["Element", "Count"],
                )
                st.dataframe(elem_df, use_container_width=True, hide_index=True)
            with c2:
                st.caption("Bond distribution")
                bond_df = pd.DataFrame(
                    sorted(stats["bond_type_counts"].items(), key=lambda x: -x[1]),
                    columns=["Bond", "Count"],
                )
                st.dataframe(bond_df, use_container_width=True, hide_index=True)

    with col_mp:
        st.markdown('<div class="sec-heading">Molecular Properties</div>',
                    unsafe_allow_html=True)
        try:
            props      = compute_properties(smiles)
            ro5_detail = props.pop("_ro5_detail", {})

            st.markdown(
                f'<div class="stat-row">'
                f'<div class="stat-card"><div class="sc-val">{props["Exact MW (Da)"]}</div>'
                f'<div class="sc-lbl">MW (Da)</div></div>'
                f'<div class="stat-card"><div class="sc-val">{props["LogP"]}</div>'
                f'<div class="sc-lbl">LogP</div></div>'
                f'<div class="stat-card"><div class="sc-val">{props["H-Bond Donors"]}</div>'
                f'<div class="sc-lbl">HBD</div></div>'
                f'<div class="stat-card"><div class="sc-val">{props["H-Bond Acceptors"]}</div>'
                f'<div class="sc-lbl">HBA</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            prop_df = pd.DataFrame(
                [(k, v) for k, v in props.items()],
                columns=["Property", "Value"],
            )
            st.dataframe(prop_df, use_container_width=True, hide_index=True, height=252)

            if ro5_detail:
                st.caption("Lipinski Rule-of-Five")
                ro5_df = pd.DataFrame(
                    [(rule, "Pass" if ok else "Fail")
                     for rule, ok in ro5_detail.items()],
                    columns=["Rule", "Result"],
                )
                st.dataframe(ro5_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not compute molecular properties: {exc}")

    # ── Row 4: OGB feature tables (collapsed) ─────────────────────────────────
    with st.expander("OGB Atom Feature Matrix — raw input to GNN"):
        try:
            from ogb.utils.mol import smiles2graph as _s2g
            from rdkit import Chem as _Chem
            _mol = _Chem.MolFromSmiles(smiles)
            _g   = _s2g(smiles)
            _nf  = _g["node_feat"]
            _col_names = [
                "Atomic#", "Chirality", "Degree", "FormalChg",
                "NumHs", "NumRadical", "Hybridise", "IsAromatic", "IsInRing",
            ]
            _feat_df = pd.DataFrame(_nf, columns=_col_names)
            _feat_df.insert(0, "Symbol", [a.GetSymbol() for a in _mol.GetAtoms()])
            _feat_df.insert(0, "AtomIdx", list(range(len(_mol.GetAtoms()))))
            st.caption(
                "OGB categorical codes (not raw atomic numbers). "
                "Each column is mapped through a learned embedding table in AtomEncoder."
            )
            st.dataframe(_feat_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not build atom feature table: {exc}")

    with st.expander("OGB Bond Feature Matrix — raw edge attributes"):
        try:
            from ogb.utils.mol import smiles2graph as _s2g2
            _g2  = _s2g2(smiles)
            _ef  = _g2["edge_feat"]
            _ei  = _g2["edge_index"]
            _bdf = pd.DataFrame(_ef, columns=["BondType", "Stereo", "IsConjugated"])
            _bdf.insert(0, "To",      _ei[1].tolist())
            _bdf.insert(0, "From",    _ei[0].tolist())
            _bdf.insert(0, "EdgeIdx", list(range(len(_ef))))
            st.caption(
                "Each undirected bond is stored twice (both directions). "
                "BondType: 0=SINGLE, 1=DOUBLE, 2=TRIPLE, 3=AROMATIC."
            )
            st.dataframe(_bdf, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not build bond feature table: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Molecule Analysis ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if page == "Molecule Analysis":
    _page_header(
        "AI Drug Discovery Platform",
        "Molecular graph analysis · GNN-based HIV activity prediction · OGB-MolHIV benchmark",
    )

    # ── Run analysis if triggered ─────────────────────────────────────────────
    if st.session_state["analysis_triggered"]:
        st.session_state["analysis_triggered"] = False  # consume the trigger

        raw = st.session_state["smiles_text"].strip()
        if not raw:
            st.session_state["last_analysed_smiles"] = ""
            st.session_state["analysis_error"] = "Please enter a SMILES string before analysing."
        else:
            is_valid, val_msg = validate_smiles(raw)
            if not is_valid:
                st.session_state["last_analysed_smiles"] = ""
                st.session_state["analysis_error"] = val_msg
            else:
                st.session_state["last_analysed_smiles"] = raw
                st.session_state["analysis_error"] = None

    last_smiles = st.session_state["last_analysed_smiles"]
    error_msg   = st.session_state["analysis_error"]

    # ── Show error if last action produced one ────────────────────────────────
    if error_msg:
        st.error(f"Invalid SMILES: {error_msg}")
        st.info(
            "Check the SMILES string and click **Analyse Molecule** again, "
            "or pick an example from the sidebar."
        )

    # ── Show results if a valid SMILES has been analysed ──────────────────────
    elif last_smiles:
        st.success(f"Analysing: `{last_smiles}`")
        _render_analysis(last_smiles)

    # ── Landing state (nothing submitted yet) ─────────────────────────────────
    else:
        col_l, col_r = st.columns([1.1, 1], gap="large")

        with col_l:
            st.markdown(f"""
            <div class="content-card">
              <h3>How to use this tool</h3>
              <div class="how-step">
                <div class="how-step-num">1</div>
                <div class="how-step-text">
                  Pick an <strong style="color:{C['text']}">example molecule</strong> from the
                  sidebar dropdown, or paste your own SMILES string into the text box.
                </div>
              </div>
              <div class="how-step">
                <div class="how-step-num">2</div>
                <div class="how-step-text">
                  Click <strong style="color:{C['teal']}">Analyse Molecule</strong>. Selecting
                  an example automatically triggers the analysis.
                </div>
              </div>
              <div class="how-step">
                <div class="how-step-num">3</div>
                <div class="how-step-text">
                  View the <strong style="color:{C['text']}">2D chemical structure</strong>,
                  the <strong style="color:{C['text']}">interactive molecular graph</strong>
                  (hover atoms for feature details), and the
                  <strong style="color:{C['text']}">GNN-predicted HIV-activity probability</strong>.
                </div>
              </div>
              <div class="how-step">
                <div class="how-step-num">4</div>
                <div class="how-step-text">
                  Explore drug-likeness properties, element composition, bond statistics,
                  and the raw OGB feature matrices fed into the GNN.
                </div>
              </div>
              <div class="how-step">
                <div class="how-step-num">5</div>
                <div class="how-step-text">
                  Enter a different molecule and click <strong style="color:{C['teal']}">
                  Analyse Molecule</strong> again to repeat the analysis.
                </div>
              </div>
              <p style="margin-top:1rem;font-size:0.82rem;color:{C['text3']};">
                A <strong style="color:{C['text2']}">GNN</strong> treats each molecule as a
                graph. Atoms are nodes with 9-dimensional feature vectors; bonds are edges
                with 3-dimensional feature vectors. Five GINEConv layers aggregate
                neighbourhood information, and global pooling produces a single vector
                that the classifier head maps to a probability.
              </p>
            </div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown(f"""
            <div class="content-card" style="padding-bottom:0.8rem">
              <h3>Example molecules</h3>
            </div>""", unsafe_allow_html=True)
            examples_df = pd.DataFrame(
                [{"Molecule": k, "SMILES": v} for k, v in EXAMPLES.items() if v],
                columns=["Molecule", "SMILES"],
            )
            st.dataframe(
                examples_df,
                use_container_width=True,
                hide_index=True,
                height=340,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Explore Molecules ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Explore Molecules":
    _page_header(
        "Explore Molecules",
        "Pre-computed GNN predictions for 23 molecules — real model output, no fabrication",
    )

    # ── Context note ─────────────────────────────────────────────────────────
    st.markdown(
        f"""<div class="disclaimer-strip" style="margin-bottom:1rem">
          <strong>About these predictions:</strong>
          Every probability in this table was produced by running the SMILES string
          through the trained GNN checkpoint (<code>best_gnn_model.pt</code>) using
          the same <code>inference.py</code> pipeline as the Molecule Analysis page.
          No values were fabricated or adjusted. The model probabilities are fixed —
          only the classification threshold can be changed below.
          See <em>Understanding the results</em> at the bottom for context.
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Build base dataframe with raw probabilities ───────────────────────────
    # Keep raw floats in a separate working copy; formatted strings are for display.
    _explore_rows = [
        {
            "Molecule":   name,
            "Drug class": drug_class,
            "_prob_raw":  prob,   # raw float — never modified
            "SMILES":     smiles,
        }
        for name, smiles, drug_class, prob, _stored_label in EXPLORE_MOLECULES
    ]
    base_df = pd.DataFrame(_explore_rows)

    # ── Threshold slider ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-heading">Classification threshold</div>',
                unsafe_allow_html=True)

    _thresh_col, _thresh_info_col = st.columns([2, 3], gap="large")

    with _thresh_col:
        threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            format="%.2f",
            key="explore_threshold",
            label_visibility="collapsed",
        )

    with _thresh_info_col:
        _default_note = " (model default)" if abs(threshold - 0.50) < 0.001 else ""
        st.markdown(
            f'<div style="padding:0.55rem 0.9rem;background:{C["panel2"]};'
            f'border:1px solid {C["border2"]};border-radius:8px;font-size:0.85rem;">'
            f'<span style="color:{C["text2"]}">Current threshold: </span>'
            f'<strong style="color:{C["teal"]}">{threshold:.0%}{_default_note}</strong>'
            f'<br><span style="color:{C["text3"]};font-size:0.76rem;">'
            f'HIV-active prob. ≥ {threshold:.0%} → <strong style="color:{C["active"]}">Active</strong>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'&lt; {threshold:.0%} → <strong style="color:{C["inactive"]}">Inactive</strong>'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    # Show a warning any time the threshold is not the model default
    if abs(threshold - 0.50) > 0.001:
        st.markdown(
            f'<div class="disclaimer-strip" style="margin-top:0.5rem;margin-bottom:0.3rem">'
            f'<strong>Threshold warning:</strong> You are using a non-default threshold of '
            f'<strong>{threshold:.0%}</strong>. Changing this value changes only the '
            f'<em>classification rule</em> applied to the model output — it does '
            f'<strong>not</strong> change the model\'s learned weights, the computed '
            f'probabilities, or a molecule\'s real biological activity. '
            f'Any "Active" classification at a lower threshold is a computational '
            f'estimate based on an exploratory cut-off, not experimental evidence.'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Apply the threshold to derive current Active/Inactive labels
    base_df["Predicted class"] = base_df["_prob_raw"].apply(
        lambda p: "Active" if p >= threshold else "Inactive"
    )
    base_df["HIV-active prob."] = base_df["_prob_raw"].apply(lambda p: f"{p:.2%}")

    # ── Summary counts at current threshold ───────────────────────────────────
    n_total       = len(base_df)
    n_active_now  = (base_df["Predicted class"] == "Active").sum()
    n_inactive_now = n_total - n_active_now

    st.markdown(
        f'<div class="stat-row" style="margin-top:0.6rem">'
        f'<div class="stat-card"><div class="sc-val">{n_total}</div>'
        f'<div class="sc-lbl">Total</div></div>'
        f'<div class="stat-card" style="border-color:{C["active_bd"]}">'
        f'<div class="sc-val" style="color:{C["active"]}">{n_active_now}</div>'
        f'<div class="sc-lbl">Active at {threshold:.0%}</div></div>'
        f'<div class="stat-card" style="border-color:{C["inactive_bd"]}">'
        f'<div class="sc-val" style="color:{C["inactive"]}">{n_inactive_now}</div>'
        f'<div class="sc-lbl">Inactive at {threshold:.0%}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Filter controls ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-heading" style="margin-top:0.8rem">'
                'Filter</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns([1, 1, 3], gap="small")
    with col_f1:
        show_active = st.checkbox("Active", value=True, key="explore_active")
    with col_f2:
        show_inactive = st.checkbox("Inactive", value=True, key="explore_inactive")
    with col_f3:
        class_options = ["All drug classes"] + sorted(
            base_df["Drug class"].unique().tolist()
        )
        class_filter = st.selectbox(
            "Drug class",
            options=class_options,
            label_visibility="collapsed",
            key="explore_class",
        )

    # Build filter mask against threshold-derived labels
    _wanted_classes = (
        (["Active"]   if show_active   else []) +
        (["Inactive"] if show_inactive else [])
    )
    mask = base_df["Predicted class"].isin(_wanted_classes)
    if class_filter != "All drug classes":
        mask = mask & (base_df["Drug class"] == class_filter)

    filtered_df = base_df[mask].reset_index(drop=True)
    n_shown = len(filtered_df)

    st.markdown(
        f'<p style="font-size:0.78rem;color:{C["text3"]};margin:0 0 0.6rem;">'
        f'Showing {n_shown} of {n_total} molecules</p>',
        unsafe_allow_html=True,
    )

    # ── Probability bar chart ─────────────────────────────────────────────────
    st.markdown('<div class="sec-heading">HIV-active probability by molecule</div>',
                unsafe_allow_html=True)
    st.caption(
        "Bars show the model's fixed HIV-active probability for each molecule. "
        "Bar colour reflects the classification at the current threshold. "
        "The dashed vertical line marks the selected threshold."
    )

    try:
        # Use filtered rows, sorted descending by probability
        _chart_rows = (
            filtered_df[["Molecule", "_prob_raw", "Predicted class"]]
            .sort_values("_prob_raw", ascending=False)
        )
        chart_names  = _chart_rows["Molecule"].tolist()
        chart_probs  = _chart_rows["_prob_raw"].tolist()
        chart_labels = _chart_rows["Predicted class"].tolist()

        # Colour bars by their label at the current threshold
        bar_colors = [
            "rgba(16,185,129,0.75)"  if lbl == "Active" else "rgba(56,189,248,0.60)"
            for lbl in chart_labels
        ]
        bar_borders = [
            C["active"] if lbl == "Active" else C["blue"]
            for lbl in chart_labels
        ]

        # X-axis upper bound: show at least up to threshold + 0.08 so the line
        # is always visible even when all probabilities are well below it.
        x_max = max(0.45, threshold + 0.08, max(chart_probs) + 0.06) if chart_probs else 0.55
        x_max = min(x_max, 1.0)

        explore_fig = go.Figure()
        explore_fig.add_trace(go.Bar(
            y=chart_names,
            x=chart_probs,
            orientation="h",
            marker_color=bar_colors,
            marker_line_color=bar_borders,
            marker_line_width=1.2,
            text=[f"{p:.1%}" for p in chart_probs],
            textposition="outside",
            textfont=dict(color=C["text2"], size=10, family="Inter"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "HIV-active probability: %{x:.2%}<br>"
                "<extra></extra>"
            ),
        ))
        # Threshold line — position and label update with slider
        explore_fig.add_vline(
            x=threshold,
            line_color=C["inactive"],
            line_dash="dot",
            line_width=1.8,
            annotation_text=f"Threshold: {threshold:.0%}",
            annotation_font_color=C["inactive"],
            annotation_font_size=10,
            annotation_position="top right",
        )
        explore_fig.update_layout(
            height=max(320, len(chart_names) * 28 + 70),
            margin=dict(l=10, r=90, t=30, b=20),
            plot_bgcolor=C["panel"],
            paper_bgcolor=C["panel"],
            xaxis=dict(
                range=[0, x_max],
                tickformat=".0%",
                showgrid=True,
                gridcolor=C["border"],
                tickfont=dict(color=C["text2"], size=10),
                title=dict(
                    text="HIV-active probability (model output, unchanged)",
                    font=dict(color=C["text2"], size=11),
                ),
            ),
            yaxis=dict(
                tickfont=dict(color=C["text"], size=11),
                autorange="reversed",
            ),
            showlegend=False,
        )
        st.plotly_chart(explore_fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not render probability chart: {exc}")
        with st.expander("Error traceback"):
            import traceback as _tb_exp
            st.code(_tb_exp.format_exc())

    # ── Filterable table ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-heading">Molecule table</div>',
                unsafe_allow_html=True)

    if filtered_df.empty:
        st.info("No molecules match the current filters.")
    else:
        display_df = (
            filtered_df
            .drop(columns=["_prob_raw", "SMILES"])
            .reset_index(drop=True)
        )
        display_df.index = display_df.index + 1
        st.dataframe(
            display_df,
            use_container_width=True,
            height=min(600, 38 + len(display_df) * 35),
        )
        with st.expander("Show SMILES strings"):
            smiles_df = filtered_df[["Molecule", "SMILES"]].reset_index(drop=True)
            smiles_df.index = smiles_df.index + 1
            st.dataframe(smiles_df, use_container_width=True)

    # ── Understanding the results ─────────────────────────────────────────────
    st.markdown('<div class="sec-heading" style="margin-top:1rem">'
                'Understanding the results</div>',
                unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2, gap="large")
    with col_e1:
        st.markdown(f"""
        <div class="content-card">
          <h3>Why probabilities are low</h3>
          <p>
            The model was trained on the
            <strong style="color:{C['text']}">OGB-MolHIV dataset</strong>
            (41,127 molecules, ~3.6% active). A model trained on such a
            heavily imbalanced set learns that predicting <em>Inactive</em>
            is usually correct, so the raw probabilities are skewed toward
            zero even for known HIV drugs.
          </p>
          <p style="margin-top:0.6rem">
            The model's discriminative power is captured by
            <strong style="color:{C['teal']}">ROC-AUC = 0.7906</strong>:
            it ranks active molecules above inactive ones ~79% of the time.
            The absolute probabilities are not calibrated to real-world rates.
          </p>
          <p style="margin-top:0.6rem">
            <strong style="color:{C['text']}">AZT (34.0%)</strong> and
            <strong style="color:{C['text']}">Lamivudine (23.2%)</strong>
            score highest — the model ranks HIV drugs above common drugs
            such as Aspirin (2.0%) and Caffeine (0.9%), which is the
            expected direction even though no molecule crosses 50%.
          </p>
        </div>""", unsafe_allow_html=True)

    with col_e2:
        st.markdown(f"""
        <div class="content-card">
          <h3>What the threshold slider does and does not do</h3>
          <p>
            Moving the slider <strong style="color:{C['text']}">changes only
            the classification rule</strong> applied to the model's output:
          </p>
          <ul>
            <li>
              The model's learned weights are <strong style="color:{C['text']}">
              not modified</strong>.
            </li>
            <li style="margin-top:0.4rem">
              The computed HIV-active probabilities are
              <strong style="color:{C['text']}">not modified</strong>.
            </li>
            <li style="margin-top:0.4rem">
              A molecule's real biological activity is
              <strong style="color:{C['text']}">not affected</strong>.
            </li>
          </ul>
          <p style="margin-top:0.6rem">
            Lowering the threshold allows you to see which molecules the
            model ranks highest relative to others. It is useful for
            <em>exploratory ranking</em>, but any resulting "Active"
            classification is a <strong>computational estimate only</strong>
            — not experimental confirmation of HIV inhibition, drug efficacy,
            or safety. This model cannot replace laboratory assays or clinical
            trials.
          </p>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: Model Performance ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    _page_header(
        "Model Performance",
        "Evaluation metrics from training on the OGB-MolHIV benchmark",
    )

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown(
        f"""<div class="perf-grid">
          <div class="perf-card">
            <div class="pc-label">Validation ROC-AUC</div>
            <div class="pc-value">{VAL_AUC:.4f}</div>
            <div class="pc-sub">Scaffold split · OGB evaluator</div>
          </div>
          <div class="perf-card">
            <div class="pc-label">Test ROC-AUC</div>
            <div class="pc-value">{TEST_AUC:.4f}</div>
            <div class="pc-sub">Scaffold split · OGB evaluator</div>
          </div>
          <div class="perf-card">
            <div class="pc-label">GNN Layers</div>
            <div class="pc-value">5</div>
            <div class="pc-sub">GINEConv message-passing</div>
          </div>
          <div class="perf-card">
            <div class="pc-label">Hidden Dimension</div>
            <div class="pc-value">128</div>
            <div class="pc-sub">Atoms, bonds, graph embedding</div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")

    # ── ROC-AUC bar chart (real values only) ──────────────────────────────────
    with col_a:
        st.markdown('<div class="sec-heading">ROC-AUC Comparison</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Bars show the real Val/Test ROC-AUC scores from the training run, "
            "compared against the random-chance baseline of 0.5. "
            "No ROC curve data was saved at training time, so only the scalar "
            "scores are displayed."
        )
        try:
            auc_fig = go.Figure()
            auc_fig.add_trace(go.Bar(
                x=["Random baseline", "Validation ROC-AUC", "Test ROC-AUC"],
                y=[0.5, VAL_AUC, TEST_AUC],
                # All colors are rgba() — Plotly 6 does not accept 8-digit hex (#RRGGBBAA)
                marker_color=[
                    "rgba(100,116,139,0.45)",   # muted slate
                    "rgba(56,189,248,0.75)",    # blue
                    "rgba(45,212,191,0.85)",    # teal
                ],
                marker_line_color=[C["text3"], C["blue"], C["teal"]],
                marker_line_width=1.5,
                text=["0.5000", f"{VAL_AUC:.4f}", f"{TEST_AUC:.4f}"],
                textposition="outside",
                textfont=dict(color=C["text"], size=13, family="Inter"),
                width=0.48,
                showlegend=False,
            ))
            auc_fig.add_hline(
                y=0.5,
                line_color=C["text3"],
                line_dash="dot",
                line_width=1.2,
                annotation_text="Random chance (0.50)",
                annotation_font_color=C["text3"],
                annotation_font_size=10,
                annotation_position="bottom right",
            )
            auc_fig.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=14, b=10),
                plot_bgcolor=C["panel"],
                paper_bgcolor=C["panel"],
                yaxis=dict(
                    range=[0, 1.08],
                    tickformat=".2f",
                    showgrid=True,
                    gridcolor=C["border"],
                    tickfont=dict(color=C["text2"], size=11),
                    title=dict(
                        text="ROC-AUC score",
                        font=dict(color=C["text2"], size=12),
                    ),
                ),
                xaxis=dict(tickfont=dict(color=C["text2"], size=11)),
            )
            st.plotly_chart(auc_fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render ROC-AUC chart: {exc}")
            with st.expander("Error traceback"):
                st.code(_traceback.format_exc())

    # ── Architecture table ────────────────────────────────────────────────────
    with col_b:
        st.markdown('<div class="sec-heading">Architecture Summary</div>',
                    unsafe_allow_html=True)
        arch_df = pd.DataFrame([
            ("Convolution type",     "GINEConv (edge-feature aware)"),
            ("Number of layers",     "5"),
            ("Hidden dimension",     "128"),
            ("Atom encoder",         "9 embeddings summed (OGB vocab)"),
            ("Bond encoder",         "3 embeddings summed (OGB vocab)"),
            ("Normalisation",        "BatchNorm1d after each layer"),
            ("Dropout (conv)",       "0.3 (training only)"),
            ("Graph pooling",        "global_add_pool"),
            ("Classifier head",      "Linear 128 → 64 → 1"),
            ("Dropout (classifier)", "0.5 (training only)"),
            ("Output",               "Sigmoid → probability ∈ [0, 1]"),
            ("Loss",                 "BCEWithLogitsLoss"),
            ("Optimiser",            "Adam · lr = 0.001"),
            ("Training epochs",      "50"),
            ("Batch size",           "64"),
            ("Dataset",              "ogbg-molhiv · 41,127 molecules"),
            ("Split",                "Scaffold split (OGB standard)"),
            ("Metric",               "ROC-AUC (OGB Evaluator)"),
        ], columns=["Component", "Detail"])
        st.dataframe(arch_df, use_container_width=True, hide_index=True, height=490)

    # ── Explanation ───────────────────────────────────────────────────────────
    st.markdown(
        f"""<div class="disclaimer-strip" style="margin-top:0.8rem">
          <strong>Understanding ROC-AUC:</strong>&nbsp;
          ROC-AUC (Receiver Operating Characteristic – Area Under Curve) measures
          how well the model <em>ranks</em> HIV-active molecules above inactive ones,
          across every possible decision threshold. A score of 1.0 means perfect
          ranking; 0.5 means no better than random chance. It is
          <strong>not</strong> the same as classification accuracy.
          The OGB-MolHIV dataset is highly imbalanced
          (roughly 1,500 actives out of 41,127 molecules, or ~3.6%), so
          ROC-AUC is a more informative metric than accuracy for this task.
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: About the Project ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About the Project":
    _page_header(
        "About the Project",
        "Final-year project — AI Drug Discovery Using Graph Neural Networks",
    )

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown(f"""
        <div class="content-card">
          <h3>Project Objective</h3>
          <p>
            The goal of this project is to apply Graph Neural Networks (GNNs)
            to the computational drug discovery task of predicting whether a
            molecule inhibits HIV replication, using only its chemical structure.
          </p>
          <p style="margin-top:0.7rem">
            Traditional drug screening requires expensive wet-lab experiments.
            Computational models trained on historical experimental data can
            rapidly triage millions of candidate molecules, helping chemists
            prioritise which compounds are worth synthesising and testing.
          </p>
        </div>

        <div class="content-card" style="margin-top:0.9rem">
          <h3>OGB-MolHIV Dataset</h3>
          <p>
            <strong style="color:{C['text']}">Source:</strong>
            HIV replication inhibition experiments from the Drug Therapeutics
            Program of the US National Cancer Institute.<br>
            <strong style="color:{C['text']}">Size:</strong> 41,127 molecules.<br>
            <strong style="color:{C['text']}">Labels:</strong> Binary —
            active (inhibits HIV, ~3.6%) or inactive (~96.4%).<br>
            <strong style="color:{C['text']}">Split strategy:</strong>
            Scaffold split — training, validation, and test sets contain
            structurally distinct chemical scaffolds, making generalisation
            to unseen chemical series genuinely challenging.<br>
            <strong style="color:{C['text']}">Evaluation:</strong>
            ROC-AUC via the OGB standardised evaluator, enabling direct
            comparison with published leaderboard results.
          </p>
        </div>

        <div class="content-card" style="margin-top:0.9rem">
          <h3>Molecules as Graphs</h3>
          <p>
            Each molecule is represented as an undirected graph:
          </p>
          <ul>
            <li>
              <strong style="color:{C['text']}">Atoms → nodes.</strong>
              Each node carries a 9-dimensional integer feature vector
              encoding atom type, chirality, degree, formal charge, number of
              hydrogens, number of radical electrons, hybridisation state,
              aromaticity, and ring membership — all using the OGB categorical
              encoding scheme.
            </li>
            <li style="margin-top:0.4rem">
              <strong style="color:{C['text']}">Bonds → edges.</strong>
              Each edge carries a 3-dimensional integer feature vector encoding
              bond type (single/double/triple/aromatic), bond stereo, and
              whether the bond is conjugated.
            </li>
          </ul>
          <p style="margin-top:0.5rem">
            No hand-crafted molecular fingerprints are used. The GNN learns
            which structural patterns are predictive directly from the graph.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="content-card">
          <h3>Why a Graph Neural Network?</h3>
          <p>
            Classical machine learning for molecules typically uses fixed-size
            fingerprint vectors that lose structural information. GNNs operate
            directly on the graph, so the model can learn arbitrary
            distance-dependent structural motifs without manual feature
            engineering.
          </p>
          <p style="margin-top:0.5rem">
            GINEConv (Graph Isomorphism Network with Edge features) is used
            here because it is theoretically as powerful as the
            Weisfeiler-Lehman graph isomorphism test, and it natively
            incorporates bond features at each message-passing step rather
            than ignoring them.
          </p>
        </div>

        <div class="content-card" style="margin-top:0.9rem">
          <h3>GINEConv Architecture</h3>
          <p>
            At each of the five message-passing layers, every atom node
            aggregates information from its bonded neighbours, weighted by the
            bond embedding:
          </p>
          <p style="margin-top:0.5rem;font-family:monospace;font-size:0.8rem;
                     color:{C['teal']};background:{C['panel2']};
                     padding:0.5rem 0.7rem;border-radius:6px;
                     border:1px solid {C['border']};">
            h_v = MLP((1+ε)·h_v + Σ_u ReLU(h_u + e_uv))
          </p>
          <p style="margin-top:0.6rem">
            After five such layers,
            <code style="color:{C['teal']}">global_add_pool</code>
            sums all node embeddings into a single 128-dimensional graph
            vector, which a two-layer classifier head maps to a scalar logit.
            Applying sigmoid gives a probability in [0, 1].
          </p>
          <p style="margin-top:0.5rem">
            <strong style="color:{C['text']}">Test ROC-AUC: {TEST_AUC}</strong>
            — the model ranks a randomly chosen active molecule above a
            randomly chosen inactive molecule approximately {TEST_AUC*100:.0f}%
            of the time.
          </p>
        </div>

        <div class="content-card" style="margin-top:0.9rem">
          <h3>Scientific Limitations and Disclaimer</h3>
          <p>
            Predictions from this model are
            <strong style="color:{C['inactive']}">computational estimates only</strong>.
            They are not experimental confirmation of drug efficacy, safety,
            toxicity, or clinical suitability. The model:
          </p>
          <ul>
            <li>Was trained on one specific HIV assay; it does not generalise to other diseases.</li>
            <li>May produce incorrect predictions for molecules structurally unlike the training set.</li>
            <li>Does not account for pharmacokinetics, metabolism, off-target effects, or dosing.</li>
          </ul>
          <p style="margin-top:0.5rem">
            This application is for <strong style="color:{C['text']}">academic
            demonstration only</strong>. Do not use it for any medical, clinical,
            or safety-critical purpose.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer (shown on all pages)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="footer-bar">'
    f"AI Drug Discovery Using Graph Neural Networks &nbsp;·&nbsp; Final-Year Project"
    f"&nbsp;·&nbsp; OGB-MolHIV &nbsp;·&nbsp; GINEConv × 5, hidden=128"
    f"&nbsp;·&nbsp; Val ROC-AUC {VAL_AUC} &nbsp;·&nbsp; Test ROC-AUC {TEST_AUC}"
    f"</div>",
    unsafe_allow_html=True,
)
