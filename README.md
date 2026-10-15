# AI Drug Discovery Using Graph Neural Networks

A final-year project that trains a Graph Neural Network (GNN) on the
[OGB-MolHIV](https://ogb.stanford.edu/docs/graphprop/) benchmark dataset and
exposes a Streamlit web application for molecular graph visualisation and
HIV-activity prediction.

---

## Project structure

```
drug-discovery-gnn/
├── train.py            # Model training script (do not modify)
├── download_data.py    # Dataset download helper
├── model.py            # GNN architecture (mirrors train.py exactly)
├── inference.py        # Checkpoint loading + SMILES → prediction pipeline
├── molecule_utils.py   # RDKit utilities, Plotly graph builder, property calc
├── app.py              # Streamlit web application
├── best_gnn_model.pt   # Trained model checkpoint
├── requirements.txt    # Python dependencies
├── data/               # OGB-MolHIV dataset (auto-downloaded)
└── venv/               # Python virtual environment
```

---

## Model performance

| Metric          | Value  |
|-----------------|--------|
| Val ROC-AUC     | 0.8266 |
| Test ROC-AUC    | 0.7906 |
| Architecture    | 5-layer GINEConv |
| Hidden dim      | 128    |
| Pooling         | global_add_pool |
| Dataset         | ogbg-molhiv (41,127 molecules) |

---

## Prerequisites

- **Linux** (tested on Ubuntu 22.04+)
- **Python 3.10–3.14** (project uses Python 3.14)
- **NVIDIA GPU** with CUDA (recommended; CPU inference works too)
- The virtual environment `venv/` should already contain all packages if
  you trained the model. If starting fresh, follow the setup steps below.

---

## Setup (fresh install)

### 1. Clone / navigate to the project

```bash
cd ~/Desktop/Projects/drug-discovery-gnn
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch (match your CUDA version)

Visit https://pytorch.org/get-started/locally/ for the exact command.
Example for CUDA 12.1:

```bash
pip install torch==2.14.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install PyTorch Geometric

```bash
pip install torch-geometric==2.8.1
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit application

```bash
# Activate the virtual environment first
source venv/bin/activate

# Launch the app (opens in your browser at http://localhost:8501)
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

---

## Training the model from scratch (optional)

> The checkpoint `best_gnn_model.pt` already exists. Only run this if you
> want to retrain.

```bash
source venv/bin/activate

# Download the dataset (only needed once)
python download_data.py

# Train (50 epochs, ~30 min on a modern GPU)
python train.py
```

---

## Application features

| Feature | Description |
|---------|-------------|
| SMILES input | Text area with live validation via RDKit |
| Example selector | 10 pre-loaded drug molecules |
| 2D structure | RDKit SVG rendering |
| Molecular graph | Interactive Plotly graph — hover for atom features |
| Graph statistics | Node/edge count, degree, element/bond breakdown |
| Molecular properties | MW, LogP, HBD/HBA, TPSA, Lipinski Ro5 |
| GNN prediction | HIV-activity probability + confidence badge |
| Atom feature table | Raw OGB 9-dim node features fed to the GNN |
| Bond feature table | Raw OGB 3-dim edge features fed to the GNN |

---

## Disclaimer

> This application is for **academic demonstration only**. Predictions are
> computational outputs of a machine learning model trained on the
> OGB-MolHIV benchmark. They are **not** confirmed drug activities, medical
> diagnoses, or clinical recommendations.
