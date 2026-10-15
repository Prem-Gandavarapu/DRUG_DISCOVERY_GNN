
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

ROOT = "data"
BATCH_SIZE = 64
EPOCHS = 50
HIDDEN = 128
LAYERS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = PygGraphPropPredDataset(
    name="ogbg-molhiv",
    root=ROOT
)

split = dataset.get_idx_split()
train_loader = DataLoader(
    dataset[split["train"]], batch_size=BATCH_SIZE, shuffle=True
)
valid_loader = DataLoader(
    dataset[split["valid"]], batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    dataset[split["test"]], batch_size=BATCH_SIZE, shuffle=False
)

atom_dims = get_atom_feature_dims()
bond_dims = get_bond_feature_dims()


class AtomEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden) for dim in atom_dims
        ])

    def forward(self, x):
        result = 0
        for i, embedding in enumerate(self.embeddings):
            result = result + embedding(x[:, i])
        return result


class BondEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden) for dim in bond_dims
        ])

    def forward(self, edge_attr):
        result = 0
        for i, embedding in enumerate(self.embeddings):
            result = result + embedding(edge_attr[:, i])
        return result


class GNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.atom_encoder = AtomEncoder(HIDDEN)
        self.bond_encoder = BondEncoder(HIDDEN)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(LAYERS):
            mlp = nn.Sequential(
                nn.Linear(HIDDEN, HIDDEN),
                nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN)
            )
            self.convs.append(
                GINEConv(mlp, edge_dim=HIDDEN)
            )
            self.norms.append(nn.BatchNorm1d(HIDDEN))

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN // 2, 1)
        )

    def forward(self, data):
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        graph_features = global_add_pool(x, data.batch)
        return self.classifier(graph_features)


model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
evaluator = Evaluator(name="ogbg-molhiv")


def evaluate(loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)

            predictions.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(labels, axis=0)

    return evaluator.eval({
        "y_true": y_true,
        "y_pred": y_pred
    })["rocauc"]


best_auc = -1
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch)
        loss = criterion(logits, batch.y.float().view(-1, 1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    train_loss = total_loss / len(train_loader.dataset)
    val_auc = evaluate(valid_loader)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Validation ROC-AUC: {val_auc:.4f}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "best_gnn_model.pt")

model.load_state_dict(best_state)
test_auc = evaluate(test_loader)

print("\nTraining complete.")
print(f"Best validation ROC-AUC: {best_auc:.4f}")
print(f"Test ROC-AUC: {test_auc:.4f}")
print("Saved model: best_gnn_model.pt")
