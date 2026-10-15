
import torch
from ogb.graphproppred import PygGraphPropPredDataset

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nDownloading molecular dataset...")

dataset = PygGraphPropPredDataset(
    name="ogbg-molhiv",
    root="data"
)

splits = dataset.get_idx_split()

print("\nDataset downloaded successfully")
print("Total molecules:", len(dataset))
print("Training molecules:", len(splits["train"]))
print("Validation molecules:", len(splits["valid"]))
print("Test molecules:", len(splits["test"]))

molecule = dataset[0]

print("\nFirst molecular graph:")
print("Atom features:", molecule.x.shape)
print("Bond connections:", molecule.edge_index.shape)
print("Target label:", molecule.y)
