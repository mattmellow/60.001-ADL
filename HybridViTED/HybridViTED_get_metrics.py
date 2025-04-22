import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.config import load_config
from model.genconvit_ed import GenConViTED
from dataset.loader import load_data
import os

# Load config and set device
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the model
model = GenConViTED(config)
checkpoint_path = "weight/genconvit_ed_Apr_15_2025_08_31_02.pth"  # <- replace with actual .pth file
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

# Load test data
dataloaders, dataset_sizes = load_data("kaggle_train_data", batch_size=32)  # adjust path and batch size if needed

# Evaluate
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in dataloaders["test"]:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs).float()
        _, preds = torch.max(output, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")