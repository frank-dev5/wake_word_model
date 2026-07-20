import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np

# Import your prepared classes
from spectrogram_generator import WakeWordDataset
from model import WakeWordCNN


# --- Configuration ---
POS_DIR = r"D:\model\assistant\src\wake_word_project\data_standardized\positive"
NEG_DIR = r"D:\model\assistant\src\wake_word_project\data_standardized\negative"
BG_DIR  = r"D:\model\assistant\src\wake_word_project\data_standardized\background"
CHECKPOINT_PATH = "wake_word_model2.pth"

# Build full dataset
dataset = WakeWordDataset(POS_DIR, NEG_DIR, BG_DIR, max_negatives=None)

# 1. Stratified Dataset Split
targets = [s[1] for s in dataset.samples]
train_idx, val_idx = train_test_split(
    np.arange(len(targets)), test_size=0.15, shuffle=True, stratify=targets, random_state=42
)
train_subset = Subset(dataset, train_idx)
val_subset = Subset(dataset, val_idx)

# 2. Weighted Sampler (Oversamples the rare positive clips)
train_targets = [targets[i] for i in train_idx]
class_counts = np.bincount(train_targets)  # [NegCount, PosCount]

class_weights = 1. / (class_counts + 1e-9)
sample_weights = torch.from_numpy(np.array([class_weights[t] for t in train_targets])).double()
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# 3. DataLoaders
train_loader = DataLoader(train_subset, batch_size=64, sampler=sampler)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

# Confirm split
print(f"Total dataset size: {len(dataset)}")
print(f"Train size: {len(train_subset)} | Val size: {len(val_subset)}")
print(f"Batches per epoch (train): {len(train_loader)}")
print(f"Class counts (train): {class_counts}")

# 4. Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WakeWordCNN().to(device)

pos_weight_val = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. Load checkpoint (model + optimizer) if one exists, so training continues instead of restarting
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    if "model" in ckpt and "optimizer" in ckpt:
        # New format: dict containing both model and optimizer state
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded existing checkpoint (new format) from {CHECKPOINT_PATH}, continuing training...")
    else:
        # Old format: file IS the raw model state_dict, no optimizer state saved
        model.load_state_dict(ckpt)
        print(f"Loaded existing checkpoint (old format) from {CHECKPOINT_PATH}.")
        print("No optimizer state found in old checkpoint — optimizer starts fresh. Continuing training...")
else:
    print("No existing checkpoint found, training from scratch...")

def validate():
    model.eval()
    val_loss, tp, tn, fp, fn = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device).float().reshape(-1, 1)
            logits = model(features)
            val_loss += loss_fn(logits, labels).item()

            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()

            tp += ((predictions == 1) & (labels == 1)).sum().item()
            tn += ((predictions == 0) & (labels == 0)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = 100 * (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return val_loss / len(val_loader), accuracy, recall


def train_model(epochs=20):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device).float().reshape(-1, 1)
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_acc, val_recall = validate()
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Recall: {val_recall:.3f}")

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, CHECKPOINT_PATH)
    print(f"Training Complete! Model saved as '{CHECKPOINT_PATH}'")


if __name__ == "__main__":
    train_model(epochs=20)