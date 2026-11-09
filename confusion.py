"""Retrain the models (no augmentation, full training set) and save test-set
predictions, confusion matrices and the list of misclassified images.

    python confusion.py            # scratch CNN + ResNet18 (frozen, then fine-tuned)
    python confusion.py resnet     # only the ResNet18 models
    python confusion.py scratch    # only the scratch CNN
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from data import load_data, eval_tf
from engine import fit
from models import SmallCNN, build_resnet_frozen, unfreeze_layer4

seed = 28
epochs, ft_epochs = 10, 5
TAG = "noaug_100"


def find_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for xb, _ in loader:
        preds += model(xb.to(device)).argmax(1).cpu().tolist()
    return preds


def report(name, model, test_loader, classes, device, out):
    ds = test_loader.dataset  # Subset of an ImageFolder
    labels = [ds.dataset.targets[i] for i in ds.indices]
    files = [os.path.basename(ds.dataset.samples[i][0]) for i in ds.indices]
    preds = predict(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    n = len(classes)
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    print(f"\n{name}: test accuracy {acc:.4f}, macro-F1 {f1:.4f}")

    pairs = sorted(((cm[i, j], classes[i], classes[j])
                    for i in range(n) for j in range(n) if i != j and cm[i, j]), reverse=True)
    print("most common confusions (count, true -> predicted):")
    for c, t, p in pairs[:8]:
        print(f"  {c:>2}  {t} -> {p}")

    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=60, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted species"); ax.set_ylabel("True species")
    ax.set_title(f"{name}: test accuracy {acc:.1%}")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    fig.tight_layout()
    fig.savefig(f"confusion_{name.replace('+', '_')}.png", dpi=150)
    plt.close(fig)

    out[name] = {
        "test_accuracy": acc, "test_macro_f1": f1, "confusion_matrix": cm.tolist(),
        "misclassified": [{"file": f, "true": classes[l], "pred": classes[p]}
                          for f, l, p in zip(files, labels, preds) if l != p],
    }


def main():
    which = set(sys.argv[1:]) or {"scratch", "resnet"}
    device = find_device()
    print("device:", device)
    train_loader, val_loader, test_loader, classes = load_data(eval_tf, 1.0)
    n = len(classes)
    loss_fn = nn.CrossEntropyLoss()
    out = {"classes": classes}

    if "scratch" in which:
        torch.manual_seed(seed)
        m = SmallCNN(n).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        fit(m, train_loader, val_loader, opt, loss_fn, device, epochs, scheduler=sched, name="SmallCNN")
        report("SmallCNN", m, test_loader, classes, device, out)

    if "resnet" in which:
        torch.manual_seed(seed)
        m = build_resnet_frozen(n).to(device)
        opt = torch.optim.Adam(m.fc.parameters(), lr=1e-3)
        fit(m, train_loader, val_loader, opt, loss_fn, device, epochs, freeze_bn=True, name="ResNet18_Frozen")
        report("ResNet18_Frozen", m, test_loader, classes, device, out)
        opt = unfreeze_layer4(m)
        fit(m, train_loader, val_loader, opt, loss_fn, device, ft_epochs, freeze_bn=True,
            name="ResNet18_Unfrozen+finetune")
        report("ResNet18_Unfrozen+finetune", m, test_loader, classes, device, out)

    with open(f"predictions_{TAG}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved predictions_{TAG}.json and the confusion_*.png files")


if __name__ == "__main__":
    main()
