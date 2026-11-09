from sklearn.metrics import f1_score
from torch import nn
import time,copy, torch


def count_parameters(model):
    total =  sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def set_model(model, freeze_bn):
    model.train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def train_one_epoch(model, loader, optimizer, loss_fn, device, freeze_bn=False):
    set_model(model, freeze_bn)
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        total_loss += loss.item() * xb.size(0)

        batch_preds = logits.argmax(dim=1)

        preds.extend(batch_preds.cpu().tolist())
        labels.extend(yb.cpu().tolist())

    if len(labels) == 0:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)

    return {
        "loss": total_loss / len(labels),
        "accuracy": acc,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }

def sync(device):
    if device == "cuda": torch.cuda.synchronize()
    elif device == "mps": torch.mps.synchronize()

def fit(model, train_loader, val_loader, optimizer, loss_fn, device, epochs,freeze_bn = False, name = "", scheduler=None):
    history, best_val, best_state, train_time = [], float("inf"), None, 0.0
    for epoch in range(1, epochs +1):
        t0=time.perf_counter()
        train_loss = train_one_epoch(model , train_loader, optimizer, loss_fn, device, freeze_bn)
        sync(device)
        train_time += time.perf_counter() - t0
        val = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        print (f"{name} epoch {epoch}/{epochs} train_loss = {train_loss:.3f} "
               f"val_loss = {val['loss']:.3f} val_accuracy = {val['accuracy']:.3f} val_macro_f1 = {val['macro_f1']:.3f}")
        if val["loss"]< best_val:
            best_val, best_state = val["loss"], copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()  # once per epoch, after validation
    model.load_state_dict(best_state)
    return history, train_time

@torch.no_grad()
def inference_ms_per_image(model, loader, device, batches=10):
    model.eval()
    data = []
    for i, (xb, _) in enumerate(loader):
        if i == batches: break
        data.append(xb.to(device))
    for _ in range(3): model(data[0])
    sync(device)
    t0 = time.perf_counter()
    for xb in data: model(xb)
    sync(device)
    n = sum(len(xb) for xb in data)
    return (time.perf_counter() - t0) / n * 1000


