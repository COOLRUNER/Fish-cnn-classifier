import json
import sys

import torch
from torch import nn

from engine import count_parameters, evaluate, inference_ms_per_image, fit
from models import SmallCNN, build_resnet_frozen, unfreeze_layer4
from data import load_data, eval_tf, augmentation

seed = 28
smoke = False
epochs = 1 if smoke else 10
ft_epochs = 1 if smoke else 5

CONFIGS = {
    "aug_100": (augmentation, 1.0),
    "noaug_100": (eval_tf, 1.0),
    "noaug_25": (eval_tf, 0.25),
    "noaug_50": (eval_tf, 0.50),
}

def find_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def summarize(name, model, history, train_time, test_loader, loss_fn, device):
    total, trainable = count_parameters(model)
    best = min(history, key=lambda x: x["val_loss"])
    test = evaluate(model, test_loader, loss_fn, device)
    return {
        "model": name, "total_params": total, "trainable_params": trainable,
        "train_time_s": train_time, "best_val_acc": best["val_accuracy"],
        "test_acc": test["accuracy"], "test_macro_f1": test["macro_f1"],
        "inference_ms_per_img": inference_ms_per_image(model, test_loader, device),
    }

def main(tag, train_tf, frac):
    device = find_device()
    print("device: ", device)
    train_loader, val_loader, test_loader, classes = load_data(train_tf, frac)
    print(f"run: {tag} | training images: {len(train_loader.dataset)}")
    n = len(classes)
    loss_fn = nn.CrossEntropyLoss()
    results, histories = [], {}

    #Model 1 my CNN
    torch.manual_seed(seed)
    m1 = SmallCNN(n).to(device)
    opt = torch.optim.Adam(m1.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    h1, t1 = fit(m1, train_loader, val_loader, opt, loss_fn, device, epochs, scheduler=sched, name="SmallCNN")
    results.append(summarize("SmallCNN", m1, h1, t1, test_loader, loss_fn, device))
    histories["SmallCNN"] = h1

    #Model 2 pretrained ResNet18, frozen backbone, trainable head
    torch.manual_seed(seed)
    m2 = build_resnet_frozen(n).to(device)
    opt = torch.optim.Adam(m2.fc.parameters(), lr=1e-3)
    h2, t2 = fit(m2, train_loader, val_loader, opt, loss_fn, device, epochs, freeze_bn=True, name = "ResNet18_Frozen")
    results.append(summarize("ResNet18_Frozen", m2, h2, t2, test_loader, loss_fn, device))
    histories["ResNet18_Frozen"] = h2

    #Model 3 model 2, unfrozen layer 4 qnd fine tune
    opt = unfreeze_layer4(m2)
    h3, t3 = fit(m2, train_loader, val_loader, opt, loss_fn, device, ft_epochs, freeze_bn=True, name = "ResNet18_Unfrozen+finetune")
    r3 = summarize("ResNet18_Unfrozen+finetune", m2, h3,t2+ t3, test_loader, loss_fn, device)
    r3["head_time_s"], r3["finetune_time_s"] =t2, t3
    results.append(r3)
    histories["ResNet18_Unfrozen+finetune"] = h3

    print(f"\n{'model':<20}{'trainable':>12}{'train s':>9}{'val acc':>9}"
          f"{'test acc':>10}{'test F1':>9}{'ms/img':>8}")
    for r in results:
        print(f"{r['model']:<20}{r['trainable_params']:>12,}{r['train_time_s']:>9.0f}"
              f"{r['best_val_acc']:>9.3f}{r['test_acc']:>10.3f}"
              f"{r['test_macro_f1']:>9.3f}{r['inference_ms_per_img']:>8.2f}")

    with open(f"results_{tag}.json", "w") as f:
        json.dump({"results": results, "histories": histories, "classes": classes}, f, indent=2)

if __name__ == "__main__":
    tags = sys.argv[1:] or ["noaug_100"]
    for tag in tags:
        main(tag, *CONFIGS[tag])


