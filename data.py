from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
DATA = "BD-FISH15-CLEAN-DATASET"

seed = 28
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats

eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

augmentation = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8,1)) ,transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),transforms.ToTensor(), transforms.Normalize(mean, std)])

def load_data(train_tf=augmentation, train_fraction=1.0):
    train_data = datasets.ImageFolder(DATA , transform=train_tf)
    val_data = datasets.ImageFolder(DATA , transform=eval_tf)
    targets = val_data.targets

    idx = list(range(len(targets)))
    train_idx, hold_idx = train_test_split(idx, test_size=0.3, random_state=seed)
    val_idx, test_idx = train_test_split(hold_idx, test_size=0.5, stratify=[targets[i] for i in hold_idx],random_state=seed)

    # Optionally shrink the training set (val and test stay unchanged).
    if train_fraction < 1.0:
        train_idx, _ = train_test_split(train_idx, train_size=train_fraction,
                                        stratify=[targets[i] for i in train_idx], random_state=seed)

    train_loader = DataLoader(Subset(train_data, train_idx), batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(val_data, val_idx), batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(Subset(val_data, test_idx), batch_size=64, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, val_data.classes

if __name__ == "__main__":
    tr, va, te, classes = load_data()
    print(len(classes), "classes: ", classes )
    print("train/val/test: ", len(tr.dataset), len(va.dataset), len(te.dataset))


