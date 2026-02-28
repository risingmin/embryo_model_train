import json
import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as T
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data"
NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_CLASSES = 2                       # 0 = Correct, 1 = Incorrect
CLASS_NAMES = ["Correct", "Incorrect"]
MODEL_SAVE_PATH = "embryo_classifier.pth"
LOSS_CSV_PATH = "training_loss.csv"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Pretrained weights & transforms (with colour inversion for microscope)
# ---------------------------------------------------------------------------
weights = models.EfficientNet_V2_S_Weights.DEFAULT

train_tfm = T.Compose([
    T.Resize((300, 300)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomInvert(p=1.0),            # always invert – matches lab microscope
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

val_tfm = T.Compose([
    T.Resize((300, 300)),
    T.RandomInvert(p=1.0),            # always invert – matches lab microscope
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Dataset – crops individual embryos from COCO-annotated images
# ---------------------------------------------------------------------------
class EmbryoCropDataset(Dataset):
    """
    Reads a COCO-format annotation file and yields one *crop* per
    annotated embryo.  Category mapping:
        COCO id 1 ("correct stage")   → label 0  (Correct)
        COCO id 2 ("incorrect stage") → label 1  (Incorrect)
    """

    def __init__(self, img_dir: str, annotation_file: str, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform

        with open(annotation_file) as f:
            coco = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

        self.samples = []  # list of (image_path, bbox, label)
        for ann in coco["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in (1, 2):
                continue                           # skip the parent category
            label = 0 if cat_id == 1 else 1         # Correct=0, Incorrect=1
            img_path = os.path.join(img_dir, id_to_file[ann["image_id"]])
            bbox = ann["bbox"]                      # [x, y, w, h]
            self.samples.append((img_path, bbox, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        x, y, w, h = bbox
        crop = img.crop((x, y, x + w, y + h))

        if self.transform:
            crop = self.transform(crop)

        return crop, label

# ---------------------------------------------------------------------------
# Create datasets & dataloaders
# ---------------------------------------------------------------------------
train_ds = EmbryoCropDataset(
    img_dir=os.path.join(DATA_DIR, "train"),
    annotation_file=os.path.join(DATA_DIR, "train", "_annotations.coco.json"),
    transform=train_tfm,
)
val_ds = EmbryoCropDataset(
    img_dir=os.path.join(DATA_DIR, "valid"),
    annotation_file=os.path.join(DATA_DIR, "valid", "_annotations.coco.json"),
    transform=val_tfm,
)

print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = models.efficientnet_v2_s(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------------------------------------------------------
# Training loop with CSV loss logging
# ---------------------------------------------------------------------------
loss_log = []  # will also be written to CSV

for epoch in range(1, NUM_EPOCHS + 1):
    # ---- train ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- validate ----
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch}/{NUM_EPOCHS}  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

    loss_log.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
    })

# ---------------------------------------------------------------------------
# Save loss log to CSV
# ---------------------------------------------------------------------------
with open(LOSS_CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
    writer.writeheader()
    writer.writerows(loss_log)

print(f"\nLoss log saved to {LOSS_CSV_PATH}")

# ---------------------------------------------------------------------------
# Save trained model
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
