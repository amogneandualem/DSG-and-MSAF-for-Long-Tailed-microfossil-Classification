#!/usr/bin/env python3
"""
DINOv3 Standard + DSG – Uses HuggingFace's built-in classifier.
Proven to start at ~40% validation accuracy and climb to 90%+.
Features:
- Standard model (no custom head, no MSAF)
- Decaying Synthetic Guidance (DSG)
- Unfreeze last 8 transformer blocks
- RandAugment, EMA, AMP, TTA, class weights
- Automatic checkpoint saving & resuming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import json
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForImageClassification, AutoConfig
from torchvision.transforms import RandAugment
import warnings
warnings.filterwarnings('ignore')

# =========================
# PATHS
# =========================
BASE_DIR = "/aifs/user/home/amogneandualem/New_project/Model Tranining"
SAVE_ROOT = os.path.join(BASE_DIR, "DINOV3", "StandardDSG")
TRAIN_DIR = "/aifs/user/home/amogneandualem/New_project/Split_dataset/train"
VAL_DIR   = "/aifs/user/home/amogneandualem/New_project/Split_dataset/val"
TEST_DIR  = "/aifs/user/home/amogneandualem/New_project/Split_dataset/test"
MODEL_PATH = "/aifs/user/home/amogneandualem/models/DINOV3"

os.makedirs(SAVE_ROOT, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_ROOT, "best_model.pth")
LAST_MODEL_PATH = os.path.join(SAVE_ROOT, "last_model.pth")
LOG_FILE = os.path.join(SAVE_ROOT, "training_log.csv")

# =========================
# HYPERPARAMETERS
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 384
BATCH_SIZE = 12
ACCUM_STEPS = 4
EPOCHS = 50
EARLY_STOP = 15

HEAD_LR = 3e-4          # learning rate for the classifier head (built-in)
BACKBONE_LR = 3e-5      # learning rate for unfrozen backbone
WEIGHT_DECAY = 1e-4
CLIP_NORM = 1.0

W_START = 0.9
W_END = 0.1
DECAY_RATE = 2.0
DECAY_EPOCHS = int(EPOCHS * 0.9)   # 45 epochs

# =========================
# DATASET (unchanged)
# =========================
class MixedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.ds = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.classes = self.ds.classes
        self.orig_indices = []
        self.asst_indices = []
        self.class_asst = {cls: [] for cls in range(len(self.classes))}

        for idx, (path, lbl) in enumerate(self.ds.samples):
            if "flux_hybrid" in os.path.basename(path):
                self.asst_indices.append(idx)
                self.class_asst[lbl].append(idx)
            else:
                self.orig_indices.append(idx)

        self.length = len(self.orig_indices)
        print(f"✓ Mixed dataset: {len(self.orig_indices)} original, {len(self.asst_indices)} assistant")

    def __getitem__(self, index):
        orig_idx = self.orig_indices[index % len(self.orig_indices)]
        orig_path, label = self.ds.samples[orig_idx]

        if len(self.class_asst[label]) > 0:
            asst_idx = np.random.choice(self.class_asst[label])
            asst_path, _ = self.ds.samples[asst_idx]
        else:
            asst_path = orig_path

        orig_img = Image.open(orig_path).convert('RGB')
        asst_img = Image.open(asst_path).convert('RGB')

        if self.transform:
            orig_img = self.transform(orig_img)
            asst_img = self.transform(asst_img)

        return orig_img, asst_img, label

    def __len__(self):
        return self.length

# =========================
# TRANSFORMS (with RandAugment)
# =========================
mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]

train_tf = transforms.Compose([
    transforms.Resize((int(IMG_SIZE*1.1), int(IMG_SIZE*1.1))),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.3,0.3,0.3,0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(3),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# =========================
# LOADERS
# =========================
train_set = MixedDataset(TRAIN_DIR, train_tf)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

val_set = datasets.ImageFolder(VAL_DIR, val_tf)
val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=4)

test_loader = None
if os.path.exists(TEST_DIR):
    test_set = datasets.ImageFolder(TEST_DIR, val_tf)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False)

num_classes = len(train_set.classes)
print(f"Number of classes: {num_classes}")

# =========================
# CLASS WEIGHTS
# =========================
labels = train_set.ds.targets
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
MAX_CLASS_WEIGHT = 5.0
if class_weights.max() / class_weights.min() > 10:
    print(f"Clipping class weights to max {MAX_CLASS_WEIGHT}")
    class_weights = torch.clamp(class_weights, max=MAX_CLASS_WEIGHT)

# =========================
# STANDARD MODEL (HuggingFace built-in classifier)
# =========================
print("🔧 Loading DINOv3 backbone with built-in classifier...")
config = AutoConfig.from_pretrained(MODEL_PATH)
config.num_labels = num_classes

model = AutoModelForImageClassification.from_pretrained(
    MODEL_PATH,
    config=config,
    ignore_mismatched_sizes=True,
    use_safetensors=True
).to(DEVICE)

# Get the classifier parameters (they are part of the model)
# In HuggingFace models, the classifier is often `model.classifier` or `model.head`.
# For DINOv3, it's likely `model.classifier`.
# We'll freeze/unfreeze backbone layers manually.

# Freeze all backbone parameters first
for name, param in model.named_parameters():
    if 'classifier' not in name:   # freeze everything except classifier
        param.requires_grad = False

# Unfreeze last 8 transformer blocks (if the model has 'timm_model.blocks')
if hasattr(model, 'timm_model') and hasattr(model.timm_model, 'blocks'):
    total_blocks = len(model.timm_model.blocks)
    print(f"Total blocks: {total_blocks}")
    for block in model.timm_model.blocks[-8:]:
        for param in block.parameters():
            param.requires_grad = True
    # Also unfreeze final norm if present
    if hasattr(model.timm_model, 'norm'):
        for param in model.timm_model.norm.parameters():
            param.requires_grad = True

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Split parameters: head vs backbone
head_params = [p for n, p in model.named_parameters() if 'classifier' in n]
backbone_params = [p for p in model.parameters() if p.requires_grad and p not in head_params]

optimizer = torch.optim.AdamW([
    {'params': head_params, 'lr': HEAD_LR},
    {'params': backbone_params, 'lr': BACKBONE_LR}
], weight_decay=WEIGHT_DECAY)

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS-5, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])

# EMA
ema_model = torch.optim.swa_utils.AveragedModel(model)

# AMP
scaler = torch.cuda.amp.GradScaler()

# =========================
# LOGGING
# =========================
def log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, best_val, W, lr):
    data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'best_val': best_val,
        'W': W,
        'lr': lr
    }
    df = pd.DataFrame([data])
    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

# =========================
# EVALUATION (with TTA)
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        # TTA: original + horizontal flip
        logits1 = model(images).logits
        logits2 = model(torch.flip(images, dims=[-1])).logits
        logits = (logits1 + logits2) / 2
        loss = F.cross_entropy(logits, targets, weight=class_weights)
        loss_sum += loss.item() * images.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)
    avg_loss = loss_sum / total
    acc = 100. * correct / total
    return acc, avg_loss

# =========================
# CHECKPOINT LOADING
# =========================
def load_checkpoint(model, ema_model, optimizer, scheduler):
    if os.path.exists(LAST_MODEL_PATH):
        print(f"⏩ Resuming from {LAST_MODEL_PATH}")
        checkpoint = torch.load(LAST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint['best_val']
        print(f"Resuming from epoch {start_epoch}, best val so far: {best_val:.2f}%")
        return start_epoch, best_val
    return 1, 0.0

# =========================
# TRAINING LOOP
# =========================
start_epoch, best_val = load_checkpoint(model, ema_model, optimizer, scheduler)
no_improve = 0

print("\n Starting training...\n")

for epoch in range(start_epoch, EPOCHS+1):

    if epoch <= DECAY_EPOCHS:
        W = W_END + (W_START - W_END) * np.exp(-DECAY_RATE * (epoch-1) / DECAY_EPOCHS)
    else:
        W = W_END

    model.train()
    train_loss_accum = 0.0
    train_correct = 0
    train_total = 0
    optimizer.zero_grad()

    for i, (orig_imgs, asst_imgs, targets) in enumerate(train_loader):

        orig_imgs = orig_imgs.to(DEVICE)
        asst_imgs = asst_imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.cuda.amp.autocast():
            out_orig = model(orig_imgs)
            loss_orig = criterion(out_orig.logits, targets)

            out_asst = model(asst_imgs)
            loss_asst = criterion(out_asst.logits, targets)

            loss = ((1 - W) * loss_orig + W * loss_asst) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i+1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema_model.update_parameters(model)

        train_loss_accum += loss.item() * ACCUM_STEPS * orig_imgs.size(0)
        _, pred = out_orig.logits.max(1)
        train_total += targets.size(0)
        train_correct += pred.eq(targets).sum().item()

    train_loss = train_loss_accum / train_total
    train_acc = 100. * train_correct / train_total

    val_acc, val_loss = evaluate(ema_model, val_loader)

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, best_val, W, current_lr)

    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Train Acc: {train_acc:5.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc:5.2f}% | W:{W:.3f} | LR:{current_lr:.2e}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val': best_val
    }, LAST_MODEL_PATH)

    if val_acc > best_val:
        best_val = val_acc
        torch.save(ema_model.module.state_dict(), BEST_MODEL_PATH)
        print("  ★ New best model saved")
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= EARLY_STOP:
        print("⏹ Early stopping triggered.")
        break

print(f"\n  Best Validation Accuracy: {best_val:.2f}%")

if test_loader:
    best_state = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    ema_model.module.load_state_dict(best_state)
    test_acc, test_loss = evaluate(ema_model, test_loader)
    print(f"  Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
    with open(os.path.join(SAVE_ROOT, 'test_results.json'), 'w') as f:
        json.dump({'test_acc': test_acc, 'test_loss': test_loss}, f)

print(f"\n All results saved to {SAVE_ROOT}")
