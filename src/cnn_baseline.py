"""
CNN Baseline for Stop Sign Classification
Adapted from CVproject_cnn.ipynb
"""

import os
import random
import json
import math
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from time import time
import logging

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import albumentations as A

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed=123):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def conv_bn_relu(cin, cout, k=3, s=1, p=1):
    """Convolutional block with BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True)
    )


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions."""
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class SmallCNNPlus(nn.Module):

    def __init__(self, dropout=0.30, in_ch=3, widths=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = widths
        
        # Stage A
        self.stem = conv_bn_relu(in_ch, c1, 3, 1, 1)
        self.resA1 = ResidualBlock(c1)
        self.resA2 = ResidualBlock(c1)
        self.poolA = nn.MaxPool2d(2)
        
        # Stage B
        self.convAtoB = conv_bn_relu(c1, c2, 3, 1, 1)
        self.resB1 = ResidualBlock(c2)
        self.resB2 = ResidualBlock(c2)
        self.poolB = nn.MaxPool2d(2)
        
        # Stage C
        self.convBtoC = conv_bn_relu(c2, c3, 3, 1, 1)
        self.resC1 = ResidualBlock(c3)
        self.resC2 = ResidualBlock(c3)
        self.poolC = nn.MaxPool2d(2)
        
        # Classifier head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, 1)
    
    def forward(self, x):
        # Stage A
        x = self.stem(x)
        x = self.resA1(x)
        x = self.resA2(x)
        x = self.poolA(x)
        
        # Stage B
        x = self.convAtoB(x)
        x = self.resB1(x)
        x = self.resB2(x)
        x = self.poolB(x)
        
        # Stage C
        x = self.convBtoC(x)
        x = self.resC1(x)
        x = self.resC2(x)
        x = self.poolC(x)
        
        # Classifier
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x).squeeze(-1)
        return logits

def build_train_aug(size=224):
    
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.0, 0.05),
            rotate=(-15, 15),
            p=0.7
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_aug(size=224):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def simple_preprocess(img, size=224):
    # Resize maintaining aspect ratio
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded canvas
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Normalize to [0, 1]
    canvas = canvas.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    for i in range(3):
        canvas[:, :, i] = (canvas[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
    
    return canvas


class ChipDataset(Dataset):
    def __init__(self, root_dir, transform, input_size=224, split='train'):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.input_size = input_size
        self.samples = []
        
        # Load stop signs
        stop_dir = self.root_dir / 'stop'
        if stop_dir.exists():
            for img_path in stop_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 1))
        
        # Load background/non-stop
        bg_dir = self.root_dir / 'bg'
        if bg_dir.exists():
            for img_path in bg_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 0))
        
        logging.info(f"{split.upper()} set: {len(self.samples)} chips loaded")
        
        # Count class distribution
        pos_count = sum(1 for _, label in self.samples if label == 1)
        neg_count = len(self.samples) - pos_count
        logging.info(f"  Stop signs: {pos_count}, Background: {neg_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation or simple preprocessing
        if self.transform is not None:
            aug = self.transform(image=img)
            img_processed = aug["image"].transpose(2, 0, 1)
        else:
            img_processed = simple_preprocess(img, self.input_size).transpose(2, 0, 1)
        
        img_tensor = torch.from_numpy(img_processed).float()
        
        return img_tensor, torch.tensor(label, dtype=torch.float32)

@torch.no_grad()
def compute_metrics(logits, targets):
    probs = torch.sigmoid(logits).cpu().numpy()
    y = targets.cpu().numpy()
    
    try:
        roc = roc_auc_score(y, probs)
    except Exception:
        roc = float('nan')
    
    try:
        pr_auc = average_precision_score(y, probs)
    except Exception:
        pr_auc = float('nan')
    
    preds = (probs >= 0.5).astype(np.uint8)
    try:
        f1 = f1_score(y, preds)
    except Exception:
        f1 = float('nan')
    
    return dict(roc_auc=roc, pr_auc=pr_auc, f1=f1)


def run_one_epoch(model, loader, criterion, optimizer=None, device='cuda', amp=True):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    all_logits, all_targets = [], []
    running_loss = 0.0
    scaler = torch.amp.GradScaler('cuda', enabled=amp and device.type == 'cuda')
    
    for images, targets in tqdm(loader, desc='Train' if is_train else 'Val', disable=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
                logits = model(images)
                loss = criterion(logits, targets)
        
        running_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
    
    avg_loss = running_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    mets = compute_metrics(logits_cat, targets_cat)
    mets["loss"] = avg_loss
    
    return mets


def train_model(train_loader, val_loader, config, device='cuda'):
    model = SmallCNNPlus(dropout=config['dropout']).to(device)
    
    if config.get('resume', False):
        checkpoint_path = Path(config['checkpoint_dir']) / 'best_model.pth'
        if checkpoint_path.exists():
            logging.info(f"Resuming from checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            logging.warning(f"Checkpoint not found at {checkpoint_path}, starting from scratch.")
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    if config['optimizer'].lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = SGD(model.parameters(), lr=config['lr'], 
                       weight_decay=config['weight_decay'], momentum=0.9, nesterov=True)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_metric = -float('inf')
    best_state = None
    history = []
    patience = config['patience']
    bad_epochs = 0
    
    save_dir = Path(config['checkpoint_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        logging.info(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Training
        tr_mets = run_one_epoch(model, train_loader, criterion, optimizer=optimizer, 
                                device=device, amp=config['amp'])
        
        # Validation
        vl_mets = run_one_epoch(model, val_loader, criterion, optimizer=None, 
                                device=device, amp=config['amp'])
        
        scheduler.step()
        
        row = {
            "epoch": epoch,
            "train": tr_mets,
            "val": vl_mets,
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(row)
        
        logging.info(f"Train loss: {tr_mets['loss']:.4f} | Val loss: {vl_mets['loss']:.4f}")
        logging.info(f"Val ROC-AUC: {vl_mets['roc_auc']:.4f} | Val PR-AUC: {vl_mets['pr_auc']:.4f} | Val F1: {vl_mets['f1']:.4f}")
        
        # Check for improvement
        metric_key = config['metric_for_best']
        score = vl_mets.get(metric_key, -float('inf'))
        if math.isnan(score):
            score = -float('inf')
        
        if score > best_metric:
            best_metric = score
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, save_dir / 'best_model.pth')
            logging.info(f"✓ New best {metric_key}: {best_metric:.4f}")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logging.info(f"Early stopping at epoch {epoch}. Best {metric_key}: {best_metric:.4f}")
                break
        
        # Save last checkpoint
        torch.save(model.state_dict(), save_dir / 'last_model.pth')
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


def evaluate_model(model, loader, device='cuda', amp=True):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    
    all_logits, all_targets = [], []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Evaluating'):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
                logits = model(images)
            
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    
    probs = torch.sigmoid(logits_cat).numpy()
    y_true = targets_cat.numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)
    
    mets = compute_metrics(logits_cat, targets_cat)
    
    logging.info("\nEvaluation Metrics:")
    for k, v in mets.items():
        logging.info(f"  {k}: {v:.4f}")
    
    logging.info("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, target_names=["non-stop", "stop"]))
    
    logging.info("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return mets

def main():
    set_seed(123)
    
    # Configuration
    config = {
        'input_size': 224,
        'batch_size': 32,
        'epochs': 60,
        'optimizer': 'adamw',
        'lr': 2e-4,
        'weight_decay': 5e-5,
        'dropout': 0.30,
        'num_workers': 4,
        'amp': True,
        'patience': 10,
        'metric_for_best': 'pr_auc',
        'checkpoint_dir': 'computations/cnn_checkpoints',
        'resume': True
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Data paths
    chips_dir = Path('data/processed/chips')
    
    if not chips_dir.exists():
        logging.error(f"Chips directory not found: {chips_dir}")
        logging.error("Please run data_prep.py first to generate training chips.")
        return
    
    # Create datasets
    train_dataset = ChipDataset(
        chips_dir, 
        build_train_aug(config['input_size']), 
        input_size=config['input_size'],
        split='train'
    )
    val_dataset = ChipDataset(
        chips_dir, 
        build_val_aug(config['input_size']), 
        input_size=config['input_size'],
        split='val'
    )
    test_dataset = ChipDataset(
        chips_dir, 
        build_val_aug(config['input_size']), 
        input_size=config['input_size'],
        split='test'
    )
    
    if len(train_dataset) == 0:
        logging.error("No training samples found!")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Train model
    logging.info("\n" + "="*60)
    logging.info("Starting Training")
    logging.info("="*60)
    
    model, history = train_model(train_loader, val_loader, config, device)
    
    # Evaluate on validation set
    logging.info("\n" + "="*60)
    logging.info("Validation Set Evaluation")
    logging.info("="*60)
    evaluate_model(model, val_loader, device, config['amp'])
    
    # Evaluate on test set
    if len(test_dataset) > 0:
        logging.info("\n" + "="*60)
        logging.info("Test Set Evaluation")
        logging.info("="*60)
        evaluate_model(model, test_loader, device, config['amp'])
    
    logging.info(f"\nTraining complete! Model saved to: {config['checkpoint_dir']}")


if __name__ == '__main__':
    main()