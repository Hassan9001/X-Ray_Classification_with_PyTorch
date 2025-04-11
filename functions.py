#!/usr/bin/env python3
"""
Improved training script for binary classification on chest X-ray images.
This script:
  - Creates a validation set by moving random files
  - Applies proper transforms (and optional augmentation)
  - Loads a pre-trained ResNet-50 and fine-tunes its final layer
  - Trains using early stopping with a validation loop and learning rate scheduler
  - Evaluates the model on a test set with accuracy and F1-score metrics
  - Saves the trained model in both state dict and TorchScript formats
"""

import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm  # Optional: progress bar for loops
# from tqdm.notebook import tqdm  # Notebook-friendly progress bar

# ---------------------------
# Device selection (global)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Utility Function: Move files for validation set creation
# ---------------------------
def move_files(src_class_dir: str, dest_class_dir: str, n: int = 50) -> None:
    if not os.path.exists(dest_class_dir):
        os.makedirs(dest_class_dir)
    files = os.listdir(src_class_dir)
    if len(files) < n:
        raise ValueError(f"Not enough files in {src_class_dir} to move {n} files.")
    random_files = random.sample(files, n)
    for f in random_files:
        shutil.move(os.path.join(src_class_dir, f), os.path.join(dest_class_dir, f))

# ---------------------------
# Create validation split
# ---------------------------
def prepare_validation_set():
    if not os.path.exists('data/val'):
        move_files('data/train/NORMAL', 'data/val/NORMAL', n=50)
        move_files('data/train/PNEUMONIA', 'data/val/PNEUMONIA', n=50)

# ---------------------------
# Data transforms and datasets
# ---------------------------
def prepare_datasets():
    transform_mean = [0.485, 0.456, 0.406]
    transform_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std)
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std)
    ])

    train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('data/val', transform=val_test_transform)
    test_dataset = datasets.ImageFolder('data/test', transform=val_test_transform)

    print("Training set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))
    print("Test set size:", len(test_dataset))

    return train_dataset, val_dataset, test_dataset

# ---------------------------
# DataLoaders
# ---------------------------
def prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size: int = 32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ---------------------------
# Model setup
# ---------------------------
def build_model() -> torch.nn.Module:
    model_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model_resnet.parameters():
        param.requires_grad = False

    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, 1)
    model_resnet = model_resnet.to(device)
    return model_resnet

# ---------------------------
# Training loop
# ---------------------------
def train_with_validation(model: torch.nn.Module,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          criterion: nn.Module,
                          optimizer: optim.Optimizer,
                          scheduler: optim.lr_scheduler.ReduceLROnPlateau,
                          num_epochs: int = 50,
                          patience: int = 5) -> None:
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            labels = labels.float().unsqueeze(1)

            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

# ---------------------------
# Model evaluation
# ---------------------------
def evaluate_model(model: torch.nn.Module, test_loader: DataLoader):
    model.eval()
    accuracy_metric = Accuracy(task="binary")
    f1_metric = F1Score(task="binary")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).round()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.unsqueeze(1).cpu().numpy().tolist())

    all_preds_tensor = torch.tensor(all_preds)
    all_labels_tensor = torch.tensor(all_labels)

    test_acc = accuracy_metric(all_preds_tensor, all_labels_tensor).item()
    test_f1 = f1_metric(all_preds_tensor, all_labels_tensor).item()

    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test F1-score: {test_f1:.3f}")
    return test_acc, test_f1

# ---------------------------
# Save model formats
# ---------------------------
def save_model_formats(model: torch.nn.Module):
    torch.save(model.state_dict(), 'model_final.pth')

    model_scripted = torch.jit.script(model)
    torch.jit.save(model_scripted, 'model_scripted.pt')

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model_traced = torch.jit.trace(model, dummy_input)
    torch.jit.save(model_traced, 'model_traced.pt')

    print("Model saved in state dict, scripted, and traced formats.")
