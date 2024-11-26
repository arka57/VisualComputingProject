import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast
from tqdm import tqdm 

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BEST_MODEL_PATH = '/home/rs1/24CS91R03/best_resnet18.pth'

# Image transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(56),    
    transforms.RandomHorizontalFlip(p=0.5),   # 50% images will be horizonatally flipped
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.CenterCrop(56),
    transforms.ToTensor(),
])

#create a custom dataloader cuz default one is cumbersome to use
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        if mode == 'train':
            train_dir = os.path.join(root_dir, 'train')
            self.class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            for class_folder in self.class_names:
                class_path = os.path.join(train_dir, class_folder)
                images_path = os.path.join(class_path, 'images')
                if os.path.isdir(images_path):
                    for img_name in os.listdir(images_path):
                        if img_name.lower().endswith(('.jpeg','.png','.jpg')):
                            self.images.append(os.path.join(images_path, img_name))
                            self.labels.append(class_folder)
        else:
            val_annotations = pd.read_csv(os.path.join(root_dir, 'val', 'val_annotations.txt'), 
                                          sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
            self.images = [os.path.join(root_dir, 'val', 'images', file) for file in val_annotations['File']]
            self.labels = val_annotations['Class'].tolist()
            self.class_names = sorted(set(self.labels))

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.class_to_idx[label]

    def get_class_names(self):
        return self.class_names
    
# implemented early stopping to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        """Check if validation loss has improved enough."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement happens
        else:
            self.counter += 1  # Increment counter if no improvement
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping


print("Initializing the dataset")
# Prepare the train and the validation dataset for training
train_dataset = TinyImageNetDataset(root_dir='/home/rs1/24CS91R03/dataset', mode='train', transform=train_transforms)
val_dataset = TinyImageNetDataset(root_dir='/home/rs1/24CS91R03/dataset', mode='val', transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,pin_memory=True, prefetch_factor=2)

print("Donwloading the model")
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Sequential(
    nn.Dropout(p=0.50),  # Apply 50% dropout to the fully connected layer
    nn.Linear(model.fc.in_features, 200)  # Fully connected layer
)
model = model.to(DEVICE)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scaler = torch.amp.GradScaler('cuda')  

#TRAINING & VALIDATION LOOPS
def train_one_epoch(model, dataloader, criterion, optimizer,scaler,DEVICE='cuda'):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(dataloader, leave=True, desc='Training')

    for images, labels in loop:
        images, labels = images.to(DEVICE,non_blocking=True), labels.to(DEVICE,non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with autocasting to mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backpropagate with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Update tqdm progress bar with loss and accuracy
        loop.set_postfix(loss=loss.item(), acc=correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion,DEVICE='cuda'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    # Initialize tqdm progress bar for the validation loop
    loop = tqdm(dataloader, leave=True, desc='Validating')

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Update tqdm progress bar with loss and accuracy
            loop.set_postfix(loss=loss.item(), acc=correct / total )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# TRAINING THE MODEL
best_acc = 0.0
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

print("Starting training")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,scaler,DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion,DEVICE)

    print(f'Epoch [{epoch+1}/{EPOCHS}]')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Save the best model based on validation accuracy
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f'Best model saved with accuracy: {best_acc:.4f}')
    # Check for early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break    

print('Training Complete.')

