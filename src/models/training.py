# ============================================
# COMPLETE TRAINING PIPELINE
# For Your Pristine MVP Dataset
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("✅ Imports loaded\n")

# ============================================
# FIX TOKENIZER MULTIPROCESSING WARNINGS
# ============================================

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("✅ Tokenizer parallelism disabled")

# ============================================
# DATASET CLASS
# ============================================

class InteriorDesignDataset(Dataset):
    """Dataset with full bbox and spatial features"""
    
    def __init__(self, db_path: str, split: str = 'train', augment: bool = True):
        self.db_path = db_path
        self.split = split
        
        print(f"\n📊 Loading {split} dataset...")
        
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Load images with furniture data
        self.df = conn.execute("""
            SELECT 
                i.image_id,
                i.original_path,
                i.room_type,
                i.style,
                i.furniture_count
            FROM images i
            WHERE i.original_path IS NOT NULL
            AND i.room_type IS NOT NULL
            AND i.style IS NOT NULL
            AND i.furniture_count IS NOT NULL
        """).df()
        
        # Load furniture detections
        self.furniture_df = conn.execute("""
            SELECT * FROM furniture_detections
        """).df()
        
        conn.close()
        
        print(f"   Total images: {len(self.df):,}")
        print(f"   Total detections: {len(self.furniture_df):,}")
        
        # Train/val split
        train_df, val_df = train_test_split(
            self.df,
            test_size=0.2,
            stratify=self.df['room_type'],
            random_state=42
        )
        
        self.df = train_df if split == 'train' else val_df
        
        # Create label mappings
        self.room_types = sorted(self.df['room_type'].unique())
        self.styles = sorted(self.df['style'].unique())
        
        self.room_to_idx = {room: idx for idx, room in enumerate(self.room_types)}
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}
        
        print(f"   {split} set: {len(self.df):,} images")
        print(f"   Rooms: {self.room_types}")
        print(f"   Styles: {self.styles}\n")
        
        # Transforms
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        try:
            image = Image.open(row['original_path']).convert('RGB')
            image_tensor = self.transform(image)
        except:
            image_tensor = torch.zeros(3, 224, 224)
        
        # Get labels
        room_label = self.room_to_idx[row['room_type']]
        style_label = self.style_to_idx[row['style']]
        
        # Get furniture detections
        detections = self.furniture_df[
            self.furniture_df['image_id'] == row['image_id']
        ]
        
        # Calculate spatial features
        furniture_count = len(detections)
        
        if furniture_count > 0:
            # Average bbox area
            avg_area = detections['area_percentage'].mean()
            
            # Spatial distribution (variance of bbox centers)
            centers_x = (detections['bbox_x1'] + detections['bbox_x2']) / 2
            centers_y = (detections['bbox_y1'] + detections['bbox_y2']) / 2
            spatial_spread = centers_x.std() + centers_y.std()
            
            # Average confidence
            avg_confidence = detections['confidence'].mean()
            
            spatial_features = torch.tensor([
                furniture_count / 10.0,  # Normalize
                avg_area / 100.0,
                spatial_spread / 100.0,
                avg_confidence
            ], dtype=torch.float32)
        else:
            spatial_features = torch.zeros(4, dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'room_label': torch.tensor(room_label, dtype=torch.long),
            'style_label': torch.tensor(style_label, dtype=torch.long),
            'spatial_features': spatial_features,
            'image_id': row['image_id'],
            'image_path': row['original_path']
        }
    
    def get_detections(self, image_id: str):
        """Get furniture detections for an image"""
        return self.furniture_df[
            self.furniture_df['image_id'] == image_id
        ]

# ============================================
# MODEL ARCHITECTURE - IMPROVED VERSION
# ============================================

class InteriorDesignModel(nn.Module):
    """Multi-task model with spatial features - STABILIZED VERSION"""
    
    def __init__(self, num_rooms: int, num_styles: int):
        super().__init__()
        
        # Use ResNet18 instead of ResNet50 for stability
        self.backbone = models.resnet18(pretrained=True)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Spatial feature processor with BatchNorm AFTER activation
        self.spatial_processor = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),  # Moved after activation
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128)  # Moved after activation
        )
        
        combined_features = backbone_features + 128
        
        # Room classifier - Simplified
        self.room_classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_rooms)
        )
        
        # Style classifier - Simplified
        self.style_classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_styles)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, image, spatial_features):
        # Visual features with gradient clipping
        visual = self.backbone(image)
        
        # Clip visual features to prevent explosion
        visual = torch.clamp(visual, -10, 10)
        
        # Spatial features
        spatial = self.spatial_processor(spatial_features)
        
        # Clip spatial features
        spatial = torch.clamp(spatial, -10, 10)
        
        # Combine
        combined = torch.cat([visual, spatial], dim=1)
        
        # Classify
        room_logits = self.room_classifier(combined)
        style_logits = self.style_classifier(combined)
        
        return {
            'room_logits': room_logits,
            'style_logits': style_logits
        }

# ============================================
# TRAINING FUNCTIONS - WITH GRADIENT CLIPPING
# ============================================

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch with gradient clipping"""
    model.train()
    
    total_loss = 0
    room_correct = 0
    style_correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        room_labels = batch['room_label'].to(device)
        style_labels = batch['style_label'].to(device)
        spatial = batch['spatial_features'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images, spatial)
        
        # Losses
        room_loss = criterion(outputs['room_logits'], room_labels)
        style_loss = criterion(outputs['style_logits'], style_labels)
        loss = room_loss + style_loss
        
        # Check for NaN before backprop
        if torch.isnan(loss):
            print("⚠️ NaN detected in loss, skipping batch")
            continue
        
        loss.backward()
        
        # CRITICAL: Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        
        room_preds = outputs['room_logits'].argmax(dim=1)
        style_preds = outputs['style_logits'].argmax(dim=1)
        
        room_correct += (room_preds == room_labels).sum().item()
        style_correct += (style_preds == style_labels).sum().item()
        total += images.size(0)
        
        # Update progress
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'room_acc': f"{room_correct/total:.3f}",
            'style_acc': f"{style_correct/total:.3f}"
        })
    
    return total_loss / len(train_loader), room_correct / total, style_correct / total

def validate(model, val_loader, device):
    """Validate the model with NaN protection"""
    model.eval()
    
    total_loss = 0
    room_correct = 0
    style_correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            room_labels = batch['room_label'].to(device)
            style_labels = batch['style_label'].to(device)
            spatial = batch['spatial_features'].to(device)
            
            outputs = model(images, spatial)
            
            room_loss = criterion(outputs['room_logits'], room_labels)
            style_loss = criterion(outputs['style_logits'], style_labels)
            loss = room_loss + style_loss
            
            # Skip NaN losses
            if torch.isnan(loss):
                print("⚠️ NaN detected in validation, skipping batch")
                continue
            
            total_loss += loss.item()
            
            room_preds = outputs['room_logits'].argmax(dim=1)
            style_preds = outputs['style_logits'].argmax(dim=1)
            
            room_correct += (room_preds == room_labels).sum().item()
            style_correct += (style_preds == style_labels).sum().item()
            total += images.size(0)
    
    return total_loss / len(val_loader), room_correct / total, style_correct / total

# ============================================
# MAIN TRAINING LOOP - FIXED VERSION
# ============================================

def train_model(db_path: str, num_epochs: int = 25, batch_size: int = 32):
    """Main training function with all stability fixes"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎮 Training on: {device}\n")
    
    # Create datasets
    train_dataset = InteriorDesignDataset(db_path, split='train', augment=True)
    val_dataset = InteriorDesignDataset(db_path, split='val', augment=False)
    
    # Create dataloaders - FIX: num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # CHANGED from 4 to 0
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # CHANGED from 4 to 0
        pin_memory=True
    )
    
    # Create model
    model = InteriorDesignModel(
        num_rooms=len(train_dataset.room_types),
        num_styles=len(train_dataset.styles)
    ).to(device)
    
    print(f"📊 Model Info:")
    print(f"   Rooms: {len(train_dataset.room_types)}")
    print(f"   Styles: {len(train_dataset.styles)}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Optimizer & Scheduler - REDUCED LEARNING RATE
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,  # CHANGED from 0.001 to 5e-5 (20x smaller!)
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_room_acc': [], 'val_room_acc': [],
        'train_style_acc': [], 'val_style_acc': []
    }
    
    best_val_loss = float('inf')
    
    print("=" * 70)
    print(f"🚀 TRAINING FOR {num_epochs} EPOCHS")
    print("=" * 70 + "\n")
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_room_acc, train_style_acc = train_epoch(
            model, train_loader, optimizer, device
        )
        
        # Validate
        val_loss, val_room_acc, val_style_acc = validate(
            model, val_loader, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_room_acc'].append(train_room_acc)
        history['val_room_acc'].append(val_room_acc)
        history['train_style_acc'].append(train_style_acc)
        history['val_style_acc'].append(val_style_acc)
        
        # Print summary
        print(f"\n📈 Results:")
        print(f"   Loss:       Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"   Room Acc:   Train {train_room_acc:.3f} | Val {val_room_acc:.3f}")
        print(f"   Style Acc:  Train {train_style_acc:.3f} | Val {val_style_acc:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_room_acc': val_room_acc,
                'val_style_acc': val_style_acc,
                'room_types': train_dataset.room_types,
                'styles': train_dataset.styles
            }, 'pristine_mvp_model.pth')
            print(f"   💾 Saved best model!")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    # Plot training history
    plot_history(history)
    
    return history, model

def plot_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, weight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Room accuracy
    axes[1].plot(history['train_room_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_room_acc'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Room Classification Accuracy', fontsize=14, weight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Style accuracy
    axes[2].plot(history['train_style_acc'], label='Train', linewidth=2)
    axes[2].plot(history['val_style_acc'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('Style Classification Accuracy', fontsize=14, weight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Training plots saved: training_history.png")

# ============================================
# RUN TRAINING
# ============================================

db_path = "./interior_design_data_hybrid/processed/metadata.duckdb"

history, model = train_model(
    db_path=db_path,
    num_epochs=25,
    batch_size=32
)

print("\n🎉 Your Pristine MVP model is ready!")
print("📁 Model saved: pristine_mvp_model.pth")
print("📊 History saved: training_history.png")