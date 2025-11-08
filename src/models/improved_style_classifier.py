"""
Improved Style Classification with Ensemble Methods
Combines multiple models and techniques to achieve better style classification accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


class ImprovedStyleDataset(Dataset):
    """Enhanced dataset with better augmentation for style classification"""

    def __init__(self, db_path: str, split: str = 'train', augment: bool = True, use_furniture_context: bool = True):
        self.db_path = db_path
        self.split = split
        self.use_furniture_context = use_furniture_context

        # Connect to database
        conn = duckdb.connect(db_path)

        # Load images with style labels
        self.df = conn.execute("""
            SELECT
                i.image_id,
                i.original_path,
                i.style,
                i.room_type,
                i.furniture_count,
                i.color_palette
            FROM images i
            WHERE i.original_path IS NOT NULL
            AND i.style IS NOT NULL
            AND i.style != 'unknown'
        """).df()

        # Load furniture detections if using context
        if use_furniture_context:
            self.furniture_df = conn.execute("""
                SELECT * FROM furniture_detections
            """).df()
        else:
            self.furniture_df = None

        conn.close()

        print(f"   Total images: {len(self.df):,}")
        print(f"   Styles: {self.df['style'].unique()}")

        # Train/val split
        train_df, val_df = train_test_split(
            self.df,
            test_size=0.2,
            stratify=self.df['style'],
            random_state=42
        )

        self.df = train_df if split == 'train' else val_df

        # Create label mappings
        self.styles = sorted(self.df['style'].unique())
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}

        print(f"   {split} set: {len(self.df):,} images")
        print(f"   Style distribution:")
        for style in self.styles:
            count = len(self.df[self.df['style'] == style])
            print(f"      {style}: {count}")

        # Enhanced transforms with stronger augmentation
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
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

        # Get label
        style_label = self.style_to_idx[row['style']]

        # Get furniture context features
        if self.use_furniture_context and self.furniture_df is not None:
            detections = self.furniture_df[
                self.furniture_df['image_id'] == row['image_id']
            ]

            furniture_count = len(detections)

            if furniture_count > 0:
                # Furniture type distribution (one-hot encoding of top items)
                furniture_types = detections['item_type'].value_counts().to_dict()

                # Create feature vector
                context_features = [
                    furniture_count / 10.0,  # Normalized count
                    detections['area_percentage'].mean() / 100.0 if len(detections) > 0 else 0.0,
                    detections['confidence'].mean() if len(detections) > 0 else 0.0,
                ]
            else:
                context_features = [0.0, 0.0, 0.0]

            context_tensor = torch.tensor(context_features, dtype=torch.float32)
        else:
            context_tensor = torch.zeros(3, dtype=torch.float32)

        return {
            'image': image_tensor,
            'style_label': torch.tensor(style_label, dtype=torch.long),
            'context_features': context_tensor,
            'image_id': row['image_id']
        }


class EfficientNetStyleClassifier(nn.Module):
    """EfficientNet-based style classifier"""

    def __init__(self, num_styles: int, use_context: bool = True):
        super().__init__()

        # Use EfficientNet-B0 (lighter and often better than ResNet)
        self.backbone = models.efficientnet_b0(pretrained=True)
        backbone_features = self.backbone.classifier[1].in_features

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        self.use_context = use_context

        if use_context:
            # Context processor
            self.context_processor = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(64),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(128)
            )

            combined_features = backbone_features + 128
        else:
            combined_features = backbone_features

        # Style classifier with attention
        self.attention = nn.Sequential(
            nn.Linear(combined_features, combined_features // 2),
            nn.Tanh(),
            nn.Linear(combined_features // 2, combined_features),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_styles)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, context_features=None):
        # Visual features
        visual = self.backbone(image)

        if self.use_context and context_features is not None:
            # Context features
            context = self.context_processor(context_features)

            # Combine
            combined = torch.cat([visual, context], dim=1)
        else:
            combined = visual

        # Attention mechanism
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights

        # Classify
        style_logits = self.classifier(attended_features)

        return style_logits


class ResNetStyleClassifier(nn.Module):
    """ResNet50-based style classifier for ensemble"""

    def __init__(self, num_styles: int):
        super().__init__()

        self.backbone = models.resnet50(pretrained=True)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_styles)
        )

    def forward(self, image, context_features=None):
        visual = self.backbone(image)
        return self.classifier(visual)


class VisionTransformerStyleClassifier(nn.Module):
    """Vision Transformer for style classification"""

    def __init__(self, num_styles: int):
        super().__init__()

        # Use ViT-B/16 pretrained
        self.backbone = models.vit_b_16(pretrained=True)
        backbone_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_styles)
        )

    def forward(self, image, context_features=None):
        visual = self.backbone(image)
        return self.classifier(visual)


class EnsembleStyleClassifier:
    """Ensemble of multiple models for improved accuracy"""

    def __init__(self, num_styles: int, device: str = 'cuda'):
        self.num_styles = num_styles
        self.device = device

        # Create ensemble models
        self.models = {
            'efficientnet': EfficientNetStyleClassifier(num_styles, use_context=True).to(device),
            'resnet': ResNetStyleClassifier(num_styles).to(device),
            'vit': VisionTransformerStyleClassifier(num_styles).to(device)
        }

        # Model weights (can be adjusted after validation)
        self.weights = {
            'efficientnet': 0.4,
            'resnet': 0.35,
            'vit': 0.25
        }

    def train_model(self, model_name: str, train_loader, val_loader, epochs: int = 30):
        """Train individual model in ensemble"""

        model = self.models[model_name]
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        print(f"\n🏋️ Training {model_name}...")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = batch['image'].to(self.device)
                labels = batch['style_label'].to(self.device)
                context = batch['context_features'].to(self.device)

                optimizer.zero_grad()

                outputs = model(images, context)
                loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                train_total += labels.size(0)

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    labels = batch['style_label'].to(self.device)
                    context = batch['context_features'].to(self.device)

                    outputs = model(images, context)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)

            scheduler.step()

            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            print(f"   Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_{model_name}_style_classifier.pth')
                print(f"   💾 Saved best {model_name} model (Val Acc: {val_acc:.4f})")

        return best_val_acc

    def predict_ensemble(self, image, context_features=None):
        """Predict using ensemble of models"""

        predictions = {}

        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                logits = model(image, context_features)
                probs = torch.softmax(logits, dim=1)
                predictions[model_name] = probs

        # Weighted average
        ensemble_probs = sum(
            predictions[name] * self.weights[name]
            for name in self.models.keys()
        )

        return ensemble_probs

    def evaluate_ensemble(self, val_loader):
        """Evaluate ensemble on validation set"""

        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(val_loader, desc="Evaluating ensemble"):
            images = batch['image'].to(self.device)
            labels = batch['style_label'].to(self.device)
            context = batch['context_features'].to(self.device)

            ensemble_probs = self.predict_ensemble(images, context)
            preds = ensemble_probs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total

        print(f"\n✅ Ensemble Validation Accuracy: {accuracy:.4f}")

        return accuracy, all_preds, all_labels


# ============================================
# TRAINING FUNCTION
# ============================================

def train_improved_style_classifier(db_path: str, epochs: int = 30, batch_size: int = 32):
    """Train improved style classifier with ensemble"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎮 Training on: {device}\n")

    # Create datasets
    print("📊 Loading datasets...")
    train_dataset = ImprovedStyleDataset(db_path, split='train', augment=True, use_furniture_context=True)
    val_dataset = ImprovedStyleDataset(db_path, split='val', augment=False, use_furniture_context=True)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Create ensemble
    ensemble = EnsembleStyleClassifier(
        num_styles=len(train_dataset.styles),
        device=device
    )

    # Train each model
    results = {}

    for model_name in ensemble.models.keys():
        val_acc = ensemble.train_model(
            model_name,
            train_loader,
            val_loader,
            epochs=epochs
        )
        results[model_name] = val_acc

    # Evaluate ensemble
    print("\n" + "=" * 70)
    print("📊 EVALUATING ENSEMBLE")
    print("=" * 70)

    ensemble_acc, preds, labels = ensemble.evaluate_ensemble(val_loader)

    # Print individual model results
    print(f"\n📊 Individual Model Results:")
    for model_name, acc in results.items():
        print(f"   {model_name}: {acc:.4f}")

    print(f"\n🎯 Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"   Improvement: {(ensemble_acc - max(results.values())):.4f}")

    # Save results
    final_results = {
        'individual_models': results,
        'ensemble_accuracy': ensemble_acc,
        'num_styles': len(train_dataset.styles),
        'styles': train_dataset.styles
    }

    with open('improved_style_classifier_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n✅ Results saved: improved_style_classifier_results.json")

    return ensemble, results, ensemble_acc


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    DB_PATH = "./interior_design_data_hybrid/processed/metadata.duckdb"

    print("=" * 70)
    print("🚀 IMPROVED STYLE CLASSIFICATION WITH ENSEMBLE")
    print("=" * 70)

    ensemble, results, accuracy = train_improved_style_classifier(
        db_path=DB_PATH,
        epochs=30,
        batch_size=32
    )

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
