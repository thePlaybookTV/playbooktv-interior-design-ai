"""
Mask-Enhanced Style Classifier with Shape and Color Features
Integrates SAM2 mask-derived features (shape, color, spatial) with visual features
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

# Import our feature extractors
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.shape_feature_extractor import ShapeFeatureExtractor
from src.models.color_extractor import MaskBasedColorExtractor


class MaskEnhancedStyleDataset(Dataset):
    """
    Enhanced dataset with SAM2 mask-derived features
    Includes shape, color, and spatial arrangement features
    """

    def __init__(
        self,
        db_path: str,
        split: str = 'train',
        augment: bool = True,
        use_mask_features: bool = True,
        min_mask_score: float = 0.7
    ):
        self.db_path = db_path
        self.split = split
        self.use_mask_features = use_mask_features
        self.min_mask_score = min_mask_score

        # Connect to database
        conn = duckdb.connect(db_path)

        # Load images with style labels and mask detections
        if use_mask_features:
            # Only use images that have high-quality SAM2 masks
            self.df = conn.execute("""
                SELECT
                    i.image_id,
                    i.original_path,
                    i.style,
                    i.room_type,
                    i.furniture_count,
                    i.color_palette,
                    COUNT(fd.detection_id) as mask_count,
                    AVG(fd.mask_score) as avg_mask_score
                FROM images i
                INNER JOIN furniture_detections fd ON i.image_id = fd.image_id
                WHERE i.original_path IS NOT NULL
                AND i.style IS NOT NULL
                AND i.style != 'unknown'
                AND fd.has_mask = TRUE
                AND fd.mask_score >= ?
                GROUP BY i.image_id, i.original_path, i.style, i.room_type, i.furniture_count, i.color_palette
                HAVING COUNT(fd.detection_id) > 0
            """, [min_mask_score]).df()
        else:
            # Use all images
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

        conn.close()

        print(f"   Total images: {len(self.df):,}")
        print(f"   Styles: {sorted(self.df['style'].unique())}")

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

        # Initialize feature extractors if using mask features
        if use_mask_features:
            self.shape_extractor = ShapeFeatureExtractor(db_path)
            self.color_extractor = MaskBasedColorExtractor(db_path)
        else:
            self.shape_extractor = None
            self.color_extractor = None

        # Image transforms
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(15),
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

        # Get label
        style_label = self.style_to_idx[row['style']]

        # Extract mask-based features
        if self.use_mask_features and self.shape_extractor:
            try:
                # Get spatial arrangement features
                arrangement = self.shape_extractor.extract_spatial_arrangement_features(
                    row['image_id'],
                    self.min_mask_score
                )

                # Get room-level color features
                room_colors = self.color_extractor.extract_room_color_features(
                    row['original_path'],
                    row['image_id'],
                    self.min_mask_score
                )

                # Compile features
                mask_features = [
                    # Spatial arrangement
                    arrangement['item_count'] / 20.0,  # Normalized
                    arrangement['total_coverage_pct'] / 100.0,
                    arrangement['avg_distance_between_items'] / 1000.0,  # Normalized
                    arrangement['x_spread'] / 1000.0,
                    arrangement['y_spread'] / 1000.0,
                    arrangement['area_variance'] / 10000.0,

                    # Color features
                    room_colors['room_brightness'],
                    room_colors['room_colorfulness'],
                    room_colors['room_warm_cool'],
                    room_colors['background_brightness'],
                    float(room_colors['num_furniture_items']) / 20.0,
                ]

            except Exception as e:
                # Fallback to zeros if feature extraction fails
                mask_features = [0.0] * 11

        else:
            mask_features = [0.0] * 11

        mask_features_tensor = torch.tensor(mask_features, dtype=torch.float32)

        return {
            'image': image_tensor,
            'style_label': torch.tensor(style_label, dtype=torch.long),
            'mask_features': mask_features_tensor,
            'image_id': row['image_id']
        }

    def close(self):
        """Close feature extractors"""
        if self.shape_extractor:
            self.shape_extractor.close()
        if self.color_extractor:
            self.color_extractor.close()


class MaskEnhancedStyleClassifier(nn.Module):
    """
    Style classifier that combines visual features with mask-derived features

    Architecture:
    - Visual branch: EfficientNet-B0, ResNet50, or ViT-B/16
    - Mask features branch: MLP processing shape/color/spatial features
    - Fusion: Concatenate + attention mechanism
    """

    def __init__(
        self,
        num_styles: int,
        backbone: str = 'efficientnet',
        use_mask_features: bool = True,
        mask_feature_dim: int = 11
    ):
        super().__init__()

        self.use_mask_features = use_mask_features

        # Visual backbone
        if backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=True)
            backbone_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == 'vit':
            self.backbone = models.vit_b_16(pretrained=True)
            backbone_features = self.backbone.heads[0].in_features
            self.backbone.heads = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Mask features processor
        if use_mask_features:
            self.mask_processor = nn.Sequential(
                nn.Linear(mask_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(64),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256)
            )

            combined_features = backbone_features + 256
        else:
            combined_features = backbone_features

        # Fusion with attention
        self.attention = nn.Sequential(
            nn.Linear(combined_features, combined_features // 2),
            nn.Tanh(),
            nn.Linear(combined_features // 2, combined_features),
            nn.Sigmoid()
        )

        # Final classifier
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

    def forward(self, image, mask_features=None):
        # Extract visual features
        visual_features = self.backbone(image)

        # Process mask features if available
        if self.use_mask_features and mask_features is not None:
            mask_features_processed = self.mask_processor(mask_features)
            # Concatenate
            combined = torch.cat([visual_features, mask_features_processed], dim=1)
        else:
            combined = visual_features

        # Apply attention
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights

        # Classify
        logits = self.classifier(attended_features)

        return logits


class MaskEnhancedEnsemble:
    """
    Ensemble of mask-enhanced classifiers
    Combines EfficientNet, ResNet, and ViT with mask features
    """

    def __init__(self, num_styles: int, device='cuda'):
        self.num_styles = num_styles
        self.device = device

        # Create three models with different backbones
        self.model_efficientnet = MaskEnhancedStyleClassifier(
            num_styles, backbone='efficientnet', use_mask_features=True
        ).to(device)

        self.model_resnet = MaskEnhancedStyleClassifier(
            num_styles, backbone='resnet50', use_mask_features=True
        ).to(device)

        self.model_vit = MaskEnhancedStyleClassifier(
            num_styles, backbone='vit', use_mask_features=True
        ).to(device)

        self.models = [self.model_efficientnet, self.model_resnet, self.model_vit]
        self.model_names = ['EfficientNet-B0', 'ResNet50', 'ViT-B/16']

    def train_model(
        self,
        model,
        train_loader,
        val_loader,
        epochs=30,
        lr=0.001,
        model_name='model'
    ):
        """Train a single model in the ensemble"""

        print(f"\n🏋️ Training {model_name}...")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )

        best_acc = 0
        best_model_state = None

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = batch['image'].to(self.device)
                labels = batch['style_label'].to(self.device)
                mask_features = batch['mask_features'].to(self.device)

                optimizer.zero_grad()
                outputs = model(images, mask_features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_acc = 100.0 * train_correct / train_total

            # Validate
            val_acc = self.evaluate_model(model, val_loader)

            scheduler.step(val_acc)

            print(f"   Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = model.state_dict().copy()
                print(f"   ✅ New best: {best_acc:.2f}%")

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        print(f"\n✅ {model_name} training complete. Best Val Acc: {best_acc:.2f}%")

        return best_acc

    def evaluate_model(self, model, data_loader):
        """Evaluate a single model"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                labels = batch['style_label'].to(self.device)
                mask_features = batch['mask_features'].to(self.device)

                outputs = model(images, mask_features)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100.0 * correct / total

    def train_ensemble(
        self,
        train_loader,
        val_loader,
        epochs=30,
        lr=0.001
    ):
        """Train all models in the ensemble"""

        print("=" * 70)
        print("🎯 TRAINING MASK-ENHANCED ENSEMBLE")
        print("=" * 70)

        results = {}

        for model, name in zip(self.models, self.model_names):
            best_acc = self.train_model(
                model, train_loader, val_loader, epochs, lr, name
            )
            results[name] = best_acc

        print("\n" + "=" * 70)
        print("✅ ENSEMBLE TRAINING COMPLETE")
        print("=" * 70)
        print("\n📊 Individual Model Results:")
        for name, acc in results.items():
            print(f"   {name}: {acc:.2f}%")

        return results

    def predict_ensemble(self, data_loader):
        """Make predictions using ensemble voting"""

        all_predictions = []
        all_labels = []

        # Get predictions from each model
        predictions_by_model = [[] for _ in self.models]

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Ensemble prediction"):
                images = batch['image'].to(self.device)
                labels = batch['style_label'].to(self.device)
                mask_features = batch['mask_features'].to(self.device)

                # Get predictions from each model
                for i, model in enumerate(self.models):
                    model.eval()
                    outputs = model(images, mask_features)
                    _, predicted = outputs.max(1)
                    predictions_by_model[i].extend(predicted.cpu().numpy())

                all_labels.extend(labels.cpu().numpy())

        # Ensemble voting (majority vote)
        predictions_by_model = np.array(predictions_by_model)  # (3, N)

        # Majority vote
        ensemble_predictions = []
        for i in range(predictions_by_model.shape[1]):
            votes = predictions_by_model[:, i]
            # Get most common prediction
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_predictions.append(unique[np.argmax(counts)])

        ensemble_predictions = np.array(ensemble_predictions)
        all_labels = np.array(all_labels)

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, ensemble_predictions)

        return ensemble_predictions, all_labels, accuracy

    def save_ensemble(self, save_dir: str):
        """Save all models in the ensemble"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for model, name in zip(self.models, self.model_names):
            model_path = save_dir / f"{name.lower().replace('-', '_').replace('/', '_')}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved {name} to {model_path}")

    def load_ensemble(self, save_dir: str):
        """Load all models in the ensemble"""
        save_dir = Path(save_dir)

        for model, name in zip(self.models, self.model_names):
            model_path = save_dir / f"{name.lower().replace('-', '_').replace('/', '_')}.pth"
            model.load_state_dict(torch.load(model_path))
            print(f"📥 Loaded {name} from {model_path}")


# ============================================
# TRAINING SCRIPT
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train mask-enhanced style classifier ensemble')
    parser.add_argument('--db', type=str, required=True,
                        help='Path to DuckDB database')
    parser.add_argument('--output', type=str, default='./mask_enhanced_models',
                        help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--min-mask-score', type=float, default=0.7,
                        help='Minimum mask score')

    args = parser.parse_args()

    print("=" * 70)
    print("🎨 MASK-ENHANCED STYLE CLASSIFIER TRAINING")
    print("=" * 70)

    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")

    # Create datasets
    print("\n📊 Loading training data...")
    train_dataset = MaskEnhancedStyleDataset(
        args.db,
        split='train',
        augment=True,
        use_mask_features=True,
        min_mask_score=args.min_mask_score
    )

    print("\n📊 Loading validation data...")
    val_dataset = MaskEnhancedStyleDataset(
        args.db,
        split='val',
        augment=False,
        use_mask_features=True,
        min_mask_score=args.min_mask_score
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create ensemble
    num_styles = len(train_dataset.styles)
    print(f"\n🎯 Creating ensemble for {num_styles} styles")

    ensemble = MaskEnhancedEnsemble(num_styles, device=device)

    # Train ensemble
    results = ensemble.train_ensemble(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr
    )

    # Evaluate ensemble
    print("\n📊 Evaluating ensemble...")
    predictions, labels, ensemble_acc = ensemble.predict_ensemble(val_loader)

    print(f"\n🎯 Ensemble Accuracy: {ensemble_acc*100:.2f}%")

    # Save models
    ensemble.save_ensemble(args.output)

    print(f"\n✅ Training complete! Models saved to {args.output}")

    # Cleanup
    train_dataset.close()
    val_dataset.close()
