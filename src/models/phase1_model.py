"""
Phase 1 Model Architecture
This is the original architecture used for the Phase 1 trained model
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Phase1InteriorDesignModel(nn.Module):
    """
    Original Phase 1 model architecture (simpler version)
    This matches the checkpoint saved in models_best_interior_model.pth
    """

    def __init__(self, num_rooms: int, num_styles: int):
        super().__init__()

        # ResNet18 backbone
        self.backbone = models.resnet18(pretrained=False)
        backbone_features = self.backbone.fc.in_features  # 512 for ResNet18
        self.backbone.fc = nn.Identity()

        # Simple room classifier (matches checkpoint)
        # Checkpoint has layers at indices 1 and 4, so we need placeholders at 0, 2, 3
        self.room_classifier = nn.Sequential(
            nn.Identity(),  # room_classifier.0 (placeholder)
            nn.Linear(backbone_features, 128),  # room_classifier.1
            nn.ReLU(),  # room_classifier.2
            nn.Dropout(0.3),  # room_classifier.3
            nn.Linear(128, num_rooms)  # room_classifier.4
        )

        # Simple style classifier (matches checkpoint)
        # Checkpoint has layers at indices 1 and 4, so we need placeholders at 0, 2, 3
        self.style_classifier = nn.Sequential(
            nn.Identity(),  # style_classifier.0 (placeholder)
            nn.Linear(backbone_features, 128),  # style_classifier.1
            nn.ReLU(),  # style_classifier.2
            nn.Dropout(0.3),  # style_classifier.3
            nn.Linear(128, num_styles)  # style_classifier.4
        )

    def forward(self, image, furniture_features=None):
        """
        Forward pass
        Args:
            image: Image tensor
            furniture_features: Optional furniture context (not used in Phase 1)
        Returns:
            tuple: (room_logits, style_logits)
        """
        # Extract visual features
        visual = self.backbone(image)

        # Classify
        room_logits = self.room_classifier(visual)
        style_logits = self.style_classifier(visual)

        return room_logits, style_logits
