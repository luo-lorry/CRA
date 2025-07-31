"""
Difficulty Predictor Model for Adaptive Conformal Prediction
"""

import torch
import torch.nn as nn
from torchvision import models


class DifficultyPredictor(nn.Module):
    """ResNet-based model for predicting instance-specific difficulty parameters"""

    def __init__(self, embedding_size=4, pretrained=True):
        super(DifficultyPredictor, self).__init__()
        
        # Load pretrained ResNet
        self.model = models.resnet50(pretrained=pretrained)

        # Modify first layer to accept 4-channel input (RGB + prediction probability)
        old_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            4, old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False
        )

        # Get input dimension of original fully connected layer
        old_fc_in_features = self.model.fc.in_features

        # Replace final FC layer with embedding layer + output
        self.model.fc = nn.Sequential(
            nn.Linear(old_fc_in_features, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

        self.return_embedding = False

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (B, 4, H, W) - RGB + prediction channel
            
        Returns:
            If return_embedding=False: Output predictions (B, 1)
            If return_embedding=True: (predictions, embeddings)
        """
        if self.return_embedding:
            # Extract features before final FC layer
            for name, module in list(self.model._modules.items())[:-1]:
                x = module(x)

            # Flatten features
            embedding = x.view(x.size(0), -1)

            # Apply final FC layers
            x = self.model.fc[0](embedding)  # First linear layer
            features = x  # Save embedding features
            x = self.model.fc[1](x)  # ReLU
            x = self.model.fc[2](x)  # Output layer

            return x, features
        else:
            return self.model(x)

    def set_return_embedding(self, value):
        """Set whether to return embeddings along with predictions"""
        self.return_embedding = value

    def freeze_backbone(self):
        """Freeze all layers except the final FC layer"""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = True