import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class RadioNet_NeuralWave(nn.Module):

    def __init__(self, input_dim, num_classes):
        """
        RadioNet Implementation based on NeuralWave paper.

        Parameters:
        - input_dim: Number of input features (e.g., 354 for PCA-reduced CSI data).
        - num_classes: Number of output classes (e.g., number of users).
        """
        super(RadioNet_NeuralWave, self).__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=17, stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=11, stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=11, stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(176, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # [batch_size, 1, input_dim]

        # x = x.transpose(1, 2)

        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
