import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class LSTM_HumanFi(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.2, bidirectional=False):
        """
        Initialize the LSTM-based classifier.

        Parameters:
        - input_dim: Number of features (e.g., 30 for subcarriers).
        - hidden_dim: Number of hidden units in LSTM.
        - num_classes: Number of classes (e.g., 3 for background, stand, walk).
        - num_layers: Number of LSTM layers.
        """
        super(LSTM_HumanFi, self).__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*self.num_directions, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        output = self.dropout(hidden)
        output = self.fc(output)
        return output

###############################################################################################################

class CNN_LSTM(nn.Module):

    def __init__(self, input_channels, num_classes, conv_channels=64, lstm_hidden=128, lstm_layers=1,
                 bidirectional=False):

        super(CNN_LSTM, self).__init__()

        # 1D CNN to extract local features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Halves the time dimension
        )

        self.bidirectional = bidirectional
        lstm_input_size = conv_channels
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        lstm_out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # back to [B, T, C] for LSTM

        # LSTM expects (B, T, F)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Use last time step
        return self.fc(out)

###############################################################################################################

class CNN_BiLSTM_TemporalAttention(nn.Module):
    def __init__(self, input_dim, cnn_channels, lstm_hidden_dim, lstm_layers, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels)
        )
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)

        self.attn = nn.Linear(2 * lstm_hidden_dim, 1)
        self.fc = nn.Linear(2 * lstm_hidden_dim, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.cnn(x)        # [B, C', T]
        x = x.transpose(1, 2)  # [B, T, C']
        lstm_out, _ = self.lstm(x)  # [B, T, 2H]
        weights = F.softmax(self.attn(lstm_out), dim=1)  # [B, T, 1]
        attended = (weights * lstm_out).sum(dim=1)  # [B, 2H]
        return self.fc(attended)


class CNN_BiLSTM_ChannelAttention(nn.Module):
    def __init__(self, input_dim, cnn_channels, lstm_hidden_dim, lstm_layers, num_classes):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1)

        # SE block (channel attention)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_attn = nn.Sequential(
            nn.Linear(cnn_channels, cnn_channels // 4),
            nn.ReLU(),
            nn.Linear(cnn_channels // 4, cnn_channels),
            nn.Sigmoid()
        )

        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * lstm_hidden_dim, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        feat = self.cnn(x)     # [B, C', T]

        # Channel attention
        pooled = self.global_pool(feat).squeeze(-1)  # [B, C']
        scale = self.channel_attn(pooled).unsqueeze(-1)  # [B, C', 1]
        feat = feat * scale  # [B, C', T]

        x = feat.transpose(1, 2)  # [B, T, C']
        lstm_out, _ = self.lstm(x)
        output = lstm_out[:, -1, :]  # or mean pooling
        return self.fc(output)

class CNN_BiLSTM_DualAttention(nn.Module):
    def __init__(self, input_dim, cnn_channels, lstm_hidden_dim, lstm_layers, num_classes):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1)

        # Channel attention (SE)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_attn = nn.Sequential(
            nn.Linear(cnn_channels, cnn_channels // 4),
            nn.ReLU(),
            nn.Linear(cnn_channels // 4, cnn_channels),
            nn.Sigmoid()
        )

        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)

        # Temporal attention
        self.temporal_attn = nn.Linear(2 * lstm_hidden_dim, 1)
        self.fc = nn.Linear(2 * lstm_hidden_dim, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)       # [B, C, T]
        feat = self.cnn(x)          # [B, C', T]

        # Channel attention
        pooled = self.global_pool(feat).squeeze(-1)
        scale = self.channel_attn(pooled).unsqueeze(-1)
        feat = feat * scale

        x = feat.transpose(1, 2)  # [B, T, C']
        lstm_out, _ = self.lstm(x)  # [B, T, 2H]

        # Temporal attention
        weights = F.softmax(self.temporal_attn(lstm_out), dim=1)
        attended = (weights * lstm_out).sum(dim=1)
        return self.fc(attended)


class Attention3DBlock(nn.Module):
    """
    3D attention mechanism over LSTM outputs (time step attention).
    Input shape: (B, T, F) → Output shape: (B, T, F) with attention weights applied
    """
    def __init__(self, feature_dim):
        super(Attention3DBlock, self).__init__()
        self.attention_dense = nn.Linear(feature_dim, feature_dim)

    def forward(self, inputs):
        # inputs: [B, T, F]
        attention_weights = F.softmax(self.attention_dense(inputs), dim=1)  # shape: [B, T, F]
        return inputs * attention_weights  # element-wise multiplication


class CNN_BiLSTM_Attention(nn.Module):
    """
    CNN-BiLSTM-Attention model based on Keras implementation from the SOH prediction paper.
    """

    def __init__(self, input_dim, num_classes, cnn_filters=64, lstm_units=128, dropout=0.3, num_layers=1):
        super(CNN_BiLSTM_Attention, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(input_size=cnn_filters,
                            hidden_size=lstm_units,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.attention = Attention3DBlock(feature_dim=2 * lstm_units)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(2 * lstm_units, num_classes)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # → [B, C, T]
        x = self.cnn(x)        # → [B, C', T]
        x = x.transpose(1, 2)  # → [B, T, C']

        lstm_out, _ = self.lstm(x)  # → [B, T, 2 * H]
        attended = self.attention(lstm_out)  # [B, T, 2H]
        out = attended[:, -1, :]  # or: attended.mean(dim=1)

        return self.output(out)
