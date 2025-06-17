import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from src.models._ECA_module import ECALayer1D


###############################################################################################################

# source: https://github.com/BangguWu/ECANet/blob/master/models/eca_resnet.py
# follows original ECA_ResNet implementation

class ECABasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.eca = ECALayer1D(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

class ECABottleneckBlock1D(nn.Module):
    """
    1D Bottleneck block with Efficient Channel Attention (ECA).
    """
    expansion = 4  # Matches the ResNet bottleneck design

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, k_size=3):
        super(ECABottleneckBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.eca = ECALayer1D(out_channels * self.expansion, k_size=k_size)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

class ECAResNet1D(nn.Module):

    def __init__(self, block, layers, input_channels=52, num_classes=20, k_size=(3, 3, 3, 3)):
        """
        ECA-ResNet1D model for activity classification.
        :param block:
        :param layers:
        :param input_channels:
        :param num_classes:
        :param k_size:
        """
        super(ECAResNet1D, self).__init__()
        self.inplanes = 128

        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, k_size=k_size[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, k_size=k_size[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, k_size=k_size[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, k_size=k_size[3])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, k_size=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def eca_resnet1d18(input_channels=52, num_classes=20, k_size=(3, 3, 3, 3)):
    return ECAResNet1D(ECABasicBlock1D, [2, 2, 2, 2], input_channels=input_channels, num_classes=num_classes, k_size=k_size)

def eca_resnet1d34(input_channels=52, num_classes=20, k_size=(3, 3, 3, 3)):
    return ECAResNet1D(ECABasicBlock1D, [3, 4, 6, 3], input_channels=input_channels, num_classes=num_classes, k_size=k_size)

def eca_resnet1d50(input_channels=52, num_classes=20, k_size=(3, 3, 3, 3)):
    return ECAResNet1D(ECABottleneckBlock1D, [3, 4, 6, 3], input_channels=input_channels, num_classes=num_classes, k_size=k_size)

def eca_resnet1d101(input_channels=52, num_classes=20, k_size=(3, 3, 3, 3)):
    return ECAResNet1D(ECABottleneckBlock1D, [3, 4, 23, 3], input_channels=input_channels, num_classes=num_classes, k_size=k_size)

def eca_resnet1d152(input_channels=52, num_classes=20, k_size=(3, 3, 3, 3)):
    return ECAResNet1D(ECABottleneckBlock1D, [3, 8, 36, 3], input_channels=input_channels, num_classes=num_classes, k_size=k_size)

###############################################################################################################

# source: https://github.com/geekfeiw/ARIL/blob/master/models/apl.py
# follows original paper 'Joint Activity Recognition and Indoor Localization With WiFi Fingerprints' implementation

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock_JARILWWF(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_JARILWWF, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

class ResNet1D_JARILWWF(nn.Module):

    def __init__(self, input_channels, num_classes, layers):
        super(ResNet1D_JARILWWF, self).__init__()
        self.inplanes = 128

        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock_JARILWWF, 128, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock_JARILWWF, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock_JARILWWF, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock_JARILWWF, 512, layers[3], stride=2)

        self.classifier = nn.Sequential(
            nn.Conv1d(512 * BasicBlock_JARILWWF.expansion, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, Time, Channels) → (B, Channels, Time)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# similar, but better implementation of JARILWWF

class OptBasicBlock_JARILWWF(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

class OptResNet1D_JARILWWF(nn.Module):

    def __init__(self, input_channels, num_classes, layers=(2, 2, 2, 2), base_width=128):
        super().__init__()
        self.inplanes = base_width

        # More conservative initial stride
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_width, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

        self.layer1 = self._make_layer(OptBasicBlock_JARILWWF, base_width, layers[0], stride=1)
        self.layer2 = self._make_layer(OptBasicBlock_JARILWWF, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(OptBasicBlock_JARILWWF, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(OptBasicBlock_JARILWWF, base_width * 8, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_width * 8, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm1d(planes)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, L, C] → [B, C, L]
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

class OptECAResNet1D_JARILWWF(nn.Module):

    def __init__(self, input_channels=52, num_classes=20, layers=(2, 2, 2, 2), k_sizes=(3, 3, 3, 3)):
        super().__init__()
        self.inplanes = 128

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

        self.layer1 = self._make_layer(ECABasicBlock1D, 128, layers[0], stride=1, k_size=k_sizes[0])
        self.layer2 = self._make_layer(ECABasicBlock1D, 256, layers[1], stride=2, k_size=k_sizes[1])
        self.layer3 = self._make_layer(ECABasicBlock1D, 512, layers[2], stride=2, k_size=k_sizes[2])
        self.layer4 = self._make_layer(ECABasicBlock1D, 512, layers[3], stride=2, k_size=k_sizes[3])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride, k_size):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

        layers = [block(self.inplanes, planes, stride, downsample, k_size)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, L, C] → [B, C, L]
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).squeeze(-1)
        return self.fc(x)


###############################################################################################################

# based on models above, but with a more basic structure

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        A basic residual block with Conv1D layers, BatchNorm1D, and ReLU.

        Parameters:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels.
        - kernel_size: Size of the convolutional filter.
        - stride: Stride for the convolutional layers.
        - padding: Padding for the convolutional layers.
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                  stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu(x)
        return x

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

class ECAResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, eca_kernel_size=3):
        super(ECAResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.eca = ECALayer1D(out_channels, k_size=eca_kernel_size)

        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.eca(x)
        x += residual
        x = self.relu(x)
        return x

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

class CustomResNet1D(nn.Module):

    def __init__(self, input_channels, num_classes):
        """
        Residual CNN-based model for activity classification.

        """
        super(CustomResNet1D, self).__init__()

        self.initial_conv = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.initial_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.rb1 = ResidualBlock(in_channels=128, out_channels=128)
        self.rb2 = ResidualBlock(in_channels=128, out_channels=128)
        self.rb3 = ResidualBlock(in_channels=128, out_channels=256)
        self.rb4 = ResidualBlock(in_channels=256, out_channels=512)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.initial_conv(x)
        x = self.initial_pool(x)

        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        x = self.fc(x)
        return x

class CustomECAResNet1D(nn.Module):

    def __init__(self, input_channels, num_classes):
        """
        Residual CNN-based model for activity classification with ECA attention.

        """
        super(CustomECAResNet1D, self).__init__()

        self.initial_conv = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.initial_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.rb1 = ECAResidualBlock(in_channels=128, out_channels=128)
        self.rb2 = ECAResidualBlock(in_channels=128, out_channels=128)
        self.rb3 = ECAResidualBlock(in_channels=128, out_channels=256)
        self.rb4 = ECAResidualBlock(in_channels=256, out_channels=512)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.initial_conv(x)
        x = self.initial_pool(x)

        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        x = self.fc(x)
        return x