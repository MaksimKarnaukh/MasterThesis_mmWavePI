"""
Settings file with default configuration values for the project.
"""

import torch
from typing import Dict

folder_path_5ghz_10hz_collected: str = '../data/collected_csi_data_original_processed/5ghz/' # num_classes = 20
folder_path_5ghz_200hz_collected: str = '../data/collected_csi_data_original_processed/5ghz_200hz/' # num_classes = 18
folder_path_60ghz_collected: str = '../data/collected_csi_data_original_processed/60ghz/' # num_classes = 20
folder_path_60ghz_external: str = '../data/external_data_combined/' # num_classes = 7
output_path: str = '../output/'

DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES: int = 20
BATCH_SIZE: int = 32
EARLY_STOPPING_PATIENCE: int = 50
CRITERION = torch.nn.CrossEntropyLoss()
NUM_EPOCHS: int = 500
LR: float = 0.0007
OPTIMIZER: str = 'adam'
SMOOTHING_PROBABILITY: float = 0.0
MIXUP_ALPHA: float = 0.4 # for mixup augmentation
SMOOTHING_KERNEL_SIZE: int = 6 # for gaussian blurring
SMOOTHING_SIGMA: float = 1.0 # for gaussian blurring

ROWS_PER_SECOND: Dict[str, int] = {'5ghz_10hz': 10, '5ghz_200hz': 200, '60ghz_collected': 10, '60ghz_external': 22}
INPUT_DIM: Dict[str, int] = {'5ghz_10hz': 52, '5ghz_200hz': 52, '60ghz_collected': 60, '60ghz_external': 30}

VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15
