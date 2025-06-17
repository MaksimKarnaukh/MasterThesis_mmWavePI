import numpy as np
import os
import random

# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch
import re
import time
from datetime import timedelta
from collections import Counter
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from settings import BATCH_SIZE, NUM_CLASSES, VAL_SPLIT, TEST_SPLIT


class SignalDataset(Dataset):

    def __init__(self,
                 folder_path: str,
                 seconds_per_sample: int = 5,
                 rows_per_second: int = 10,
                 data_preprocessor=None,
                 background_subtraction: bool = False,
                 number_of_people: int = NUM_CLASSES,
                 sliding_window: bool = False,
                 stride: int = 1) -> None:
        """
        Custom dataset for signal classification.

        :param folder_path: Path to the folder containing .npy files.
        :param seconds_per_sample: Number of seconds per sample (default is 5).
        :param rows_per_second: Number of rows equalling one second of time in the signal.
        """
        self.data: List[np.ndarray] = []
        self.labels: np.ndarray | List[int] = []
        self.rows_per_sample: int = seconds_per_sample * rows_per_second

        background_average_signal = self.get_avg_background_sequence(folder_path=folder_path) if background_subtraction else None

        walk_user_files: List[str] = sorted([f for f in os.listdir(folder_path) if "walkarrayuser_" in f])

        if number_of_people != NUM_CLASSES: # the collected dataset has a maximum of NUM_CLASSES=20 classes (people)
            walk_user_files = random.sample(walk_user_files, min(number_of_people, len(walk_user_files)))

        for idx, file in enumerate(walk_user_files):
            signal: np.ndarray = np.load(os.path.join(folder_path, file))
            label = idx # pytorch wants the labels to be integers starting from 0, so easier to use the index as label
            # print(f"label: {label}, file: {file}")
            split_samples, labels = self.split_signal(signal, label, background_average_signal, sliding_window, stride)
            self.data.extend(split_samples)
            self.labels.extend(labels)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        # print(f"Created {len(self.data)} samples.")
        # print(f"Data shape: {self.data.shape}")
        # print(f"Labels shape: {self.labels.shape}")

        if data_preprocessor is not None:
            self.data = data_preprocessor(self.data)

    def get_avg_background_sequence(self, folder_path: str) -> np.ndarray:
        """
        Compute the average background sequence from the background files.
        :param folder_path: Path to the folder containing the background files.
        :return:
        """

        background_files: List[str] = [f for f in os.listdir(folder_path) if "backgroundarrayuser_" in f]
        background_data: List[List[int] | np.ndarray] = []

        for file in background_files:
            signal = np.load(os.path.join(folder_path, file))
            split_samples, labels = self.split_signal(signal)
            background_data.extend(split_samples)

        # background_data = np.stack(background_data, axis=0)
        background_average: np.ndarray = np.mean(background_data, axis=0) # final shape: [self.rows_per_sample, antennas]
        # print(f"Background average shape: {background_average.shape}")
        return background_average

    def split_signal(self,
                     signal: np.ndarray,
                     label: str | int = None,
                     subtract_seq: np.ndarray = None,
                     sliding_window: bool = False,
                     stride: int = 1) -> Tuple[List[np.ndarray], List[int]]:
        """
        Split the signal into chunks of rows_per_sample amount.

        :param signal: Input signal.
        :param label: Label for the signal.
        :param subtract_seq: Sequence to subtract from the signal.
        :param sliding_window: Whether to use sliding window approach.
        :param stride: Stride for the sliding window.
        :return: List of signal chunks.
        """
        samples: List[np.ndarray] = []
        labels: List[int | str] = []

        if sliding_window:
            num_steps = (len(signal) - self.rows_per_sample) // stride + 1
            for i in range(0, num_steps * stride, stride):
                sample = signal[i:i + self.rows_per_sample]
                if sample.shape[0] == self.rows_per_sample:
                    if subtract_seq is not None:
                        sample = sample - subtract_seq
                    samples.append(sample)
                    labels.append(label)
        else:
            num_samples: int = len(signal) // self.rows_per_sample
            for i in range(num_samples):
                start: int = i * self.rows_per_sample
                end: int = start + self.rows_per_sample
                sample: np.ndarray = signal[start:end]
                if sample.shape[0] == self.rows_per_sample:
                    if subtract_seq is not None:
                        sample = sample - subtract_seq
                    samples.append(sample)
                    labels.append(label)
        return samples, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label


def load_data(folder_path,
              seconds_per_sample=5,
              rows_per_second=10,
              batch_size=BATCH_SIZE,
              val_split=VAL_SPLIT,
              test_split=TEST_SPLIT,
              train_split: Optional[float] = None,
              data_preprocessor=None,
              background_subtraction: bool = False,
              number_of_people: int = NUM_CLASSES,
              verbose: int = 0,
              split_signal_stride: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare train and validation data loaders.

    :param folder_path: Path to the folder containing .npy files.
    :param seconds_per_sample: Number of seconds per sample.
    :param rows_per_second: Number of rows equalling one second of time in the signal.
    :param batch_size: Number of samples per batch.
    :param val_split: Fraction of the dataset to use for validation.
    :param test_split: Fraction of the dataset to use for testing.
    :param train_split: Fraction of the entire dataset to use for training (after reserving validation and test splits per class).
    :param data_preprocessor: Data preprocessor function.
    :param background_subtraction: Whether to subtract the background signal.
    :return: Train and validation data loaders.
    """
    if verbose > 0: print(f'Started loading data...')
    start: float = time.time()

    dataset: SignalDataset = SignalDataset(folder_path=folder_path,
                                           seconds_per_sample=seconds_per_sample,
                                           rows_per_second=rows_per_second,
                                           data_preprocessor=data_preprocessor,
                                           background_subtraction=background_subtraction,
                                           number_of_people=number_of_people,
                                           sliding_window=split_signal_stride > 0,
                                           stride=split_signal_stride)

    # the following splitting is done to ensure correct training without data leakage
    # we need to keep track of each person's (label) indices, so we know which samples belong to which person
    user_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(dataset.labels):
        if label not in user_indices:
            user_indices[label] = []
        user_indices[label].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    # sequentially split each person's data into training, validation and test
    for label, indices in user_indices.items():
        num_samples = len(indices)
        val_size = int(num_samples * val_split)
        test_size = int(num_samples * test_split)
        train_size_max = num_samples - val_size - test_size
        train_size = int(num_samples * train_split) if train_split is not None else train_size_max # only used for decreasing the training size during an experiment

        assert train_size <= train_size_max, f"Train size {train_size} is larger than available samples {num_samples - val_size - test_size} for label {label}."
        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size_max:train_size_max + val_size])
        test_indices.extend(indices[train_size_max + val_size:])

    # remove overlapping val/test samples if training used stride=1
    if split_signal_stride == 1:
        val_indices = val_indices[::rows_per_second]
        test_indices = test_indices[::rows_per_second]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if verbose > 1:
        print(f"Train size: {len(train_indices)}, Validation size: {len(val_indices)}, Test size: {len(test_indices)}")

        print("Train labels distribution:", Counter([label.item() for _, label in train_loader.dataset]))
        print("Validation labels distribution:", Counter([label.item() for _, label in val_loader.dataset]))
        print("Test labels distribution:", Counter([label.item() for _, label in test_loader.dataset]))

    if verbose > 0: print(f'Done loading data (batch size: {batch_size}) in {str(timedelta(seconds=(time.time() - start)))}.')

    return train_loader, val_loader, test_loader


def mixup_data(x, y, alpha=0.4, use_cuda=True):
    """
    Apply mixup augmentation.
    https://arxiv.org/pdf/1710.09412
    https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py

    Parameters:
    - x (Tensor): Input batch of data (e.g., [batch_size, channels, length])
    - y (Tensor): Corresponding labels
    - alpha (float): Mixup interpolation strength (default 0.4)

    Returns:
    - mixed_x (Tensor): Augmented data
    - y_a (Tensor): Original labels
    - y_b (Tensor): Mixed labels
    - lam (float): Mixing coefficient
    """

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # randomly shuffle batch indices
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # create the mixed inputs and labels
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute the mixup loss.

    Parameters:
    - criterion: Loss function (e.g., CrossEntropyLoss)
    - pred: Model predictions
    - y_a: Original labels
    - y_b: Mixed labels
    - lam: Mixing coefficient

    Returns:
    - loss: Mixup loss
    """

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def gaussian_smooth(x, kernel_size=5, sigma=1.0):
    """
    Apply 1D Gaussian smoothing to a batch of input signals.

    :param x: Input tensor of shape [batch_size, channels, time_steps]
    :param kernel_size: Size of the Gaussian kernel
    :param sigma: Standard deviation of the Gaussian distribution
    :return: Smoothed tensor
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # If input is 1D, reshape to [1, time_steps, 1] ([batch_size, time_steps, channels])
    d3_original = True
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(-1)  # Shape: [1, time_steps, 1]
        d3_original = False
    elif x.dim() == 2:
        x = x.unsqueeze(-1)  # Add a channels dimension if missing
        d3_original = False

    # swap time_steps and channels -> [batch_size, channels, time_steps]
    x = x.transpose(1, 2)

    kernel_size = min(kernel_size, x.shape[-1])

    # create a Gaussian kernel
    kernel = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-0.5 * (kernel / sigma)**2)
    kernel /= kernel.sum()

    # expand the kernel to match the number of input channels
    num_channels = x.shape[1]  # Get number of channels (30)
    kernel = kernel.view(1, 1, -1).repeat(num_channels, 1, 1).to(x.device)

    # padding to maintain the same size, ensure padding is not larger than half the input size
    padding = min(kernel_size // 2, x.shape[-1] // 2)
    x_padded = F.pad(x, (padding, padding), mode='constant', value=0)  # Use zero-padding

    # apply convolution (depthwise convolution)
    smoothed = F.conv1d(x_padded, kernel, groups=x.shape[1])

    smoothed = smoothed.transpose(1, 2)

    if not d3_original:
        smoothed = smoothed.squeeze(0)

    return smoothed


from scipy.ndimage import gaussian_filter1d

def gaussian_smooth_scipy(x, sigma=1.0):
    """
    Apply 1D Gaussian smoothing to a batch of input signals using scipy.

    :param x: Input tensor of shape [batch_size, channels, time_steps]
    :param kernel_size: Size of the Gaussian kernel
    :param sigma: Standard deviation of the Gaussian distribution
    :return: Smoothed tensor
    """

    # check if x is a torch tensor, since we need to apply operations on it
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # If input is 1D, reshape to [1, time_steps, 1] ([batch_size, time_steps, channels])
    d3_original = True
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(-1)  # Shape: [1, time_steps, 1]
        d3_original = False
    elif x.dim() == 2:
        x = x.unsqueeze(-1)  # Add a channels dimension if missing
        d3_original = False

    # swap time_steps and channels -> [batch_size, channels, time_steps]
    x = x.transpose(1, 2)

    # apply Gaussian smoothing
    smoothed = gaussian_filter1d(x, sigma=sigma, axis=-1, mode='reflect')

    smoothed = torch.tensor(smoothed, dtype=torch.float32)

    smoothed = smoothed.transpose(1, 2)

    if not d3_original:
        smoothed = smoothed.squeeze(0)

    return smoothed

def test():
    from scipy.ndimage import gaussian_filter1d
    import numpy as np
    nd_array = torch.tensor([[[1], [2], [3], [4], [5]]], dtype=torch.float32)
    print(nd_array.shape)
    print("###")
    print(gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1))
    print(gaussian_smooth(nd_array, 5, 1))
    print(gaussian_smooth_scipy(nd_array, 1))
    print("###")
    # array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
    print(gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4))
    print(gaussian_smooth(nd_array, 24, 4))
    print(gaussian_smooth_scipy(nd_array, 4))
    print("###")
    # array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
    import matplotlib.pyplot as plt
    rng = np.random.default_rng()
    x = rng.standard_normal(101).cumsum()
    print(x.shape)
    y3 = gaussian_filter1d(x, 3)
    y6 = gaussian_filter1d(x, 6)
    plt.plot(x, 'k', label='original data')
    plt.plot(y3, '--', label='filtered, sigma=3')
    plt.plot(y6, ':', label='filtered, sigma=6')
    plt.legend()
    plt.grid()
    plt.show()
    plt.cla()

    x = torch.tensor(x, dtype=torch.float32)

    y3 = gaussian_smooth(x, 18, 3)
    y6 = gaussian_smooth(x, 36, 6)
    plt.plot(x, 'k', label='original data')
    plt.plot(y3, '--', label='filtered, sigma=3')
    plt.plot(y6, ':', label='filtered, sigma=6')
    plt.legend()
    plt.grid()
    plt.show()
    plt.cla()

    y3 = gaussian_smooth_scipy(x, 3)
    y6 = gaussian_smooth_scipy(x, 6)
    plt.plot(x, 'k', label='original data')
    plt.plot(y3, '--', label='filtered, sigma=3')
    plt.plot(y6, ':', label='filtered, sigma=6')
    plt.legend()
    plt.grid()
    plt.show()
    plt.cla()
