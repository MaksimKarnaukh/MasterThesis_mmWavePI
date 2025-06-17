import torch
import time
import random
import os
from datetime import timedelta
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from typing import List, Tuple, Dict, Optional
from itertools import product
import pandas as pd

from dataset import load_data, mixup_data, mixup_criterion, gaussian_smooth
from settings import NUM_CLASSES, CRITERION, output_path, OPTIMIZER, LR, MIXUP_ALPHA, SMOOTHING_KERNEL_SIZE, SMOOTHING_SIGMA, NUM_EPOCHS, DEVICE, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT, EARLY_STOPPING_PATIENCE, SMOOTHING_PROBABILITY


def extract_signal_type(folder_path: str) -> str:
    """
    Extracts the signal type from a folder path.
    Returns '5ghz_10hz' if the path ends with '5ghz', otherwise uses the folder name.
    """
    folder_name: str = os.path.basename(os.path.normpath(folder_path))
    if folder_name.lower() == '5ghz':
        return '5ghz_10hz'
    elif folder_name.lower() == '60ghz':
        return '60ghz_collected'
    elif folder_name.lower() == 'external_data_combined':
        return '60ghz_external'
    return folder_name.lower()

def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    :param model: PyTorch model
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.
    https://pytorch.org/docs/stable/notes/randomness.html
    :param seed: seed for everything
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model_to_file(model, model_name: str, output_dir: str = output_path) -> None:
    """
    Save the model to a specified directory.
    :param model: PyTorch model to save
    :param model_name: Name of the model file
    :param output_dir: Directory to save the model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, model_name))

def load_model_from_file(model, model_name: str, output_dir: str = output_path) -> None:
    """
    Load the model from a specified directory.
    :param model: PyTorch model to load into
    :param model_name: Name of the model file
    :param output_dir: Directory to load the model from
    """
    model.load_state_dict(torch.load(os.path.join(output_dir, model_name)))

def safe_str(val):
    """
    Convert a value to a safe string representation.
    Useful for torch.nn.Module objects.
    :param val: Value/Object to convert
    :return:
    """
    if isinstance(val, type):
        return val.__name__
    elif isinstance(val, torch.nn.Module):
        return val.__class__.__name__
    else:
        return str(val)

def plot_loss_and_accuracy(train_loss_list_per_epoch,
                           val_loss_list,
                           val_accuracy_per_epoch,
                           train_accuracy_per_epoch,
                           train_name_id,
                           plot_output_path=output_path):
    """
    Function to plot the training and validation loss and accuracy curves.
    :param train_loss_list_per_epoch: training loss per epoch
    :param val_loss_list: validation loss per epoch
    :param val_accuracy_per_epoch: validation accuracy per epoch
    :param train_accuracy_per_epoch: training accuracy per epoch
    :param train_name_id: filename identifier for the training run
    :param plot_output_path: output path for the plots
    :return:
    """

    # plotting the loss curve over all epochs
    plt.plot(np.arange(len(train_loss_list_per_epoch)), train_loss_list_per_epoch, color='blue', label='Train Loss')
    plt.plot(np.arange(len(val_loss_list)), val_loss_list, color='red', label='Validation Loss')
    plt.legend()
    plt.title('Train and Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_output_path + f'train_val_loss_{train_name_id}.png')
    # plt.show()
    # plt.cla()
    plt.close()

    # plotting the accuracy curve over all epochs
    plt.plot(np.arange(len(train_accuracy_per_epoch)), train_accuracy_per_epoch, label='Train Accuracy', color='blue')
    plt.plot(np.arange(len(val_accuracy_per_epoch)), val_accuracy_per_epoch, label='Validation Accuracy', color='green')
    plt.title('Train and Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_output_path + f'train_val_accuracy_{train_name_id}.png')
    # plt.show()
    plt.close()


def train_model(model,
                param_dict,
                train_loader,
                val_loader,
                device,
                output_dir=output_path,
                verbose=1,
                mixup=False,
                smoothing_prob=0.0,
                signal_type='5ghz_10hz',
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                background_subtraction: bool = False,
                used_seed: int = None,
                save_model: bool = True,
                seconds_per_sample: float = 5) -> Tuple[torch.nn.Module, float, float]:
    """
    Function to train the model.
    :param model: Model to train
    :param param_dict: Dictionary of hyperparameters and their explicit values for this training run
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param output_path: Path to save the plots
    :param verbose: whether to print extra output, 0 for none, 1 for some, 2 for all
    :return: Trained model
    """

    print(f'\n(Device: {device}) Started training model: ' + model.__class__.__name__)

    model = model.to(device)
    criterion = CRITERION # for torch.nn.CrossEntropyLoss(): "This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class."

    # extract hyperparameters
    optimizer_name = param_dict.get('optimizer', OPTIMIZER)
    lr = param_dict.get('learning_rate', LR)
    num_epochs = param_dict.get('num_epochs', NUM_EPOCHS)
    mixup_alpha = param_dict.get('mixup_alpha', MIXUP_ALPHA)
    smoothing_kernel_size = param_dict.get('smoothing_kernel_size', SMOOTHING_KERNEL_SIZE)
    smoothing_sigma = param_dict.get('smoothing_sigma', SMOOTHING_SIGMA)
    optimizer = {
        'adam': torch.optim.Adam(model.parameters(), lr=lr),
        'sgd': torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    }.get(optimizer_name)

    # lists to store loss and accuracy values
    train_loss_list_per_epoch, val_loss_list_per_epoch = [], []
    val_accuracy_per_epoch, train_accuracy_per_epoch = [], []

    # early stopping variables
    best_val_loss = float('inf')
    best_val_acc: float = 0.0
    epochs_without_improvement: int = 0
    best_model_state = None
    stopped_at_epoch: int = 0

    time_s: float = time.time() # to measure total training time

    for epoch in range(num_epochs):
        start: float = time.time() # to measure epoch time

        model.train()
        train_loss_list_per_itr: List[float] = []
        # all_preds, all_labels = [], []

        correct = 0.0
        total = 0

        for batch_idx, (signals, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            signals, labels = signals.to(device), labels.to(device)

            # if torch.isnan(signals).any() or torch.isinf(signals).any():
            #     raise "NaN or Inf detected in signals"

            if smoothing_prob > 0.0 and np.random.rand() < smoothing_prob:
                signals = gaussian_smooth(signals, kernel_size=smoothing_kernel_size, sigma=smoothing_sigma)

            if mixup:
                signals, labels_a, labels_b, lam = mixup_data(signals, labels, mixup_alpha)
                outputs = model(signals).to(device)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                preds = torch.argmax(outputs, dim=1)

                # all_preds.append(preds)
                # all_labels.append(labels_a)

                correct += (lam * preds.eq(labels_a).sum().item() +
                            (1 - lam) * preds.eq(labels_b).sum().item())
                total += labels.size(0)
            else:
                outputs = model(signals).to(device)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                # all_preds.append(preds)
                # all_labels.append(labels)

                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            loss.backward()
            optimizer.step()

            train_loss_list_per_itr.append(loss.item())

        train_loss_list_per_epoch.append(np.mean(train_loss_list_per_itr))
        train_accuracy_per_epoch.append(correct / total)

        # train_preds = torch.cat(all_preds).cpu()
        # train_targets = torch.cat(all_labels).cpu()
        # train_acc = (train_preds == train_targets).float().mean().item()
        # train_accuracy_per_epoch.append(train_acc)

        eval_loss, eval_acc = evaluate_model(model, val_loader, criterion, device)
        val_loss_list_per_epoch.append(eval_loss)
        val_accuracy_per_epoch.append(eval_acc)

        stopped_at_epoch = epoch + 1
        if eval_loss < best_val_loss:
            best_val_loss = eval_loss
            best_val_acc = eval_acc
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement in {early_stopping_patience} epochs)")
                break

        end: float = time.time()

        if verbose == 3:
            print(
                f'Epoch {epoch + 1}/{num_epochs} ({(end - start):.2f} seconds) - Train Loss: {train_loss_list_per_epoch[-1]:.4f}, Val Loss: {eval_loss:.4f}, Val Acc: {eval_acc:.4f}')

    if best_model_state is not None: # load the best model state
        model.load_state_dict(best_model_state)

    print("Training time: ", str(timedelta(seconds=(time.time() - time_s))))

    model_args_str = "_".join(safe_str(v) for v in param_dict['model_args'].values())
    train_name_id = f'{signal_type}_{model.__class__.__name__}_{param_dict['batch_size']}_{optimizer_name}_{lr}_{model_args_str}_{stopped_at_epoch}-{num_epochs}_{mixup_alpha}_{smoothing_prob}_{used_seed}_{str(int(background_subtraction))}_{seconds_per_sample}'

    if verbose >= 1:
        plot_output_path = output_dir + f'plots/{signal_type}/'
        plot_loss_and_accuracy(train_loss_list_per_epoch, val_loss_list_per_epoch, val_accuracy_per_epoch, train_accuracy_per_epoch, train_name_id, plot_output_path)

    if save_model:
        save_model_to_file(model, f'{train_name_id}.pt', output_dir=output_dir + f'trained_models/{signal_type}/')

    return model, max(val_accuracy_per_epoch), best_val_acc


def evaluate_model(model, validation_loader, criterion, device, plot_confusion_matrix=False):
    """
    Function to evaluate the model.

    :param model: Model to evaluate
    :param validation_loader: DataLoader for validation data
    :param criterion: Loss function
    :param device: Device
    :param plot_confusion_matrix: Whether to show a plot of the confusion matrix
    :return: Mean loss and accuracy
    """
    model.eval()
    model = model.to(device)

    val_loss: List[float] = []
    real_label = None
    pred_label = None
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())

            _, preds = torch.max(outputs, 1)
            if real_label is None:
                real_label = labels.data
                pred_label = preds
            else:
                real_label = torch.cat((real_label, labels.data), dim=0)
                pred_label = torch.cat((pred_label, preds), dim=0)
            del inputs
            del labels

    real_label = real_label.detach().cpu().numpy()
    pred_label = pred_label.detach().cpu().numpy()

    if plot_confusion_matrix:
        cm: np.ndarray = confusion_matrix(real_label, pred_label, labels=np.arange(NUM_CLASSES))
        disp: ConfusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    report: str = classification_report(real_label, pred_label, zero_division=0, labels=np.arange(NUM_CLASSES))
    eval_acc: float = float(report.split('accuracy')[1].split(' ')[27])

    return np.mean(val_loss), eval_acc


def grid_search(param_grid,
                folder_path,
                device = DEVICE,
                background_subtraction=False,
                train_verbose=1,
                seconds_per_sample=3,
                rows_per_second=10,
                num_classes: int = NUM_CLASSES,
                val_split: float = VAL_SPLIT,
                test_split: float = TEST_SPLIT,
                used_seed: int = None,
                output_dir: str = output_path,
                filename: str = None,
                split_signal_stride: int = 0) -> pd.DataFrame:
    """
    Perform a grid search for hyperparameter optimization.

    :param param_grid: Dictionary of parameter lists to search.
    :param folder_path: Path to the dataset folder.
    :return: DataFrame summarizing results for each configuration.
    """
    results = []
    param_combinations = list(product(*[param_grid[k] for k in param_grid if k not in ['batch_size', 'model']]))
    total_combinations = len(param_grid['batch_size']) * len(param_grid['model']) * len(param_combinations)
    print(f"Running {total_combinations} configurations...")

    counter = 0

    for model_config in param_grid['model']:

        model_class = model_config['model_class']
        model_args = model_config['model_args']

        data_preprocessor = model_config.get('data_preprocessor', None)

        print(f"\nProcessing Model: {model_class.__name__} with params {model_args}")

        for batch_size in param_grid['batch_size']: # in case we parametrize batch size

            # Load data once for the current batch size
            train_loader, val_loader, test_loader = load_data(
                folder_path=folder_path,
                seconds_per_sample=seconds_per_sample,
                rows_per_second=rows_per_second,
                batch_size=batch_size,
                val_split=val_split,
                test_split=test_split,
                data_preprocessor=data_preprocessor,
                background_subtraction=background_subtraction,
                verbose=2,
                split_signal_stride=split_signal_stride,
            )

            for params in param_combinations:

                # unpack current parameters
                current_params = dict(zip([k for k in param_grid if k != 'batch_size'], params))

                current_params['model_args'] = model_args
                current_params['batch_size'] = batch_size
                current_params['num_epochs'] = model_config['num_epochs']

                learning_rate = current_params.get('learning_rate', 0.001)
                optimizer_name = current_params.get('optimizer', 'adam')
                mixup_alpha = current_params.get('mixup_alpha', 0.0)
                smoothing_prob = current_params.get('smoothing_prob', 0.0)

                model = model_class(**model_args, num_classes=num_classes)
                num_params = count_parameters(model)

                print(f"\nTesting Configuration {counter}: batch_size={batch_size}, lr={learning_rate}, optimizer={optimizer_name}, mixup_alpha={mixup_alpha}, smoothing_prob={smoothing_prob}")

                signal_type = extract_signal_type(folder_path) if 'external' not in folder_path else '60ghz_external'

                try:
                    best_model, best_accuracy, final_accuracy = train_model(
                        model=model,
                        param_dict=current_params,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        mixup=True if mixup_alpha > 0 else False,
                        smoothing_prob=smoothing_prob,
                        verbose=train_verbose,
                        signal_type=signal_type,
                        background_subtraction=background_subtraction,
                        used_seed=used_seed,
                        seconds_per_sample=seconds_per_sample,
                    )

                    if test_loader is not None:
                        test_loss, test_accuracy = evaluate_model(best_model, test_loader, CRITERION, device)
                    else:
                        test_accuracy = None

                except RuntimeError as e: # mostly handles certain models that can't handle gaussian blurring because of their signal preprocessing requirements
                    print(f"RuntimeError: {e}")
                    test_accuracy = None
                    best_accuracy = None
                    final_accuracy = None

                results.append({
                    # 'signal_type': signal_type,
                    'seed': used_seed,
                    'batch_size': batch_size,
                    'lr': learning_rate,
                    'optimizer': optimizer_name,
                    'model': model_class.__name__,
                    'model_params': str(model_args),
                    'mixup_alpha': mixup_alpha,
                    'smoothing_prob': smoothing_prob,
                    'best_val_acc': best_accuracy,
                    'final_val_acc': final_accuracy,
                    'test_acc': test_accuracy,
                    'num_params': num_params
                })

                counter += 1

    assert counter == total_combinations

    results_df = pd.DataFrame(results)

    if filename:
        os.makedirs(output_dir, exist_ok=True)
        filename = filename + f'_sps{seconds_per_sample}_bgsub{str(int(background_subtraction))}_{used_seed}.csv'
        full_path = os.path.join(output_dir, filename)
        results_df.to_csv(full_path, index=False)

    return results_df

### Functions for learning curves and varying people experiments ###

def get_learning_curve_data(model_class,
                            model_args,
                            folder_path,
                            device,
                            train_splits: List[float],
                            seconds_per_sample: int = 5,
                            rows_per_second: int = 10,
                            batch_size: int = BATCH_SIZE,
                            num_epochs: int = NUM_EPOCHS,
                            optimizer_name: str = OPTIMIZER,
                            learning_rate: float = LR,
                            smoothing_prob: float = SMOOTHING_PROBABILITY,
                            mixup: bool = True,
                            data_preprocessor=None,
                            background_subtraction: bool = True,
                            val_split: float = VAL_SPLIT,
                            test_split: float = TEST_SPLIT,
                            signal_type='5ghz_10hz',
                            used_seed: int = None) -> Tuple[List[float], List[float], List[float]]:
    """
    Runs training for various validation splits and returns training sizes and corresponding accuracies.
    """
    val_accuracies = []
    test_accuracies = []
    train_sizes = []

    for train_split in train_splits:
        print(f"\n--- Training split {train_split:.2f} ---")
        train_loader, val_loader, test_loader = load_data(
            folder_path=folder_path,
            seconds_per_sample=seconds_per_sample,
            rows_per_second=rows_per_second,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            train_split=train_split,
            data_preprocessor=data_preprocessor,
            background_subtraction=background_subtraction
        )

        model = model_class(**model_args, num_classes=NUM_CLASSES)
        param_dict = {
            'optimizer': optimizer_name,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'model_args': model_args,
        }

        best_model, best_accuracy, final_val_acc = train_model(
            model=model,
            param_dict=param_dict,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            verbose=0,
            mixup=mixup,
            smoothing_prob=smoothing_prob,
            signal_type=signal_type,
            background_subtraction=background_subtraction,
            used_seed=used_seed,
            seconds_per_sample=seconds_per_sample,
            save_model=False
        )

        test_accuracy = evaluate_model(best_model, test_loader, CRITERION, device)[1] if test_loader is not None else None

        val_accuracies.append(final_val_acc)
        test_accuracies.append(test_accuracy)
        train_sizes.append(train_split)

    return train_sizes, val_accuracies, test_accuracies

def plot_learning_curves(results_dict: Dict[str, Tuple[List[float], List[float]]],
                         std_dict: Optional[Dict[str, List[float]]] = None,
                         output_path: str = output_path,
                         filename: str = "learning_curves",
                         plot_title: str = "Accuracy vs. Training Size",
                         save_plot: bool = False):
    """
    Plots learning curves with shaded error bands for multiple configurations.

    :param results_dict: Dictionary where keys are labels (e.g., "5GHz") and values are (train_sizes, accuracies)
    :param std_dict: Optional dictionary with standard deviations matching keys in results_dict
    """
    plt.figure(figsize=(6.4, 4.8))

    for label, (train_sizes, accuracies) in results_dict.items():
        accuracies = np.array(accuracies)
        train_sizes = np.array(train_sizes)
        stds = np.array(std_dict[label]) if std_dict and label in std_dict else None

        plt.plot(train_sizes, accuracies, marker='o', label=label)
        if stds is not None:
            plt.fill_between(train_sizes,
                             accuracies - stds,
                             accuracies + stds,
                             alpha=0.05)

    plt.title(plot_title)
    plt.xlabel("Training Size (absolute fraction)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend(loc='lower right',
               framealpha=1,
               fontsize='x-large')
    plt.tight_layout()
    if save_plot:
        plt.savefig(output_path + filename + ".png")
    plt.show()

def get_varying_people_data(model_class,
                            model_args,
                            folder_path,
                            device,
                            people_counts,
                            batch_size=32,
                            num_epochs=NUM_EPOCHS,
                            val_split=0.15,
                            test_split=0.15,
                            optimizer_name=OPTIMIZER,
                            learning_rate=LR,
                            smoothing_prob=SMOOTHING_PROBABILITY,
                            mixup=True,
                            data_preprocessor=None,
                            background_subtraction=True,
                            signal_type='5ghz_10hz',
                            seconds_per_sample=5,
                            rows_per_second=10,
                            used_seed: int = None) -> Tuple[List[int], List[float], List[float]]:

    val_accuracies = []
    test_accuracies = []
    people_used = []

    for num_people in people_counts:
        print(f"\n==== Running for {num_people} people ====")

        train_loader, val_loader, test_loader = load_data(
            folder_path=folder_path,
            seconds_per_sample=seconds_per_sample,
            rows_per_second=rows_per_second,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            data_preprocessor=data_preprocessor,
            background_subtraction=background_subtraction,
            number_of_people=num_people
        )

        model = model_class(**model_args, num_classes=num_people)

        param_dict = {
            'optimizer': optimizer_name,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'model_args': model_args
        }

        best_model, best_val_acc, final_val_acc = train_model(
            model,
            param_dict,
            train_loader,
            val_loader,
            device,
            verbose=0,
            mixup=mixup,
            smoothing_prob=smoothing_prob,
            signal_type=signal_type,
            background_subtraction=background_subtraction,
            used_seed=used_seed,
            seconds_per_sample=seconds_per_sample,
            save_model=False
        )

        test_accuracy = evaluate_model(best_model, test_loader, CRITERION, device)[1] if test_loader is not None else None

        people_used.append(num_people)
        val_accuracies.append(final_val_acc)
        test_accuracies.append(test_accuracy)

    return people_used, val_accuracies, test_accuracies

def plot_people_vs_accuracy(results_dict: Dict[str, Tuple[List[int], List[float]]],
                            std_dict: Optional[Dict[str, List[float]]] = None,
                            output_path: str = output_path,
                            filename: str = "people_vs_accuracy.png",
                            plot_title: str = "Accuracy vs. Number of People",
                            save_plot: bool = False):
    """
    Plots bar plots comparing accuracy vs number of people for multiple signal types.

    :param results_dict: Dictionary where keys are signal labels (e.g., "5GHz", "60GHz") and values are lists of (people_count, accuracy) tuples
    """
    signal_types = list(results_dict.keys())
    num_signal_types = len(signal_types)

    people_counts = results_dict[signal_types[0]][0]
    num_people_counts = len(people_counts)

    bar_width = 0.6 / num_signal_types # narrower bars if more signal types
    indices = np.arange(num_people_counts)

    plt.figure(figsize=(6.4, 5.6))

    for i, (signal_type, results) in enumerate(results_dict.items()):
        people_used, accuracies = results
        stds = std_dict[signal_type] if std_dict and signal_type in std_dict else None
        bar_pos = indices + i * bar_width
        plt.bar(bar_pos, accuracies, width=bar_width-0.05, yerr=stds, label=signal_type, capsize=5)

    plt.title(plot_title)
    plt.xlabel('Number of People')
    plt.ylabel('Accuracy')
    plt.xticks(indices + bar_width * (num_signal_types - 1) / 2, people_counts)
    plt.ylim(0, 1.1)
    # plt.yticks(np.linspace(0, 1.0, 5))
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.grid(axis='y')
    # plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='lower left',
               framealpha=1,
               fontsize='x-large')

    plt.tight_layout()
    if save_plot:
        plt.savefig(output_path + filename)

    plt.show()

def run_with_seeds(func, func_args: dict, n_seeds: int = 5) -> list:
    """
    Runs the given function multiple times with different random seeds.

    Additional material:
    https://stackoverflow.com/questions/10900852/may-near-seeds-in-random-number-generation-give-similar-random-numbers

    :param func: Function to run.
    :param func_args: Arguments to the function.
    :param n_seeds: Number of seeds to run with.
    :return: List of outputs from the function.
    """
    outputs = []
    seeds = [42, 420, 101010, 119, 1234, 1337, 123456789, 987654321, 999999999, 12]

    for seed in seeds[:n_seeds]:
        set_seed(seed)
        func_args['used_seed'] = seed

        print(f"\n--- Running seed {seed} ---")
        output = func(**func_args)
        outputs.append(output)

    return outputs

def run_experiment_with_seeds(experiment_func,
                              signal_configs: Dict[str, Dict],
                              varying_param_name: str,
                              varying_param_value,
                              n_seeds: int = 5,
                              accuracy_type: int = 2,
                              seconds_per_sample: int = 5,
                              ) -> Tuple[Dict[str, Tuple[List[float], List[float]]], Dict[str, List[float]]]:
    """
    Run a generic experiment function across multiple signals and seeds, and compute mean/std over seeds.

    :param experiment_func: Function to run. Must return (x_values, y_values).
    :param signal_configs: Dictionary mapping signal name to a dict of kwargs for the experiment_func.
    :param varying_param_name: Name of the argument that changes per experiment (e.g., 'train_splits', 'people_counts').
    :param n_seeds: Number of seeds to run for averaging.
    :return: Tuple of (results_per_signal, stds_per_signal)
    """
    results_per_signal = {}
    stds_per_signal = {}

    for signal, config in signal_configs.items():
        print(f"Running experiment for {signal}...")

        func_args = {
            'model_class': config['model_class'],
            'model_args': config['model_args'],
            'folder_path': config['folder_path'],
            'device': DEVICE,
            varying_param_name: varying_param_value,
            'num_epochs': config['num_epochs'],
            'background_subtraction': config['background_subtraction'],
            'rows_per_second': config['rows_per_second'],
            'seconds_per_sample': seconds_per_sample,
            'signal_type': signal,
        }

        seed_outputs = run_with_seeds(
            func=experiment_func,
            func_args=func_args,
            n_seeds=n_seeds
        )

        print("seed outputs: ", seed_outputs)

        x_values = seed_outputs[0][0]
        all_y_values = np.stack([out[accuracy_type] for out in seed_outputs], axis=0)  # shape: [n_seeds, len(x_values)]

        mean_y = np.mean(all_y_values, axis=0)
        std_y = np.std(all_y_values, axis=0)

        results_per_signal[signal] = (x_values, mean_y.tolist())
        stds_per_signal[signal] = std_y.tolist()

    return results_per_signal, stds_per_signal

def run_gridsearch_with_seeds(gridsearch_func,
                              gridsearch_args: dict,
                              n_seeds: int = 5,
                              output_dir: str = output_path,
                              filename: str = None
                              ) -> pd.DataFrame:
    """
    Run the grid search function multiple times with different random seeds.

    :param gridsearch_func: Function to run.
    :param gridsearch_args: Arguments to the function.
    :param n_seeds: Number of seeds to run with.
    :return: DataFrame with averaged results across seeds.
    """

    gridsearch_args['filename'] = filename
    gridsearch_args['output_dir'] = output_dir + "gridsearch/"

    seed_outputs = run_with_seeds(
        func=gridsearch_func,
        func_args=gridsearch_args,
        n_seeds=n_seeds
    )

    dfs = seed_outputs
    combined = pd.concat(dfs)

    if 'seed' in combined.columns:
        combined = combined.drop(columns=['seed']) # drop the seed column because it is unique per dataframe, meaning it will mess with the grouping

    metric_cols = ['best_val_acc', 'final_val_acc', 'test_acc']
    group_cols = [col for col in combined.columns if col not in metric_cols] # list(combined.columns.difference(metric_cols))

    # compute both mean and std for each metric
    grouped = combined.groupby(group_cols).agg({
        'best_val_acc': ['mean', 'std'],
        'final_val_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std']
    })

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    # reorder: group columns first, then metrics (mean + std)
    metric_order = []
    for metric in metric_cols:
        metric_order.extend([f"{metric}_mean", f"{metric}_std"])
    final_columns = group_cols + metric_order
    grouped = grouped[final_columns]

    grouped[metric_order] = grouped[metric_order].round(5)

    if filename:
        os.makedirs(output_dir, exist_ok=True)
        filename = filename + f'_avg{n_seeds}_bgsub{gridsearch_args["background_subtraction"]}_sps{gridsearch_args['seconds_per_sample']}.csv'
        full_path = os.path.join(output_dir, filename)
        grouped.to_csv(full_path, index=False)

    return grouped


def evaluate_on_individual_file(model_class,
                                model_args,
                                model_path: str,
                                npy_file_path: str,
                                label_index: int,
                                device,
                                background_subtraction: bool = False,
                                data_preprocessor=None,
                                seconds_per_sample: int = 5,
                                rows_per_second: int = 10):
    """
    Loads a trained model and evaluates it on a single person's signal file.

    :param model_class: Class of the model to load
    :param model_args: Arguments to instantiate the model
    :param model_path: Path to the saved .pt model
    :param npy_file_path: Path to the .npy file of the second-day recording
    :param label_index: Expected class index for the person (e.g., 14)
    :param device: Torch device
    :return: Accuracy on this specific file
    """

    signal = np.load(npy_file_path)

    rows_per_sample = seconds_per_sample * rows_per_second
    num_samples = len(signal) // rows_per_sample

    folder_path = os.path.dirname(npy_file_path)
    subtract_seq = None
    if background_subtraction:
        def get_avg_background_sequence(folder_path: str) -> np.ndarray:
            background_files = [f for f in os.listdir(folder_path) if "backgroundarrayuser_" in f]
            print(f"Background files: {background_files}")
            background_data = []
            for file in background_files:
                raw_signal = np.load(os.path.join(folder_path, file))
                num_segments = len(raw_signal) // rows_per_sample
                for i in range(num_segments):
                    segment = raw_signal[i * rows_per_sample:(i + 1) * rows_per_sample]
                    if segment.shape[0] == rows_per_sample:
                        background_data.append(segment)
            return np.mean(background_data, axis=0) if background_data else None

        subtract_seq = get_avg_background_sequence(folder_path)

    samples = []
    for i in range(num_samples):
        segment = signal[i * rows_per_sample:(i + 1) * rows_per_sample]
        if segment.shape[0] == rows_per_sample:
            if subtract_seq is not None:
                segment = segment - subtract_seq
            samples.append(segment)

    samples = np.array(samples)

    if data_preprocessor is not None:
        samples = data_preprocessor(samples)

    samples_tensor = torch.tensor(samples, dtype=torch.float32)
    labels_tensor = torch.tensor([label_index] * len(samples), dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(samples_tensor, labels_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model = model_class(**model_args)
    load_model_from_file(model, model_path)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    _, acc = evaluate_model(model, loader, criterion, device, plot_confusion_matrix=True)

    print(f"Accuracy on {os.path.basename(npy_file_path)}: {acc:.4f}")
    return acc

def evaluate_on_multiple_files(model_class,
                                model_args,
                                model_path: str,
                                file_label_pairs: list,
                                device,
                                background_subtraction: bool = False,
                                data_preprocessor=None,
                                seconds_per_sample: int = 5,
                                rows_per_second: int = 10):
    """
    Loads a trained model and evaluates it on multiple .npy signal files, combining their predictions into one confusion matrix.

    :param model_class: Class of the model to load
    :param model_args: Arguments to instantiate the model
    :param model_path: Path to the saved .pt model
    :param file_label_pairs: List of tuples (npy_file_path, label_index)
    :param device: Torch device
    :return: Combined accuracy across all files
    """
    rows_per_sample = seconds_per_sample * rows_per_second
    all_samples = []
    all_labels = []

    for npy_file_path, label_index in file_label_pairs:
        signal = np.load(npy_file_path)
        num_samples = len(signal) // rows_per_sample

        folder_path = os.path.dirname(npy_file_path)
        subtract_seq = None
        if background_subtraction:
            def get_avg_background_sequence(folder_path: str) -> np.ndarray:
                background_files = [f for f in os.listdir(folder_path) if "backgroundarrayuser_" in f]
                background_data = []
                for file in background_files:
                    raw_signal = np.load(os.path.join(folder_path, file))
                    num_segments = len(raw_signal) // rows_per_sample
                    for i in range(num_segments):
                        segment = raw_signal[i * rows_per_sample:(i + 1) * rows_per_sample]
                        if segment.shape[0] == rows_per_sample:
                            background_data.append(segment)
                return np.mean(background_data, axis=0) if background_data else None

            subtract_seq = get_avg_background_sequence(folder_path)

        for i in range(num_samples):
            segment = signal[i * rows_per_sample:(i + 1) * rows_per_sample]
            if segment.shape[0] == rows_per_sample:
                if subtract_seq is not None:
                    segment = segment - subtract_seq
                all_samples.append(segment)
                all_labels.append(label_index)

    samples = np.array(all_samples)
    labels = np.array(all_labels)

    if data_preprocessor is not None:
        samples = data_preprocessor(samples)

    samples_tensor = torch.tensor(samples, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(samples_tensor, labels_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model = model_class(**model_args)
    load_model_from_file(model, model_path)
    model = model.to(device)

    # Evaluate all at once with shared confusion matrix
    criterion = torch.nn.CrossEntropyLoss()
    _, acc = evaluate_model(model, loader, criterion, device, plot_confusion_matrix=True)

    print(f"Combined accuracy over {len(file_label_pairs)} files: {acc:.4f}")
    return acc

