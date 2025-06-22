# Project Structure

```
root/ 
│
├── _misc/ # Contains setup photos and participants' statistics
│   └── ... # setup photos, participants' statistics, etc
│   
├── csi_collection/ # Contains everything related to CSI data collection
│   ├── input/ # input folder for the csi_parsing.py file
│   ├── output/ # output folder for the csi_parsing.py file
│   ├── csi_collection.py # script for collecting CSI data from the router (ran after router_setup.py)
│   ├── csi_parsing.py # manual CSI data parsing script (not really used)
│   ├── csi_processing_pipeline.py # CSI data processing pipeline (aligning/synchronizing 5GHz and 60GHz data)
│   ├── readme.md # short documentation of steps for the router setup
│   └── router_setup.py # script for the router setup for CSI data collection
│
├── data/ # Contains the external dataset and our collected dataset
│   ├── collected_csi_data_original/ # raw collected CSI data
│   │   ├── 5ghz/ # raw 5GHz CSI data
│   │   │   ├── background/ # background data
│   │   │   └── walking/ # walking data, contains subfolders for each participant
│   │   └── 60ghz/ # raw 60GHz CSI data (same structure as 5GHz)
│   │
│   ├── collected_csi_data_original_processed/ # processed (aligned/synchronized) collected CSI data
│   │   ├── 5ghz/ # processed 5GHz CSI data (10 packets per second), contains .npy files
│   │   ├── 5ghz_200hz/ # processed 5GHz CSI data (200 packets per second), contains .npy files
│   │   └── 60ghz/ # processed 60GHz CSI data (10 packets per second), contains .npy files
│   │
│   └── external_data_combined/ # external dataset (60GHz data only)
│
├── output/ # Contains all output files from the project experiments
│   ├── gridsearch/ # gridsearch results for all models, per seed
│   ├── plots/ # plots for the experiments, including training/validation loss and accuracy
│   ├── trained_models.py # saved trained models
│   └── ... # averaged gridsearch results (averaged over all seeds)
│
├── src/ 
│   ├── models/ # Implemented model architectures 
│   ├── _main_exp_<experiment_type>.ipynb # main experiment notebook for each (special) experiment type (3 files in total)
│   ├── _main_gridsearch_<signal_type>.ipynb # main gridsearch notebook for each signal type (5GH@10Hz, 5GH@200Hz, collected 60GHz, external 60GHz)
│   ├── dataset.py # dataset class for loading CSI data
│   ├── helper_functions.py # helper functions for the project, including training and validation loops
│   ├── preprocess.py # file containing all data preprocessing classes (not really used eventually)
│   └── settings.py # settings file for the project, including paths and default parameters
│   
└── requirements.txt # Python package requirements for the project
```

---

## Dataset Details
The data/ directory contains all raw and processed CSI data used in this project. 
Below is a breakdown of what each folder represents:

### **collected_csi_data_original/**: <br>
This folder contains the raw CSI recordings from our own multi-modal dataset.
- Frequency bands: Both 5 GHz (sub-6) and 60 GHz (mmWave)
- Participants: Data from 20 participants, collected over 3 days (the filenames contain the exact dates)
- Procedure: Each participant walked continuously for 2 minutes, following a round-trip walking path (walk back and forth)

Subfolders:

- `5ghz/`
  - `walking/`: Raw CSI per participant (300 packets/sec (300 Hz) during collection)
  - `background/`: Static samples (no person present), recorded only on day 3
- `60ghz/` (Identical structure to 5 GHz, sampled at ~10 packets/sec (firmware limited))

### **collected_csi_data_original_processed/**: <br>
This contains the aligned, synchronized, and resampled CSI data, used directly for training/testing.

Subfolders:

- `5ghz/`: Downsampled to 10 Hz to match mmWave sampling rate; 52 subcarrier values
- `5ghz_200hz/`: High-rate 5 GHz CSI at 200 Hz, available for 18 participants
  - 2 participants had low original sampling rate (~30 Hz), so were excluded here
- `60ghz/`: 10 Hz; 60 antenna values per frame (30+30 antennas because of the X-formation of the 4 devices)

All processed files are stored in .npy format

### **external_data_combined**/: <br>
This folder contains a private mmWave-only dataset collected independently on 60 GHz hardware only.
This dataset is used primarily to evaluate the potential of mmWave CSI in a different environment.

- Participants: 7 individuals
- Sampling rate: 22 Hz
- CSI format: Amplitude-only, 30 antennas per frame
- Data: Includes walking trajectories and background samples

---

## Output details

The `output/` directory contains all results from the experiments, including trained models, gridsearch results, and plots.
If a folder is empty, this is because they were too large to be included in the repository.

Each folder, except for the `gridsearch/` folder, contains the results for all signal types:
- `5ghz_10hz/`
- `5ghz_200hz/`
- `60ghz_collected/`
- `60ghz_external/`

Aside from that, the `plots/` folder contains all plots from the 3 special experiments (learning curve analysis, varying people analysis and cross-session evaluation).

The naming convention for the files w.r.t. the used model, parameters, etc. is as follows:

```python
model_args_str = "_".join(safe_str(v) for v in param_dict['model_args'].values())
train_name_id = f'{signal_type}_{model.__class__.__name__}_{param_dict['batch_size']}_{optimizer_name}_{lr}_{model_args_str}_{stopped_at_epoch}-{num_epochs}_{mixup_alpha}_{smoothing_prob}_{used_seed}_{str(int(background_subtraction))}_{seconds_per_sample}'
```

The above is taken from the training loop function (`train_model(...)`) in `helper_functions.py` file.
