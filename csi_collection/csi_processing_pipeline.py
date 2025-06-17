"""
This script processes collected 5GHz and 60GHz CSI data, synchronizing them based on timestamps and aligning them.
"""

import os
import numpy as np
import re
import csv
import math
from datetime import datetime, timedelta
from CSIKit.reader import get_reader
from CSIKit.util import csitools
import struct
from scapy.all import rdpcap, wrpcap
from collections import defaultdict
import pandas as pd
from typing import List, Tuple, Union, Optional

# input data folders
data_folder_5ghz_background = "./data/collected_csi_data_original/csi_data_5ghz/background/" # 5GHz background data (no person present)
data_folder_60ghz_background = "./data/collected_csi_data_original/csi_data_60ghz/background/" # 60GHz background data (no person present)
data_folder_5ghz_walking = "./data/collected_csi_data_original/csi_data_5ghz/walking/" # 5GHz data of a person walking
data_folder_60ghz_walking = "./data/collected_csi_data_original/csi_data_60ghz/walking/" # 60GHz data of a person walking
# general output folder
output_folder = "./data/collected_csi_data_original_processed/"

os.makedirs(output_folder, exist_ok=True)


def Parse_csi(lines: List[str]) -> Tuple[np.ndarray, np.ndarray, List[float], List[int]]:
    """
    Parses a list of lines from a CSI file and extracts magnitudes, phases, and timestamps.
    This is an external function not made by us.
    Source: https://github.com/JakobStruye/60gx3_scripts
    :param lines: list of lines from a CSI file
    :return:
    """
    num_lines = len(lines)
    bad_idxs = []

    correct = 0
    incorrect = 0

    times = []
    magnitudes = np.zeros((num_lines, 32))
    phases = np.zeros((num_lines, 32))

    for idx, line in enumerate(lines):
        if "[AOA]" in line and line.count(",") == 71:
            splitted = line.split(",")
            time = float(splitted[1])

            if time not in times:
                times.append(time)
            else:
                print("duplicated")
                continue

            correct += 1
            if correct - 1 >= num_lines:
                break
            for i in range(8, 8 + 32):
                phases[correct - 1][i - 8] = float(splitted[i])

            for i in range(8 + 32, 8 + 32 + 32):
                magnitudes[correct - 1][i - (8 + 32)] = float(splitted[i])

        else:
            incorrect += 1
            bad_idxs.append(idx)

    return magnitudes, phases, times, bad_idxs

def get_next_line(file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[datetime], Optional[List[int]]]:
    """
    Reads the next line from the file and parses it to extract magnitude, phase, timestamp, and bad indices.
    This is an external function not made by us.
    :param file: CSI file to read from
    :return:
    """
    time_csi = []
    mag = None
    phase = None
    bad_idxs = None
    while type(time_csi) != datetime and len(time_csi) == 0:
        line = file.readline()
        if line == "":
            return None, None, None, None
        mag, phase, time_csi, bad_idxs = Parse_csi([line])
        if len(time_csi) == 0:
            continue
        time_csi = datetime.fromtimestamp(time_csi[0])
    return mag, phase, time_csi, bad_idxs

def check_all_times(file: str) -> None:
    """
    Checks whether all items are correctly ordered in time
    :param file:
    :return:
    """
    with open(file, mode='r') as out_file:
        csvreader = csv.reader(out_file)
        prev_time = datetime.fromtimestamp(0)
        for row in csvreader:
            if prev_time > datetime.fromisoformat(row[0]):
                assert "Error"
            prev_time = datetime.fromisoformat(row[0])

def combine_out13_and_out17_one_pass(out13: str, out17: str, combined_csv: str) -> str:
    """
    Combines (aligns) `out13` and `out17` CSV files by matching timestamps within a 1-second threshold.
    Returns the path to the combined CSV file.
    :param out13: out13 CSV file path
    :param out17: out17 CSV file path
    :param combined_csv: output CSV file path for combined data
    :return:
    """
    with open(out13, mode='r') as f:
        out13_data = list(csv.reader(f))

    with open(out17, mode='r') as f:
        out17_data = list(csv.reader(f))

    combined_data = []
    index_17 = 0
    total_17 = len(out17_data)

    used_indices = set() # so that we don't have to delete duplicates later
    skipped_rows = 0

    for row_13 in out13_data:
        time_13 = datetime.fromisoformat(row_13[0])
        best_match = None
        best_index = None
        smallest_delta = timedelta(seconds=10)

        # find the closest available time in out17_data
        while index_17 < total_17 - 1 and abs(datetime.fromisoformat(out17_data[index_17][0]) - time_13) > abs(
                datetime.fromisoformat(out17_data[index_17 + 1][0]) - time_13):
            index_17 += 1

        # search for an unused match that meets the 1s threshold
        while index_17 < total_17:
            time_17 = datetime.fromisoformat(out17_data[index_17][0])
            delta = abs(time_17 - time_13)

            if delta.seconds > 1: # if the difference is more than 1 second, stop searching
                break

            if index_17 not in used_indices and delta < smallest_delta:
                best_match = out17_data[index_17]
                best_index = index_17
                smallest_delta = delta
            else:
                break

            index_17 += 1

        if best_match is not None:
            used_indices.add(best_index)
            combined_data.append(row_13 + best_match)
        else:
            print(f"Skipping row_13 at {time_13}, no valid match found within 1s.")
            skipped_rows += 1

    print(f"Out13 rows: {len(out13_data)}, Out17 rows: {len(out17_data)}, Combined rows: {len(combined_data)}")

    with open(combined_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(combined_data)

    print(f"Saved combined 60GHz CSV: {combined_csv}")

    return combined_csv


def process_60ghz_files(person_id: str, data_folder_60ghz: str) -> list:
    """
    Reads and synchronizes `out13` (first antenna pair) and `out17` (second antenna pair) files, then writes to a CSV.
    This function processes all files for a given person in the 60GHz data folder.
    :param person_id: ID of the person whose data is being processed, which is actually their name
    :param data_folder_60ghz: Folder containing 60GHz data files for the person
    :return:
    """
    print(f"Processing 60GHz data for {person_id}...")

    # Read both `out13` and `out17`
    data_13, data_17 = [], []
    person_60ghz_folder = os.path.join(data_folder_60ghz, person_id)
    out13_files = sorted(f for f in os.listdir(person_60ghz_folder) if "out13" in f and ".csv" not in f)
    out17_files = sorted(f for f in os.listdir(person_60ghz_folder) if "out17" in f and ".csv" not in f)

    combined_files = []

    for i in range(len(out13_files)):

        file_13 = os.path.join(person_60ghz_folder, out13_files[i])
        file_17 = os.path.join(person_60ghz_folder, out17_files[i])

        out13_csv = file_13.replace("out13", "60GHz_out13") + ".csv"
        out17_csv = file_17.replace("out17", "60GHz_out17") + ".csv"
        combined_csv = file_13.replace("out13", "60GHz_combined") + ".csv"

        # get the data from the files
        data_13, data_17 = [], []
        with open(file_13, 'r') as mmwave_file13:
            mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file13)
            while time_csi is not None:
                data_13.append([time_csi.isoformat()] + mag.tolist())
                mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file13)
        with open(file_17, 'r') as mmwave_file17:
            mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file17)
            while time_csi is not None:
                data_17.append([time_csi.isoformat()] + mag.tolist())
                mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file17)

        # write the data to intermediate CSV files
        with open(out13_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_13)
        with open(out17_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_17)

        check_all_times(out13_csv)
        check_all_times(out17_csv)

        print(f"Saved {out13_csv} and {out17_csv}")

        combined_file: str = combine_out13_and_out17_one_pass(out13_csv, out17_csv, combined_csv)
        combined_files.append(combined_file)

    return combined_files

def extract_timestamp_5ghz(filename: str) -> tuple:
    """
    Extracts timestamps from the 5GHz filename and calculates delta correction.
    :param filename: Filename of the 5GHz pcap file, expected to contain SYS and ROUTER timestamps.
    :return:
    """
    match = re.search(r"SYS-(\d{8}_\d{6}\.\d+)_ROUTER-(\d{8}_\d{6}\.\d+)", filename)
    if match:
        sys_time = datetime.strptime(match.group(1)[:22], "%Y%m%d_%H%M%S.%f")
        ssh_time = datetime.strptime(match.group(2)[:22], "%Y%m%d_%H%M%S.%f")
        delta = (sys_time - ssh_time).total_seconds()
        return sys_time, delta
    return None, None

def extract_timestamp_60ghz(filename: str) -> Optional[datetime]:
    """
    Extracts the timestamp from a 60GHz filename.
    :param filename: Filename of the 60GHz CSV file, expected to contain a timestamp in the format YYYYMMDD-HHMMSS.
    :return:
    """
    match = re.search(r"(\d{8})-(\d{6})", filename)
    if match:
        return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return None

def remove_last_packet(filename: str) -> None:
    """
    Removes the last packet from a pcap file to avoid issues with corrupted packets.
    :param filename: Filename of the pcap file to modify.
    :return:
    """
    packets = rdpcap(filename)
    if len(packets) > 0:
        wrpcap(filename, packets[:-1])

def get_csi_data_5ghz(pcap_file: str, delta: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Reads CSI data from a 5GHz pcap file and returns timestamps + amplitude list.
    :param pcap_file: path to the pcap file containing 5GHz CSI data.
    :param delta: Time difference between system time and router time, used to correct timestamps.
    :return:
    """
    reader = get_reader(pcap_file)
    try:
        # sometimes the reader fails to properly read the last (corrupted) packet, but acts as if everything was fine.
        # The result is that the size of the csi_data returned by the reader is less than the reader.pcap.framecount.
        # In this case, we remove the last packet manually here ( (un)comment below )
        remove_last_packet(pcap_file)
        csi_data = reader.read_file(pcap_file, scaled=True)
    except (struct.error, ValueError):
        print(f"Corrupt packet found in {pcap_file}, removing last packet and retrying...")
        try:
            remove_last_packet(pcap_file)
            csi_data = reader.read_file(pcap_file, scaled=True)
        except Exception as e:
            print(f"Error reading file: {e}")
            return None, None

    timestamps = np.array(csi_data.timestamps) + delta  # apply delta correction
    csi_matrix, _, _ = csitools.get_CSI(csi_data, metric="amplitude") # extract CSI amplitudes matrix
    csi_amplitudes = np.squeeze(csi_matrix) if csi_matrix.shape[2] == 1 else csi_matrix # squeeze if Rx antenna dimension is 1

    return timestamps, csi_amplitudes

def synchronize_data(time_5ghz: List[datetime],
                     csi_5ghz: List[datetime],
                     time_60ghz: List[datetime],
                     csi_60ghz: List[datetime]) -> Tuple[List[datetime], List[datetime], List[datetime], List[datetime]]:
    """
    Synchronizes 5GHz and 60GHz data based on timestamps.
    We align based on the latest first timestamp of both datasets and the earliest last timestamp of both datasets.
    :param time_5ghz: list of 5GHz timestamps (datetime objects)
    :param csi_5ghz: list of 5GHz CSI amplitudes (list of lists)
    :param time_60ghz: list of 60GHz timestamps (datetime objects)
    :param csi_60ghz: list of 60GHz CSI amplitudes (list of lists)
    :return: tuple of (synced_time, synced_csi_5ghz, synced_csi_60ghz)
    """

    # Convert to timestamps if inputs are datetime objects
    if isinstance(time_5ghz[0], datetime):
        time_5ghz = [t.timestamp() for t in time_5ghz]
    if isinstance(time_60ghz[0], datetime):
        time_60ghz = [t.timestamp() for t in time_60ghz]

    first_time_5ghz = time_5ghz[0]
    first_time_60ghz = time_60ghz[0] # use time_13 for syncing

    hour_diff = (first_time_60ghz - first_time_5ghz) / 3600
    if abs(hour_diff) >= 0.9:  # If there's at least a 1-hour difference
        print(f"Time of 5GHz and 60GHz data is misaligned, the time of 5ghz is {datetime.fromtimestamp(first_time_5ghz)} and the time of 60ghz is {datetime.fromtimestamp(first_time_60ghz)}")
        print(f"Detected {hour_diff:.6f} hour difference. Adjusting 5GHz timestamps...")
        if hour_diff > 0:
            print("Adjusting 5GHz timestamps forward.")
            print(f"First time 5ghz before adjustment: {datetime.fromtimestamp(time_5ghz[0])}")
            print(f"First time 60ghz before adjustment: {datetime.fromtimestamp(time_60ghz[0])}")
            time_5ghz += round(hour_diff) * 3600
            print(f"First time 5ghz after adjustment: {datetime.fromtimestamp(time_5ghz[0])}")
            print(f"First time 60ghz after adjustment: {datetime.fromtimestamp(time_60ghz[0])}")
        else:
            print("Adjusting 5GHz timestamps backward.")
            print(f"First time 5ghz before adjustment: {datetime.fromtimestamp(time_5ghz[0])}")
            print(f"First time 60ghz before adjustment: {datetime.fromtimestamp(time_60ghz[0])}")
            time_5ghz -= round(abs(hour_diff)) * 3600
            print(f"First time 5ghz after adjustment: {datetime.fromtimestamp(time_5ghz[0])}")
            print(f"First time 60ghz after adjustment: {datetime.fromtimestamp(time_60ghz[0])}")
    else:
        print("Time of 5GHz and 60GHz data is aligned.")

    # find overlapping time range
    start_time = max(time_5ghz[0], time_60ghz[0])
    end_time = min(time_5ghz[-1], time_60ghz[-1])

    # truncate 5GHz data to overlapping range
    indices_5ghz = [i for i, t in enumerate(time_5ghz) if start_time <= t <= end_time]
    time_5ghz_trunc_ts = [time_5ghz[i] for i in indices_5ghz]
    csi_5ghz_trunc = [csi_5ghz[i] for i in indices_5ghz]

    # truncate 60GHz data to overlapping range
    indices_60ghz = [i for i, t in enumerate(time_60ghz) if start_time <= t <= end_time]
    time_60ghz_trunc_ts = [time_60ghz[i] for i in indices_60ghz]
    csi_60ghz_trunc = [csi_60ghz[i] for i in indices_60ghz]

    # convert timestamps back to datetime objects
    time_5ghz_trunc = [datetime.fromtimestamp(t) for t in time_5ghz_trunc_ts]
    time_60ghz_trunc = [datetime.fromtimestamp(t) for t in time_60ghz_trunc_ts]

    print(f"5GHz data truncated from {datetime.fromtimestamp(time_5ghz[0])} -> {datetime.fromtimestamp(time_5ghz[-1])} to {time_5ghz_trunc[0]} -> {time_5ghz_trunc[-1]}")
    print(f"60GHz data truncated from {datetime.fromtimestamp(time_60ghz[0])} -> {datetime.fromtimestamp(time_60ghz[-1])} to {time_60ghz_trunc[0]} -> {time_60ghz_trunc[-1]}")

    return time_5ghz_trunc, csi_5ghz_trunc, time_60ghz_trunc, csi_60ghz_trunc

def downsample_to_rate(timestamps: List[datetime],
                       csi_data: List[np.ndarray],
                       target_rate: int = 10) -> Tuple[List[datetime], List[np.ndarray]]:
    """
    Downsamples CSI data to a specified packets-per-second rate.
    :param timestamps: List of timestamps (datetime objects).
    :param csi_data: Corresponding CSI data for each timestamp.
    :param target_rate: Target rate in packets per second.
    :return:
    """

    # convert to relative seconds (starting from 0)
    start_time = timestamps[0]
    time_seconds = [(t - start_time).total_seconds() for t in timestamps]

    # group indices by the second they belong to
    groups = defaultdict(list)
    for idx, t in enumerate(time_seconds):
        second = int(t)
        groups[second].append(idx)

    # check if every group has at least the target_rate
    for second, indices in groups.items():
        if len(indices) < (target_rate - target_rate*0.2):
            print(f"Warning: Not enough samples in second {second} for downsampling. Found {len(indices)}, expected {target_rate}.")

    # for each second-group, we uniformly pick `target_rate` samples; eg 300p/s with desired 30p/s -> select every ( 300/30= ) 10th sample
    selected_indices = []
    for second, indices in groups.items():
        n = len(indices)
        if n <= target_rate:
            selected_indices.extend(indices)
        else:
            step = n / target_rate
            sampled = [indices[int(i * step)] for i in range(target_rate)]
            selected_indices.extend(sampled)

    selected_indices.sort()

    downsampled_times = [timestamps[i] for i in selected_indices]
    downsampled_csi = [csi_data[i] for i in selected_indices]

    return downsampled_times, downsampled_csi


def impute_nan_csi(csi_array: np.ndarray) -> np.ndarray:
    """
    Imputes NaN values in the CSI matrix, uses linear interpolation along the time axis for each subcarrier.
    Falls back to forward/backward fill and finally 0 if needed.
    :param csi_array: CSI matrix with Inf/NaN values
    :return:
    """
    csi_array = np.where(np.isinf(csi_array), np.nan, csi_array) # replace inf with nan, for df

    csi_df = pd.DataFrame(csi_array)
    csi_df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    csi_df.fillna(method='ffill', inplace=True)
    csi_df.fillna(method='bfill', inplace=True)
    csi_df.fillna(0, inplace=True)
    return csi_df.to_numpy()


def process_person(person_id: str,
                   idx: int,
                   data_folder_5ghz: str,
                   data_folder_60ghz: str,
                   target_rate: int = 10) -> None:
    """
    Processes a person's data, synchronizing 5GHz and 60GHz CSI, handling multiple files, and saving final .npy files.
    :param person_id: Person ID to process, which is actually their name
    :param idx: Index of the person in the list of persons, used for file naming (the external dataset (see paper) had this naming system so we adopted it too)
    :param data_folder_5ghz: 5GHz data folder path
    :param data_folder_60ghz: 60GHz data folder path
    :param target_rate: Target rate in packets per second for downsampling.
    :return:
    """
    print(f"Processing data for {person_id}...")

    # Step 1: Get synchronized 60GHz CSV
    combined_60ghz_files = process_60ghz_files(person_id, data_folder_60ghz)

    person_5ghz_folder = os.path.join(data_folder_5ghz, person_id)
    five_ghz_files = sorted([f for f in os.listdir(person_5ghz_folder) if f.endswith(".pcap")])

    grouped_5ghz = {}
    for file in five_ghz_files:
        sys_time, delta = extract_timestamp_5ghz(file)
        date_key = sys_time.date()
        if date_key not in grouped_5ghz:
            grouped_5ghz[date_key] = []
        grouped_5ghz[date_key].append((file, delta))

    grouped_60ghz = {}
    for file in combined_60ghz_files:
        date_key = extract_timestamp_60ghz(file).date()
        if date_key not in grouped_60ghz:
            grouped_60ghz[date_key] = []
        grouped_60ghz[date_key].append(file)

    print(grouped_5ghz)
    print(grouped_60ghz)

    # now we need to go over the grouped files and sync them
    # if grouped_5ghz has only one item, we can just sync it with the corresponding 60ghz file
    # if grouped_5ghz has multiple items, we need to first sync the first 5ghz file with the first 60ghz file, and then concatenate 5ghz files together and concatenate 60ghz files together

    for date_key in grouped_5ghz:
        if date_key not in grouped_60ghz:
            print(f"No matching 60GHz data for date {date_key}, skipping...")
            continue

        # process 5GHz files for this date
        all_5ghz_times = []
        all_5ghz_csi = []

        # read and concatenate all 5GHz files for this date
        for file, delta in grouped_5ghz[date_key]:
            pcap_path = os.path.join(person_5ghz_folder, file)
            timestamps, csi_amplitudes = get_csi_data_5ghz(pcap_path, delta)
            assert len(timestamps) == len(csi_amplitudes) and len(timestamps) > 0

            all_5ghz_times.append(timestamps)
            all_5ghz_csi.append(csi_amplitudes)

        if len(all_5ghz_times) == 0:
            print(f"No valid 5GHz data for date {date_key}, skipping...")
            continue

        # process 60GHz files for this date
        all_60ghz_times = []
        all_60ghz_csi = []

        # read and concatenate all 60GHz files for this date
        for file in grouped_60ghz[date_key]:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                file_times = []
                file_csi = []
                for row in reader:
                    timestamp = datetime.fromisoformat(row[0])
                    csi_data_13 = eval(row[1])
                    timestamp_17 = datetime.fromisoformat(row[2])
                    csi_data_17 = eval(row[3])
                    csi_data = [csi_data_13] + [timestamp_17] + [csi_data_17]
                    file_times.append(timestamp)
                    file_csi.append(csi_data)

                all_60ghz_times.append(file_times)
                all_60ghz_csi.append(file_csi)

        if len(all_60ghz_times) == 0:
            print(f"No valid 60GHz data for date {date_key}, skipping...")
            continue

        assert len(all_60ghz_times) == len(all_5ghz_times)

        if len(grouped_5ghz[ date_key]) == 1:
            # if only one 5GHz file exists, synchronize with the corresponding 60GHz file
            synced_5ghz_time, synced_5ghz_csi, synced_60ghz_time, synced_60ghz_csi = synchronize_data(all_5ghz_times[0], all_5ghz_csi[0], all_60ghz_times[0], all_60ghz_csi[0])
        else:
            # if multiple 5GHz files exist, we need to first sync the first 5ghz file with the first 60ghz file, and then concatenate 5ghz files together and concatenate 60ghz files together

            assert len(all_5ghz_times) == 2 and len(all_60ghz_times) == 2

            # sync the first 5GHz file with the first 60GHz file
            time_5ghz_0, csi_5ghz_0 = all_5ghz_times[0], all_5ghz_csi[0]
            time_60ghz_0, csi_60ghz_0 = all_60ghz_times[0], all_60ghz_csi[0]
            synced_5ghz_time_0, synced_5ghz_csi_0, synced_60ghz_time_0, synced_60ghz_csi_0 = synchronize_data(time_5ghz_0, csi_5ghz_0, time_60ghz_0, csi_60ghz_0)

            # sync the second 5GHz file with the second 60GHz file
            time_5ghz_1, csi_5ghz_1 = all_5ghz_times[1], all_5ghz_csi[1]
            time_60ghz_1, csi_60ghz_1 = all_60ghz_times[1], all_60ghz_csi[1]
            synced_5ghz_time_1, synced_5ghz_csi_1, synced_60ghz_time_1, synced_60ghz_csi_1 = synchronize_data(time_5ghz_1, csi_5ghz_1, time_60ghz_1, csi_60ghz_1)

            # concatenate 5GHz and 60GHz data
            synced_5ghz_time = synced_5ghz_time_0 + synced_5ghz_time_1
            synced_5ghz_csi = synced_5ghz_csi_0 + synced_5ghz_csi_1
            synced_60ghz_time = synced_60ghz_time_0 + synced_60ghz_time_1
            synced_60ghz_csi = synced_60ghz_csi_0 + synced_60ghz_csi_1

            assert len(synced_5ghz_time) == len(synced_5ghz_time_0) + len(synced_5ghz_time_1)

        assert len(synced_5ghz_time) == len(synced_5ghz_csi) and len(synced_60ghz_time) == len(synced_60ghz_csi)

        synced_60ghz_csi = [data[0][:30] + data[2][:30] for data in synced_60ghz_csi]

        data_subcarrier_indices = list(range(1, 27)) + list(range(38, 64))
        synced_5ghz_csi = np.array(synced_5ghz_csi)[:, data_subcarrier_indices]

        print(f"Downsampling 5ghz data to {target_rate}Hz...")
        synced_5ghz_time, synced_5ghz_csi = downsample_to_rate(synced_5ghz_time, synced_5ghz_csi, target_rate)
        print(f"Downsampling 60ghz data to 10Hz...")
        synced_60ghz_time, synced_60ghz_csi = downsample_to_rate(synced_60ghz_time, synced_60ghz_csi, 10)

        if np.isnan(synced_5ghz_csi).any() or np.isinf(synced_5ghz_csi).any():
            print(f"NaN/Inf values found in 5GHz data for {person_id} ({date_key}), imputing...")
            synced_5ghz_csi = impute_nan_csi(synced_5ghz_csi)
        if np.isnan(synced_60ghz_csi).any() or np.isinf(synced_60ghz_csi).any():
            print(f"NaN/Inf values found in 60GHz data for {person_id} ({date_key}), imputing...")
            synced_60ghz_csi = impute_nan_csi(synced_60ghz_csi)

        # save synchronized data
        if 'background' in data_folder_5ghz:
            np.save(os.path.join(output_folder + "5ghz/", f"{person_id}_{date_key}_5ghz_backgroundarrayuser_{idx+1}.npy"), np.array(synced_5ghz_csi))
            np.save(os.path.join(output_folder + "60ghz/", f"{person_id}_{date_key}_60ghz_backgroundarrayuser_{idx+1}.npy"), np.array(synced_60ghz_csi))
        else:
            np.save(os.path.join(output_folder + "5ghz/", f"{person_id}_{date_key}_5ghz_walkarrayuser_{idx+1}.npy"), np.array(synced_5ghz_csi))
            np.save(os.path.join(output_folder + "60ghz/", f"{person_id}_{date_key}_60ghz_walkarrayuser_{idx+1}.npy"), np.array(synced_60ghz_csi))
        print(f"Saved synchronized data for {person_id} ({date_key})")


if __name__ == "__main__":

    target_rate = 10  # packets per second

    # Process 5GHz and 60GHz walking data for all persons
    # for idx, person in enumerate(os.listdir(data_folder_5ghz_walking)):
    #     if os.path.isdir(os.path.join(data_folder_5ghz_walking, person)):
    #         process_person(person, idx, data_folder_5ghz_walking, data_folder_60ghz_walking, target_rate)

    # # Process 5GHz and 60GHz background data
    # for idx, person in enumerate(os.listdir(data_folder_5ghz_background)):
    #     if os.path.isdir(os.path.join(data_folder_5ghz_background, person)):
    #         process_person(person, idx, data_folder_5ghz_background, data_folder_60ghz_background, target_rate)


    import matplotlib.pyplot as plt

    data_5ghz = np.load('data/collected_csi_data_original_processed/5ghz/xinlei_2025-03-12_5ghz_walkarrayuser_20.npy')
    print(data_5ghz.shape)
    print(data_5ghz[0])
    #
    # data_60ghz = np.load('data/collected_csi_data_original_processed/60ghz/andre_2025-03-12_60ghz_walkarrayuser_1.npy')
    # print(data_60ghz.shape)
    # print(data_60ghz[0])
    #
    # data_60ghz_13 = data_60ghz[:, :30]
    # data_60ghz_17 = data_60ghz[:, 30:]
    #
    # avg_5ghz = np.mean(data_5ghz, axis=1)
    # avg_60ghz_13 = np.mean(data_60ghz_13, axis=1)
    # avg_60ghz_17 = np.mean(data_60ghz_17, axis=1)
    #
    # # avg_60ghz = avg_60ghz[:, :30]
    #
    # # Plotting
    # plt.figure(figsize=(12, 5))
    #
    # plt.plot(avg_5ghz, label="5 GHz (avg per time step)", alpha=0.8)
    #
    # plt.title("Average CSI Amplitude per Time Step")
    # plt.xlabel("Time Step")
    # plt.ylabel("Average Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(12, 5))
    #
    # plt.plot(avg_60ghz_13, label="60 GHz (13 GHz avg per time step)", alpha=0.8)
    # plt.plot(avg_60ghz_17, label="60 GHz (17 GHz avg per time step)", alpha=0.6)
    #
    # plt.title("Average CSI Amplitude per Time Step")
    # plt.xlabel("Time Step")
    # plt.ylabel("Average Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    #
    # def detect_gait_segments(csi_data: np.ndarray,
    #                          window_size: int = 10,
    #                          phi_var_factor: float = 0.9,
    #                          buffer_len: int = 3,
    #                          tlen_threshold: int = 1,
    #                          beta_threshold: float = 0.1,
    #                          plot: bool = True):
    #     """
    #     Detects gait start and end indices based on CSI signal variance.
    #     Based on the method described in the provided paper section.
    #
    #     :param csi_data: [timesteps, subcarriers] amplitude data
    #     :param window_size: number of packets in each non-overlapping sliding window
    #     :param phi_var: variance threshold for detecting walking
    #     :param buffer_len: how many consecutive low-variance windows are allowed before stopping detection
    #     :param tlen_threshold: minimum duration for valid walking segment
    #     :param beta_threshold: average variance threshold for rejecting weak interference segments
    #     :param plot: whether to generate a plot
    #     :return: list of (start_idx, end_idx) tuples
    #     """
    #     num_windows = len(csi_data) // window_size
    #     variances = []
    #
    #     for t in range(num_windows):
    #         window = csi_data[t * window_size: (t + 1) * window_size]
    #         mean_per_subcarrier = np.mean(window, axis=0)
    #         var = np.mean((window - mean_per_subcarrier) ** 2)
    #         variances.append(var)
    #
    #     avg_variance = np.mean(variances)
    #     phi_var = phi_var_factor * avg_variance
    #     print(f"Average variance: {avg_variance:.4f}")
    #
    #     gait_segments = []
    #     buffer = 0
    #     start = False
    #     start_idx = None
    #
    #     for t, var in enumerate(variances):
    #         if var > phi_var and not start:
    #             start = True
    #             start_idx = t
    #         elif var > phi_var and start:
    #             buffer = 0
    #         elif var < phi_var and start:
    #             if buffer > buffer_len:
    #                 end_idx = t
    #                 start = False
    #                 gait_segments.append((start_idx, end_idx))
    #             else:
    #                 buffer += 1
    #
    #     filtered_segments = []
    #     for start_t, end_t in gait_segments:
    #         j = end_t - start_t
    #         segment_mean_var = np.mean(variances[start_t:end_t])
    #         if j >= tlen_threshold or segment_mean_var >= beta_threshold:
    #             filtered_segments.append((start_t, end_t))
    #
    #     if plot:
    #         plt.figure(figsize=(12, 5))
    #         avg_amplitudes = np.mean(csi_data, axis=1)
    #         plt.plot(avg_amplitudes, label="Average Amplitude")
    #
    #         for start_t, end_t in filtered_segments:
    #             plt.axvline(start_t * window_size, color='green', linestyle='--', label='Gait Start')
    #             plt.axvline(end_t * window_size, color='red', linestyle='--', label='Gait End')
    #
    #         plt.title("Gait Detection Based on Variance")
    #         plt.xlabel("Time Index")
    #         plt.ylabel("Amplitude")
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.show()
    #
    #     return filtered_segments
    #
    #
    # segments = detect_gait_segments(data_60ghz[:, :30][:100], window_size=2, phi_var_factor=1.05)
    #
    #
    # def detect_gait_with_background(walk_data: np.ndarray,
    #                                 background_data: np.ndarray,
    #                                 window_size: int = 10,
    #                                 threshold_std_factor: float = 2.0,
    #                                 min_segment_windows: int = 2,
    #                                 plot: bool = True):
    #     """
    #     Detects gait segments in CSI data using variance thresholding based on a background reference signal.
    #
    #     Parameters:
    #     - walk_data: ndarray [timesteps, subcarriers], the CSI amplitude data with walking
    #     - background_data: ndarray [timesteps, subcarriers], background CSI data without walking
    #     - window_size: number of timesteps per non-overlapping window
    #     - threshold_std_factor: how many standard deviations above background variance mean is considered walking
    #     - min_segment_windows: minimum number of consecutive high-variance windows to be considered a valid segment
    #     - plot: whether to plot the signal and detected segments
    #
    #     Returns:
    #     - List of (start_idx, end_idx) tuples indicating detected gait segments
    #     """
    #
    #     def compute_window_variances(data, win_size):
    #         num_windows = len(data) // win_size
    #         variances = []
    #         for i in range(num_windows):
    #             window = data[i * win_size:(i + 1) * win_size]
    #             mean_sub = np.mean(window, axis=0)
    #             var = np.mean((window - mean_sub) ** 2)
    #             variances.append(var)
    #         return np.array(variances)
    #
    #     bg_variances = compute_window_variances(background_data, window_size)
    #     bg_mean = np.mean(bg_variances)
    #     bg_std = np.std(bg_variances)
    #     threshold = bg_mean + threshold_std_factor * bg_std
    #     print(f"[Threshold] Background mean: {bg_mean:.4f}, std: {bg_std:.4f}, threshold: {threshold:.4f}")
    #
    #     walk_variances = compute_window_variances(walk_data, window_size)
    #
    #     is_walking = walk_variances > threshold
    #
    #     gait_segments = []
    #     start = None
    #     for i, active in enumerate(is_walking):
    #         if active and start is None:
    #             start = i
    #         elif not active and start is not None:
    #             if i - start >= min_segment_windows:
    #                 gait_segments.append((start, i))
    #             start = None
    #     if start is not None and (len(is_walking) - start >= min_segment_windows):
    #         gait_segments.append((start, len(is_walking)))
    #
    #     if plot:
    #         avg_amplitude = np.mean(walk_data, axis=1)
    #         plt.figure(figsize=(12, 5))
    #         plt.plot(avg_amplitude, label="Avg Amplitude (Walk Signal)", alpha=0.8)
    #
    #         for start_win, end_win in gait_segments:
    #             plt.axvline(start_win * window_size, color='green', linestyle='--', label='Gait Start')
    #             plt.axvline(end_win * window_size, color='red', linestyle='--', label='Gait End')
    #
    #         plt.title("Gait Detection using Background Variance Threshold")
    #         plt.xlabel("Time Step")
    #         plt.ylabel("Average CSI Amplitude")
    #         plt.grid(True)
    #         # plt.legend()
    #         plt.tight_layout()
    #         plt.show()
    #
    #     return gait_segments
    #
    #
    # background_60ghz = np.load('data/collected_csi_data_original_processed/60ghz/background_2025-03-13_60ghz_backgroundarrayuser_1.npy')
    # segments = detect_gait_with_background(
    #     walk_data=data_60ghz[:, :30][:100],
    #     background_data=background_60ghz[:, :30],
    #     window_size=5,
    #     threshold_std_factor=4.0,
    #     min_segment_windows=2,
    #     plot=True
    # )