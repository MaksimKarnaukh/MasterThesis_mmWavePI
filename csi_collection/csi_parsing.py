"""
CSIKit Example: Parsing and Plotting CSI Data from PCAP Files
This is mainly a manual testing script.
"""

from CSIKit.reader import get_reader
from CSIKit.util import csitools
from CSIKit.filters.passband import lowpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel
from CSIKit.tools.batch_graph import BatchGraph

import matplotlib.pyplot as plt
import numpy as np
import struct
import os

data_folder = "./csi_collection/input/"
output_folder = "./csi_collection/output/"

def remove_last_packet(filename: str, new_filename: str) -> None:
    """
    Removes the last packet from a pcap file and saves the modified pcap as a new file.
    This function is handy when the last packet is incomplete or corrupted, since this can cause issues when reading the pcap file.
    :param filename: filename of the pcap file to modify
    :param new_filename: filename of the new (modified) pcap file
    :return:
    """

    try:
        from scapy.all import rdpcap, wrpcap

        packets = rdpcap(data_folder + filename)

        if len(packets) > 0:
            print(f"Number of packets in the file: {len(packets)}")
            packets = packets[:-1] # remove the last packet
        else:
            print("No packets found in the file.")
            return

        wrpcap(data_folder + new_filename, packets)

        print(f"Last packet removed. New file saved as {new_filename}")

    except Exception as e:

        print("Error removing last packet using scapy.")
        print("Trying to remove the last packet using dpkt...")

        import dpkt

        def remove_last_packet(input_pcap, output_pcap):
            with open(data_folder + input_pcap, 'rb') as f:
                pcap_reader = dpkt.pcap.Reader(f)
                packets = list(pcap_reader)
                print(f"Number of packets in the file: {len(packets)}")

            with open(data_folder + output_pcap, 'wb') as f:
                pcap_writer = dpkt.pcap.Writer(f)
                for packet in packets[:-1]:
                    pcap_writer.writepkt(packet[1], ts=packet[0])

            print(f"Saved modified pcap to: {data_folder + output_pcap}")

        remove_last_packet(filename, new_filename)

def extract_csi(pcap_file: str, metric: str = "amplitude") -> np.ndarray:
    """
    Extracts the CSI data from a pcap file and returns the CSI matrix.
    :param pcap_file: name of pcap file to extract the CSI data from
    :param metric: metric to extract from the CSI data
    :return:
    """

    reader = get_reader(data_folder + pcap_file)
    csi_data: np.ndarray = None
    try: # Exception handling for corrupted pcap files (e.g., incomplete packets)
        csi_data = reader.read_file(data_folder + pcap_file, scaled=True) # 'scaled=True' converts values to dB
    except (struct.error, ValueError) as e:
        print(f"Error reading file: {e}")
        try:
            print("Removing last packet and retrying...")
            new_filename = "new_" + pcap_file # Save the modified pcap as a new file
            remove_last_packet(pcap_file, new_filename)
            csi_data = reader.read_file(data_folder + new_filename, scaled=True)
        except Exception as e:
            print(f"Error reading file: {e}")
            print("If the error concerns an (integer) overflow error and you are running this on a Windows machine, "
                  "try running the script on a Unix-based OS.")
            return None

    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric=metric)

    print(f"Extracted CSI Matrix Shape: {csi_matrix.shape}") # -> (num_frames, num_subcarriers, num_rx, num_tx)

    if csi_matrix.shape[2] == 1 and csi_matrix.shape[3] == 1:
        csi_matrix = np.squeeze(csi_matrix) # -> (num_frames, num_subcarriers)

    return csi_matrix

def plot_csi_amplitudes(csi_amplitudes: np.ndarray) -> None:
    """
    Plots the CSI amplitude heatmap and average amplitude per subcarrier.

    :param csi_amplitudes: CSI amplitude matrix (num_frames, num_subcarriers)
    """

    # Remove pilot subcarriers and whatever else is not data
    data_subcarrier_indices = list(range(1, 27)) + list(range(38, 64))
    csi_amplitudes = csi_amplitudes[:, data_subcarrier_indices]

    plt.figure(figsize=(12, 6))
    plt.imshow(csi_amplitudes.T, aspect="auto", cmap="jet", interpolation="nearest", origin="lower")
    plt.colorbar(label="CSI Amplitude (dB)")
    plt.xlabel("Frame Index (Time Progression)")
    plt.ylabel("Subcarrier Index")
    plt.title("CSI Amplitude Heatmap")
    plt.savefig(output_folder + "fig0")
    plt.show()

    csi_amplitudes = np.where(np.isinf(csi_amplitudes), np.nan, csi_amplitudes)

    mean_per_subcarrier = np.nanmean(csi_amplitudes, axis=0)
    print("Mean CSI Amplitude per Subcarrier:", mean_per_subcarrier)

    # plot mean amplitude per subcarrier
    plt.figure(figsize=(12, 6))
    plt.plot(mean_per_subcarrier, linestyle="-", marker="o", markersize=3)
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Average CSI Amplitude (dB)")
    plt.title("Average CSI Amplitude per Subcarrier")
    plt.grid(True)
    plt.savefig(output_folder + "fig1")
    plt.show()

    avg_amplitude_per_frame = np.nanmean(csi_amplitudes, axis=1)  # average over all subcarriers

    num_frames, num_subcarriers = csi_amplitudes.shape
    assert num_subcarriers <= 64
    group_size = 5

    num_groups = num_frames // group_size
    grouped_amplitudes = np.array([
        np.mean(avg_amplitude_per_frame[i * group_size: (i + 1) * group_size])
        for i in range(num_groups)
    ])

    x_values = np.arange(num_groups) * group_size

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, grouped_amplitudes, linestyle='-')
    plt.xlabel("Frame Index (Grouped)")
    plt.ylabel("Average CSI Amplitude (dB)")
    plt.title(f"Average CSI Amplitude Over Time (Grouped by {group_size} Frames)")
    plt.grid(True)
    plt.savefig(output_folder + "fig2")
    plt.show()

def plot_csi_amplitudes_CSIKIT():
    """
    CSIKIT Example: Plotting CSI Amplitude Heatmap
    (https://github.com/Gi-z/CSIKit)
    :return:
    """

    my_reader = get_reader(data_folder + filename)
    csi_data = my_reader.read_file(data_folder + filename, scaled=True)
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")

    # CSI matrix is now returned as (no_frames, no_subcarriers, no_rx_ant, no_tx_ant).
    # First we'll select the first Rx/Tx antenna pairing.
    csi_matrix_first = csi_matrix[:, :, 0, 0]
    # Then we'll squeeze it to remove the singleton dimensions.
    csi_matrix_squeezed = np.squeeze(csi_matrix_first)

    # This example assumes CSI data is sampled at ~100Hz.
    # In this example, we apply (sequentially):
    #  - a lowpass filter to isolate frequencies below 10Hz (order = 5)
    #  - a hampel filter to reduce high frequency noise (window size = 10, significance = 3)
    #  - a running mean filter for smoothing (window size = 10)

    sampling_rate = 300

    # for x in range(no_frames):
    #     csi_matrix_squeezed[x] = lowpass(csi_matrix_squeezed[x], 10, sampling_rate, 5)
    #     csi_matrix_squeezed[x] = hampel(csi_matrix_squeezed[x], 10, 3)
    #     csi_matrix_squeezed[x] = running_mean(csi_matrix_squeezed[x], 10)

    BatchGraph.plot_heatmap(csi_matrix_squeezed, csi_data.timestamps)

def save_csi_to_npy(csi_amplitudes: np.ndarray, filename: str, output_folder: str = "./csi_collection/output/") -> None:
    """
    Saves the CSI amplitude matrix as a .npy file.

    :param csi_amplitudes: CSI amplitude matrix (num_frames, num_subcarriers)
    :param filename: Filename for the saved .npy file (including extension)
    """
    filename = filename.replace(".pcap", "")
    npy_filename = os.path.join(output_folder, f"{filename}.npy")
    np.save(npy_filename, csi_amplitudes)
    print(f"CSI data saved to {npy_filename}")

def save_csi_files_to_npy(data_folder: str, output_folder: str) -> None:
    """
    Saves all CSI amplitude matrices in the data folder as .npy files.

    :param data_folder: Folder containing the CSI pcap files
    :param output_folder: Folder to save the .npy files
    """
    for filename in os.listdir(data_folder):
        if filename.endswith(".pcap"):
            csi_amplitudes = extract_csi(filename)
            save_csi_to_npy(csi_amplitudes, filename, output_folder)

def load_csi_from_npy(npy_filename: str) -> np.ndarray:
    """
    Loads the CSI amplitude matrix from a .npy file.

    :param npy_filename: Path to the .npy file
    :return: Loaded CSI amplitude matrix
    """
    csi_amplitudes = np.load(npy_filename)
    print(f"Loaded CSI data from {npy_filename}. Shape: {csi_amplitudes.shape}")
    return csi_amplitudes

if __name__ == "__main__":

    filename = "csi_capture_20250308_024203.pcap"  # "capture_moving.pcap"
    file = "csi_capture_SYS-20250312_143855.072370_ROUTER-20180505_050605.1525496765"
    remove_last_packet(file + ".pcap", f"{file}_new.pcap")
    # csi_amplitudes = extract_csi(filename)
    # print("CSI Amplitudes example:", csi_amplitudes)
    # plot_csi_amplitudes(csi_amplitudes)
    # # plot_csi_amplitudes_CSIKIT()
    # save_csi_to_npy(csi_amplitudes, filename)
    # csi_amplitudes = load_csi_from_npy(output_folder + "csi_capture_20250308_024203.npy")
    # print(csi_amplitudes[0])