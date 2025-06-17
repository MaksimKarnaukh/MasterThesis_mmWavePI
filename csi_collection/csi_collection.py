"""
CSI Collection Script, which is run after the router has been set up to capture CSI data.
"""

import subprocess
import datetime
import sys
import time
import os

def get_ssh_date():
    """
    Retrieves the date from the SSH device with milliseconds precision.
    """
    date_command = 'sshpass -p "nabeel" ssh admin@192.168.1.1 "date +%Y%m%d_%H%M%S.%s"'
    try:
        ssh_date = subprocess.check_output(date_command, shell=True, text=True).strip()
        return ssh_date
    except subprocess.CalledProcessError:
        print("Error retrieving SSH date. Using local timestamp only.")
        return None

def check_file_growth(filename, start_time):
    """
    Monitors the file size to ensure it is being populated.
    """
    previous_size = -1
    while True:
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        if os.path.exists(filename):
            current_size = os.path.getsize(filename)
            if current_size > 0:
                print(f"[{minutes:02d}:{seconds:02d}] File {filename} is growing: {current_size} bytes")
            else:
                print(f"Warning: File {filename} is empty.")
            if current_size == previous_size:
                print("Warning: File size has not changed. Data may not be written.")
            previous_size = current_size
        time.sleep(2) # Check every 2 seconds

def record_csi():
    """
    Starts the CSI recording via SSH and saves it to a timestamped file.
    """
    import threading

    ssh_command = 'sshpass -p "nabeel" ssh admin@192.168.1.1 "/jffs/tcpdump -i eth6 dst port 5500 -w -" > {}'

    recording = False
    process = None

    print("Press Enter to start recording. Press Enter again to stop.") # for easy toggling

    while True:
        input()

        if not recording:
            ssh_timestamp = get_ssh_date()
            local_timestamp_unix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")

            if ssh_timestamp:
                filename = f"csi_capture_SYS-{local_timestamp_unix}_ROUTER-{ssh_timestamp}.pcap"
            else:
                filename = f"csi_capture_{local_timestamp_unix}.pcap"

            process = subprocess.Popen(ssh_command.format(filename), shell=True, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, text=True)
            print(f"Recording... (Saving to {filename})")

            start_time = time.time()

            monitor_thread = threading.Thread(target=check_file_growth, args=(filename,start_time), daemon=True)
            monitor_thread.start()

            recording = True
        else:
            print("Stopping recording...")
            process.terminate()
            process.wait()
            print("Stopped.")
            recording = False


if __name__ == "__main__":
    record_csi()
