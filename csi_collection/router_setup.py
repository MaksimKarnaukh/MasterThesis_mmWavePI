"""
Script to set up the router for CSI collection and execute necessary commands.
"""

import subprocess
import datetime
import sys
import time
import os

def execute_ssh_command(command):
    """
    Executes an SSH command and checks for connection issues.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        time.sleep(2)
        if "onnection" in result.stderr.lower() or "onnection" in result.stdout.lower():
            print("Warning: SSH connection issue detected!")
            print(result.stderr, result.stdout)
        return result
    except subprocess.CalledProcessError:
        print(f"Error executing command: {command}")
        return None

def setup_router():
    """
    Executes the necessary commands on the router to prepare it for CSI collection.
    """
    setup_commands = [
        "/sbin/rmmod dhd",
        "/sbin/insmod /jffs/dhd.ko",
        "wl -i eth6 up",
        "wl -i eth6 radio on",
        "wl -i eth6 country BE",
        "ifconfig eth6 up",
    ]

    print("Setting up router...")
    for cmd in setup_commands:
        ssh_command = f'sshpass -p "nabeel" ssh admin@192.168.1.1 "{cmd}"'
        print(f"executing: ", ssh_command)
        execute_ssh_command(ssh_command)

    print("Router setup complete. Starting CSI collection...")

    mcp_command = 'sshpass -p "nabeel" ssh admin@192.168.1.1 "/jffs/mcp -c 36/80 -C 1 -N 1 -m 50:EB:F6:33:16:94"'
    print(f"executing: ", mcp_command)
    result = execute_ssh_command(mcp_command)
    if result:
        key = result.stdout.strip().split()[-1]
        print("key: ", key)
    else:
        print("Error retrieving key, aborting setup.")
        return

    # run remaining CSI setup commands
    csi_commands = [
        f'/jffs/nexutil -Ieth6 -s500 -b -l34 -v{key}',
        '/usr/sbin/wl -i eth6 monitor 1'
    ]

    for cmd in csi_commands:
        ssh_command = f'sshpass -p "nabeel" ssh admin@192.168.1.1 "{cmd}"'
        print(f"executing: ", ssh_command)
        execute_ssh_command(ssh_command)

    print("CSI collection started.")


if __name__ == "__main__":
    setup_router()
