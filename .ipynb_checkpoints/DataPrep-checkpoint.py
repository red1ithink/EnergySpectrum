import os
import re
import csv

import numpy as np
import pandas as pd


#####extracting velocity(ux, uy)#####

base_path = '../../Downloads/2x'

all_folders = os.listdir(base_path)

numeric_folders = sorted(
    [folder for folder in all_folders if folder.replace('.', '').isdigit()],
    key=lambda x: float(x)
)

output_dir = '../2x/u'
os.makedirs(output_dir, exist_ok=True)

for folder in numeric_folders:
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        u_file_path = os.path.join(folder_path, "U")
        if os.path.isfile(u_file_path):
            print(f"Processing file: {u_file_path}")
            with open(u_file_path, 'r') as file:
                lines = file.readlines()
                data = []
                in_data_section = False

                for line in lines:
                    line = line.strip()
                    if line.startswith("internalField"):
                        in_data_section = True
                        continue
                    if in_data_section:
                        if line.startswith(")"):
                            break
                        match = re.match(r"\(([-\d.eE+]+) ([-\d.eE+]+) [-\d.eE+]+\)", line)
                        if match:
                            ux = float(match.group(1))
                            uy = float(match.group(2))
                            data.append((ux, uy))

                if data:
                    csv_file = os.path.join(output_dir, f"{folder}_ux_uy_data.txt")
                    with open(csv_file, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["ux", "uy"])
                        writer.writerows(data)
                    print(f"Saved {len(data)} entries for folder {folder} to {csv_file}")
                else:
                    print(f"No vector data found in {u_file_path}.")

def load_csv_data(folder_name):
    csv_file = os.path.join(output_dir, f"{folder_name}_ux_uy_data.txt")
    if os.path.isfile(csv_file):
        try:
            data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            return data
        except ValueError:
            print(f"No valid data in file: {csv_file}")
            return None
    else:
        print(f"No data file found for folder: {folder_name}")
        return None

folder_to_process = "0.2"
data = load_csv_data(folder_to_process)
if data is not None and len(data) > 0:
    print(f"Loaded {len(data)} entries from folder {folder_to_process}")
    avg_ux = np.mean(data[:, 0])
    avg_uy = np.mean(data[:, 1])
    print(f"Average ux: {avg_ux:.6f}, uy: {avg_uy:.6f}")
else:
    print(f"No data available for folder {folder_to_process}.")
