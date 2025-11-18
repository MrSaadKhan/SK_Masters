import os
import json
import ipaddress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import multiprocessing
import time
import psutil

# ----------------- SETTINGS -----------------
num_devices = 7
folder_path = '/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix/'

# Toggle whether to include “cheating” IP columns
include_cheating_columns = False  # Set to False to exclude IP columns

# Fields to exclude always (excluding IPs is optional)
exclude_fields = ['sourceMacAddress', 'destinationMacAddress']
if include_cheating_columns:
    exclude_fields.append('sourceIPv4Address')
    exclude_fields.append('destinationIPv4Address')

input_output_map = {
    "Amazon Echo Gen2": "Amazon Echo Gen2",
    "Au Network Camera": "Network Camera",
    "Au Wireless Adapter": "Wireless Adapter",
    "Bitfinder Awair Breathe Easy": "Bitfinder Smart Air Monitor",
    "Candy House Sesami Wi-fi Access Point": "Candy House Wi-Fi AP",
    "Google Home Gen1": "Google Home Gen1",
    "I-o Data Qwatch": "IO Data Camera",
    "Irobot Roomba": "iRobot Roomba",
    "Jvc Kenwood Cu-hb1": "JVC Smart Home Hub",
    "Jvc Kenwood Hdtv Ip Camera": "JVC Camera",
    "Line Clova Wave": "Line Smart Speaker",
    "Link Japan Eremote": "Link eRemote",
    "Mouse Computer Room Hub": "Mouse Computer Room Hub",
    "Nature Remo": "Nature Smart Remote",
    "Panasonic Doorphone": "Panasonic Doorphone",
    "Philips Hue Bridge": "Philips Hue Light",
    "Planex Camera One Shot!": "Planex Camera",
    "Planex Smacam Outdoor": "Planex Outdoor Camera",
    "Planex Smacam Pantilt": "Planex PanTilt Camera",
    "Powerelectric Wi-fi Plug": "PowerElectric Wi-Fi Plug",
    "Qrio Hub": "Qrio Hub",
    "Sony Bravia": "Sony Bravia",
    "Sony Network Camera": "Sony Network Camera",
    "Sony Smart Speaker": "Sony Smart Speaker",
    "Xiaomi Mijia Led": "Xiaomi Mijia LED"
}

# ----------------- UTILITY FUNCTIONS -----------------
def ip_to_int(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        return int(ip_obj)
    except ValueError:
        return None

def is_local_ip(ip):
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False

def is_ipv6(ip):
    try:
        return ipaddress.ip_address(ip).version == 6
    except ValueError:
        return False

def memory_monitor(output_dir, stop_event, mem_start):
    output_file = os.path.join(output_dir, "memory_measurements.txt")
    with open(output_file, "a") as f:
        while not stop_event.is_set():
            mem_usage = psutil.virtual_memory().used / (1024 ** 2)
            mem_usage -= mem_start
            f.write(f"{mem_usage}\n")
            time.sleep(0.001)
    print(f"Memory measurements saved to {output_file}")

def highest_value_without_outliers(filename):
    with open(filename, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    if not numbers:
        return None
    sorted_numbers = sorted(numbers)
    q1 = sorted_numbers[int(len(sorted_numbers)*0.25)]
    q3 = sorted_numbers[int(len(sorted_numbers)*0.75)]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_numbers = [num for num in sorted_numbers if lower_bound <= num <= upper_bound]
    highest_value = max(filtered_numbers) if filtered_numbers else None
    if highest_value is not None:
        output_filename = os.path.join(os.path.dirname(filename), "highest_value.txt")
        with open(output_filename, 'w') as f:
            f.write(f"Highest value without outliers: {highest_value}")
    return highest_value

# ----------------- FILE SELECTION -----------------
files = [
    "irobot_roomba.json",
    "line_clova_wave.json",
    "nature_remo.json",
    "qrio_hub.json",
    "xiaomi_mijia_led.json",
    "powerelectric_wi-fi_plug.json",
    "planex_smacam_outdoor.json"
]
files_with_size = [(file, os.path.getsize(os.path.join(folder_path, file))) for file in files if file.endswith('.json')]
smallest_files = sorted(files_with_size, key=lambda x: x[1])[:num_devices]

# ----------------- MAIN PROCESSING -----------------
all_filtered_data = []
device_flow_counts = {}

start_time = time.time()
current_directory = os.path.join(os.getcwd(), "0-Baseline")
os.makedirs(current_directory, exist_ok=True)
stop_event = multiprocessing.Event()
mem_start = psutil.virtual_memory().used / (1024 ** 2)
memory_process = multiprocessing.Process(target=memory_monitor, args=(current_directory, stop_event, mem_start))
memory_process.start()

for smallest_file in smallest_files:
    smallest_file_name = smallest_file[0]
    smallest_file_path = os.path.join(folder_path, smallest_file_name)
    print(f"\nInspecting file: {smallest_file_name}")

    filtered_data = []
    device_name = os.path.splitext(smallest_file_name)[0].replace('_', ' ').title()
    mapped_device_name = input_output_map.get(device_name, device_name)

    total_lines = 0
    with open(smallest_file_path, 'r') as f:
        for line in f:
            total_lines += 1
            try:
                json_data = json.loads(line)
                if "flows" in json_data:
                    flows_data = json_data["flows"]
                    source_ip = flows_data.get('sourceIPv4Address')
                    destination_ip = flows_data.get('destinationIPv4Address')

                    if not ((is_local_ip(source_ip) and is_local_ip(destination_ip)) or (is_ipv6(source_ip) or is_ipv6(destination_ip))):
                        filtered_flows = {k: v for k, v in flows_data.items() if k not in exclude_fields}
                        if include_cheating_columns and source_ip and destination_ip:
                            filtered_flows['sourceIPv4Address'] = ip_to_int(source_ip)
                            filtered_flows['destinationIPv4Address'] = ip_to_int(destination_ip)
                        filtered_data.append(filtered_flows)
            except json.JSONDecodeError:
                continue

    device_flow_counts[mapped_device_name] = len(filtered_data)
    print(f"  Total lines in file: {total_lines}")
    print(f"  Flows after filtering: {len(filtered_data)}")

    if filtered_data:
        df = pd.DataFrame(filtered_data)
        df['device_label'] = mapped_device_name
        all_filtered_data.append(df)

stop_event.set()
memory_process.join()

combined_df = pd.concat(all_filtered_data, ignore_index=True)
combined_numeric_df = combined_df.select_dtypes(include=[np.number]).dropna()
feature_columns = combined_numeric_df.columns.tolist()
X = combined_numeric_df[feature_columns]
y = combined_df['device_label'][:len(X)]
y_encoded = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------- RESULTS -----------------
print("\n=== Total flows per device after filtering ===")
for device, count in device_flow_counts.items():
    print(f"{device}: {count}")
print(f"\nTotal flows across all devices: {sum(device_flow_counts.values())}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=LabelEncoder().fit(y).classes_))
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1:.4f}")

# Confusion matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(conf_matrix_norm, display_labels=LabelEncoder().fit(y).classes_)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")
plt.tight_layout()
plt.savefig(f"{current_directory}/plot.png", dpi=300, transparent=True)

# ----------------- MEMORY AND TIME -----------------
end_time = time.time()
time_taken = end_time - start_time
peak_memory = highest_value_without_outliers(os.path.join(current_directory, 'memory_measurements.txt'))

print(f"Time taken: {time_taken:.2f}s")
print(f"Peak memory usage (filtered): {peak_memory / len(X):.6f} MB / flow")
