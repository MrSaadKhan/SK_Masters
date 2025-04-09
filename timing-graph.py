import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ipaddress
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# --- Configuration ---
folder_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'
current_directory = os.getcwd()
save_folder = os.path.join(current_directory, "flow_plots")
os.makedirs(save_folder, exist_ok=True)

# Number of flows to sample initially per file
sample_num = 20
min_arrow_distance = 0.1  # minimum difference between arrows in seconds
# We no longer need arrow_max_length or text_spacing for vertical offset
# because we'll let the y-axis represent the actual count of flows.

# --- Specify which files to process ---
selected_files = [
    "philips_hue_bridge.json",
    "planex_smacam_outdoor.json"
]

# --- Helper Functions ---
def protocol_name(proto_number):
    return {6: "TCP", 17: "UDP"}.get(proto_number, str(proto_number))

# --- First Pass: Build Color Map ---
common_keys = {"TCP/80", "TCP/443", "UDP/53"}
primary_colors = {"TCP/80": "red", "TCP/443": "blue", "UDP/53": "green"}
unique_keys = set()

# Dictionary to store the 20-flow sample + time ranges for each file
sample_flow_dict = {}
time_ranges = {}

for filename in selected_files:
    file_path = os.path.join(folder_path, filename)
    flows_sample = []
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if count >= sample_num:
                break
            try:
                entry = json.loads(line)
                flow = entry.get("flows", {})
                src_ip = flow.get("sourceIPv4Address", "")
                dst_ip = flow.get("destinationIPv4Address", "")
                # Skip IPv6 or flows where src/dst are both private
                if ":" in src_ip or ":" in dst_ip:
                    continue
                if ipaddress.IPv4Address(src_ip).is_private and ipaddress.IPv4Address(dst_ip).is_private:
                    continue
                start_time = datetime.strptime(flow["flowStartMilliseconds"], "%Y-%m-%d %H:%M:%S.%f")
                _ = float(flow["flowDurationMilliseconds"])  # just to ensure it parses
                proto = int(flow["protocolIdentifier"])
                dst_port = flow.get("destinationTransportPort", "")
                key = f"{protocol_name(proto)}/{dst_port}"
                
                flows_sample.append({
                    "start_time": start_time,
                    "proto": proto,
                    "dst_port": dst_port,
                    "key": key,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip
                })
                unique_keys.add(key)
                count += 1
            except:
                continue

    # If we gathered flows, record them along with their min/max times
    if flows_sample:
        sample_flow_dict[filename] = flows_sample
        sample_times = [f["start_time"] for f in flows_sample]
        time_ranges[filename] = (min(sample_times), max(sample_times))

# Assign primary colors first
universal_color_map = {}
for key in common_keys:
    if key in unique_keys:
        universal_color_map[key] = primary_colors[key]

# Assign colors to remaining keys
secondary_keys = sorted(unique_keys - common_keys)
if secondary_keys:
    colormap = cm.get_cmap("tab20", len(secondary_keys))
    for idx, key in enumerate(secondary_keys):
        universal_color_map[key] = colormap(idx)

print("Universal color map (protocol/port):")
for key, color in universal_color_map.items():
    print(f"{key}: {color}")

# --- Determine Baseline Time Range ---
# Find the file whose sample has the largest time difference
baseline_file = None
max_diff = timedelta(0)
baseline_min, baseline_max = None, None
for filename, (min_time, max_time) in time_ranges.items():
    diff = max_time - min_time
    if diff > max_diff:
        max_diff = diff
        baseline_file = filename
        baseline_min = min_time
        baseline_max = max_time

print(f"Baseline file: {baseline_file} with time range {baseline_min} to {baseline_max}")

# --- Second Pass: Gather Flows for Plotting ---
# For the baseline device, keep the 20 flows from the sample.
# For the other devices, collect ALL flows within [baseline_min, baseline_max].
file_flow_dict = {}

for filename in selected_files:
    file_path = os.path.join(folder_path, filename)
    flows_to_plot = []

    if filename == baseline_file:
        flows_to_plot = sample_flow_dict[filename]
    else:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    flow = entry.get("flows", {})
                    src_ip = flow.get("sourceIPv4Address", "")
                    dst_ip = flow.get("destinationIPv4Address", "")
                    if ":" in src_ip or ":" in dst_ip:
                        continue
                    if ipaddress.IPv4Address(src_ip).is_private and ipaddress.IPv4Address(dst_ip).is_private:
                        continue
                    start_time = datetime.strptime(flow["flowStartMilliseconds"], "%Y-%m-%d %H:%M:%S.%f")
                    if baseline_min <= start_time <= baseline_max:
                        proto = int(flow["protocolIdentifier"])
                        dst_port = flow.get("destinationTransportPort", "")
                        key = f"{protocol_name(proto)}/{dst_port}"
                        flows_to_plot.append({
                            "start_time": start_time,
                            "proto": proto,
                            "dst_port": dst_port,
                            "key": key,
                            "src_ip": src_ip,
                            "dst_ip": dst_ip
                        })
                except:
                    continue

    if flows_to_plot:
        file_flow_dict[filename] = flows_to_plot

# --- Plotting Setup ---
global_min_time = baseline_min
global_max_time = baseline_max
time_margin = (global_max_time - global_min_time) * 0.1
x_min = global_min_time - time_margin
x_max = global_max_time + time_margin

n_files = len(file_flow_dict)
fig, axes = plt.subplots(n_files, 1, figsize=(12, 5 * n_files), sharex=True)
if n_files == 1:
    axes = [axes]

# Grouping resolution in days
resolution = min_arrow_distance / (24 * 3600)

# --- Plot Each File in its Subplot ---
for ax, (filename, flows_to_plot) in zip(axes, file_flow_dict.items()):
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
    
    # Group flows by close timestamps
    grouped_flows = {}
    for flow in flows_to_plot:
        base_x = mdates.date2num(flow["start_time"])
        rounded_base_x = round(base_x / resolution) * resolution
        grouped_flows.setdefault(rounded_base_x, []).append(flow)
    
    # Plot each group, stacking flows with integer y-values
    for base_x, group in grouped_flows.items():
        group.sort(key=lambda f: f["start_time"])
        for idx, flow in enumerate(group):
            color = universal_color_map.get(flow["key"], "black")
            # Arrows from y=idx to y=idx+1
            ax.annotate(
                "",
                xy=(base_x, idx + 1),
                xytext=(base_x, idx),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
            )
    
    # X-limit is consistent for all subplots
    ax.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
    
    # Y-limit: from 0 up to the max number of flows that appear together
    max_flows_at_once = max(len(g) for g in grouped_flows.values()) if grouped_flows else 1
    ax.set_ylim(0, max_flows_at_once + 1)
    
    # Labeling
    ax.set_ylabel("Number of flows at the same time")
    ax.set_title(f"Flow Timeline ({filename})")
    ax.set_xlabel("Time (Date and Time)")

    # Create a legend for this subplot
    keys_in_plot = {flow["key"] for flow in flows_to_plot}
    legend_handles = [mpatches.Patch(color=universal_color_map[k], label=k) for k in sorted(keys_in_plot)]
    ax.legend(handles=legend_handles, loc="upper right")

plt.tight_layout()
save_path = os.path.join(save_folder, "selected_flow_timelines.svg")
plt.savefig(save_path, format="svg", dpi=100)
print(f"Combined plot saved to {os.path.abspath(save_path)}")
plt.clf()
