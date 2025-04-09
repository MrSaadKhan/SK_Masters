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

num = 20  # Number of flows per file to process
arrow_max_length = 1.5
text_spacing = 0.1  # vertical spacing between stacked arrows
min_arrow_distance = 0.1  # minimum difference between arrows in seconds
# Note: min_arrow_distance is in seconds.

# --- Specify which files to process ---
selected_files = [
    # "xiaomi_mijia_led.json",
    # "nature_remo.json",
    # "irobot_roomba.json"

    "philips_hue_bridge.json",
    "bitfinder_awair_breathe_easy.json",
    "planex_smacam_outdoor.json"
]

# --- Helper Functions ---
def protocol_name(proto_number):
    return {6: "TCP", 17: "UDP"}.get(proto_number, str(proto_number))

# --- First Pass: Build Color Map ---
common_keys = {"TCP/80", "TCP/443", "UDP/53"}
primary_colors = {"TCP/80": "red", "TCP/443": "blue", "UDP/53": "green"}
unique_keys = set()

# Iterate over the selected files to build the unique keys
for filename in selected_files:
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r') as f:
        count = 0
        for line in f:
            if count >= num:
                break
            try:
                entry = json.loads(line)
                flow = entry.get("flows", {})
                src_ip = flow.get("sourceIPv4Address", "")
                dst_ip = flow.get("destinationIPv4Address", "")
                if ":" in src_ip or ":" in dst_ip:
                    continue
                # Filter out flows where both source and destination are private
                if ipaddress.IPv4Address(src_ip).is_private and ipaddress.IPv4Address(dst_ip).is_private:
                    continue
                _ = datetime.strptime(flow["flowStartMilliseconds"], "%Y-%m-%d %H:%M:%S.%f")
                _ = float(flow["flowDurationMilliseconds"])
                proto = int(flow["protocolIdentifier"])
                dst_port = flow.get("destinationTransportPort", "")
                key = f"{protocol_name(proto)}/{dst_port}"
                unique_keys.add(key)
                count += 1
            except:
                continue

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

# --- Second Pass: Process Files and Gather Global Time Range ---
# Dictionary to store flows per file
file_flow_dict = {}
global_min_time = None
global_max_time = None

for filename in selected_files:
    file_path = os.path.join(folder_path, filename)
    flows_to_plot = []
    with open(file_path, 'r') as f:
        count = 0
        for line in f:
            if count >= num:
                break
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
                duration = float(flow["flowDurationMilliseconds"])
                proto = int(flow["protocolIdentifier"])
                dst_port = flow.get("destinationTransportPort", "")
                key = f"{protocol_name(proto)}/{dst_port}"
                flows_to_plot.append({
                    "start_time": start_time,
                    "duration": duration,
                    "protocol": proto,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "dst_port": dst_port,
                    "key": key
                })
                count += 1
            except:
                continue
    # Only add file if flows exist
    if flows_to_plot:
        file_flow_dict[filename] = flows_to_plot
        times = [flow["start_time"] for flow in flows_to_plot]
        file_min_time = min(times)
        file_max_time = max(times)
        if global_min_time is None or file_min_time < global_min_time:
            global_min_time = file_min_time
        if global_max_time is None or file_max_time > global_max_time:
            global_max_time = file_max_time

# If no flows are present, exit early.
if global_min_time is None or global_max_time is None:
    print("No flows found in the selected files.")
    exit()

# Add time margin (we use 10% of the span)
time_margin = (global_max_time - global_min_time) * 0.1
x_min = global_min_time - time_margin
x_max = global_max_time + time_margin

# --- Create a Figure with Subplots (one per file) ---
n_files = len(file_flow_dict)
fig, axes = plt.subplots(n_files, 1, figsize=(12, 5 * n_files), sharex=True)
# In case there's only one file, ensure axes is iterable
if n_files == 1:
    axes = [axes]

# --- Plot Each File in its Subplot ---
# Define a resolution in days (for grouping flows by close time)
resolution = min_arrow_distance / (24 * 3600)

for ax, (filename, flows_to_plot) in zip(axes, file_flow_dict.items()):
    # Configure date formatter for the x-axis
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
    
    # Group flows by rounded timestamp so that flows within min_arrow_distance are stacked
    grouped_flows = {}
    for flow in flows_to_plot:
        base_x = mdates.date2num(flow["start_time"])
        rounded_base_x = round(base_x / resolution) * resolution
        grouped_flows.setdefault(rounded_base_x, []).append(flow)
    
    # Plot each group with vertical stacking
    for base_x, group in grouped_flows.items():
        # Optional: sort the group if needed (here by start time)
        group.sort(key=lambda f: f["start_time"])
        for idx, flow in enumerate(group):
            vertical_offset = idx * (arrow_max_length + text_spacing)
            color = universal_color_map.get(flow["key"], "black")
            ax.annotate("",
                        xy=(base_x, arrow_max_length + vertical_offset),
                        xytext=(base_x, vertical_offset),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            # Place label next to arrow tip
            text_x = base_x + 0.00005  # slight horizontal offset
            text_y = arrow_max_length + vertical_offset
            ax.text(text_x, text_y, flow["key"],
                    va='center', ha='left', fontsize=8)
    
    # Set consistent x-axis limits
    ax.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
    # Adjust y-limit based on maximum stacks within the current subplot
    max_stack = max(len(group) for group in grouped_flows.values())
    ax.set_ylim(-0.5, arrow_max_length + max_stack * (arrow_max_length + text_spacing) + 1)
    ax.set_title(f"Flow Timeline ({filename})")
    ax.set_xlabel("Time (Date and Time)")
    
    # Create a legend for this subplot with keys present in its flows
    keys_in_plot = {flow["key"] for flow in flows_to_plot}
    legend_handles = [mpatches.Patch(color=universal_color_map[k], label=k) for k in sorted(keys_in_plot)]
    ax.legend(handles=legend_handles, loc="upper right")

plt.tight_layout()
save_path = os.path.join(save_folder, "selected_flow_timelines.svg")
plt.savefig(save_path, format="svg", dpi=100)
print(f"Combined plot saved to {os.path.abspath(save_path)}")
plt.clf()
