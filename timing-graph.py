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
sample_num = 30
min_arrow_distance = 0.1  # minimum difference between arrows in seconds

# --- Specify which files to process ---
selected_files = [
    "philips_hue_bridge.json",
    "planex_smacam_outdoor.json"
]

# --- Helper Functions ---
def protocol_name(proto_number, dst_port):
    # Add more protocol names here
    protocol_map = {
        6: "TCP",  # TCP
        17: "UDP",  # UDP
        1: "ICMP",  # ICMP
    }
    # Define human-readable protocol names for common ports
    port_map = {
        # "80": "HTTP",  # TCP/80 -> HTTP
        # "443": "HTTPS",  # TCP/443 -> HTTPS
        # "53": "DNS",  # UDP/53 -> DNS
        # "22": "SSH",  # TCP/22 -> SSH
        # "21": "FTP",  # TCP/21 -> FTP
        # "123": "NTP",
        # "1900": "SSDP",
        # "5353": "Multicast DNS",
        # "10001": "SCP"
    }
    
    proto_str = protocol_map.get(proto_number, str(proto_number))
    port_str = port_map.get(str(dst_port), str(dst_port))
    
    # Return combined string with protocol and port name
    return f"{proto_str}/{dst_port}" if proto_str != "TCP" or str(dst_port) not in port_map \
           else f"{proto_str}/{dst_port} ({port_str})"

# --- First Pass: Build Color Map ---
common_keys = {"TCP/80", "TCP/443", "UDP/53"}
primary_colors = {"TCP/80": "red", "TCP/443": "blue", "UDP/53": "green"}
unique_keys = set()

# Dictionary to store the sample flows and time ranges for each file
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
                key = protocol_name(proto, dst_port)
                
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

    if flows_sample:
        sample_flow_dict[filename] = flows_sample
        sample_times = [f["start_time"] for f in flows_sample]
        time_ranges[filename] = (min(sample_times), max(sample_times))

# Assign primary colors first
universal_color_map = {}
for key in common_keys:
    if key in unique_keys:
        universal_color_map[key] = primary_colors[key]

# Assign colors to remaining keys using a colormap
secondary_keys = sorted(unique_keys - common_keys)
if secondary_keys:
    colormap = cm.get_cmap("tab20", len(secondary_keys))
    for idx, key in enumerate(secondary_keys):
        universal_color_map[key] = colormap(idx)

print("Universal color map (protocol/port):")
for key, color in universal_color_map.items():
    print(f"{key}: {color}")

# --- Determine Baseline Time Range ---
# Find the file whose sample has the largest time difference; used to determine global time range
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
# For the baseline file, use the sample flows; for the others, collect ALL flows within [baseline_min, baseline_max].
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
                        key = protocol_name(proto, dst_port)
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
# Using the baseline's time range to keep a consistent x-axis across all plots
global_min_time = baseline_min
global_max_time = baseline_max
time_margin = (global_max_time - global_min_time) * 0.1
x_min = global_min_time - time_margin
x_max = global_max_time + time_margin

# Grouping resolution in days
resolution = min_arrow_distance / (24 * 3600)

# --- Create Separate Plot for Each File ---
for filename, flows_to_plot in file_flow_dict.items():
    # Create new figure for each file
    fig, ax = plt.subplots(figsize=(12.6, 5))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    
    # Group flows by close timestamps
    grouped_flows = {}
    for flow in flows_to_plot:
        base_x = mdates.date2num(flow["start_time"])
        rounded_base_x = round(base_x / resolution) * resolution
        grouped_flows.setdefault(rounded_base_x, []).append(flow)
    
    # Plot each group, stacking flows with integer y-values.
    # (Note: The inner loop sets idx = 0 for each group; adjust if you require stacking.)
    for base_x, group in grouped_flows.items():
        group.sort(key=lambda f: f["start_time"])
        for idx, flow in enumerate(group):
            color = universal_color_map.get(flow["key"], "black")
            # Arrows from y=idx to y=idx+1; current version resets idx to 0 for each group.
            ax.annotate(
                "",
                xy=(base_x, 0 + 1),
                xytext=(base_x, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
            )
            print(f"{filename} Flow: {(base_x, 1)}")
    
    # Set x-axis limits (consistent across all files)
    ax.set_xlim(mdates.date2num(x_min), mdates.date2num(x_max))
    
    # Y-axis settings
    max_flows_at_once = 1  # adjust if you require stacking beyond one flow per time instance
    ax.set_ylim(0, max_flows_at_once + 1)
    # ax.set_yticks(range(max_flows_at_once + 1))
    ax.set_yticks([])
    ax.set_ylabel('')
    # Labeling for this plot
    # ax.set_ylabel("Number of flows at one time instance")
    ax.set_title(f"Flow Timeline ({filename})")
    ax.set_xlabel("Time (Date and Time)")
    
    # Create legend for protocol/port used in the flows
    keys_in_plot = {flow["key"] for flow in flows_to_plot}
    legend_handles = [mpatches.Patch(color=universal_color_map.get(k, "grey"), label=k)
                      for k in sorted(keys_in_plot)]
    
    # Add legends to the plot:
    legend1 = ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    ax.add_artist(legend1)
    
    flow_count_text = f"Flow Count: {len(flows_to_plot)}"
    legend2 = ax.legend(
        handles=[mpatches.Patch(color='white', label=flow_count_text)],
        loc="upper left",
        frameon=True
    )
    
    plt.tight_layout()
    # Save this plot as a separate SVG file
    save_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}_flow_timeline.svg")
    plt.savefig(save_path, format="svg", dpi=300)
    print(f"Plot for {filename} saved to {os.path.abspath(save_path)}")
    plt.clf()
