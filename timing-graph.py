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
text_spacing = 0.1  # vertical spacing between stacked texts

# --- Helper Functions ---
def protocol_name(proto_number):
    return {6: "TCP", 17: "UDP"}.get(proto_number, str(proto_number))

# --- First Pass: Build Color Map ---
common_keys = {"TCP/80", "TCP/443", "UDP/53"}
primary_colors = {"TCP/80": "red", "TCP/443": "blue", "UDP/53": "green"}
unique_keys = set()

json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
for filename in json_files:
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

# --- Second Pass: Plot Flows ---
for filename in json_files:
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

    if not flows_to_plot:
        continue

    times = [flow["start_time"] for flow in flows_to_plot]
    min_time = min(times)
    max_time = max(times)
    time_margin = (max_time - min_time) * 0.1 if max_time != min_time else timedelta(seconds=1)
    x_min = min_time - time_margin
    x_max = max_time + time_margin

    time_span_days = (max_time - min_time).total_seconds() / 86400
    fig_width = max(20, time_span_days * 1000)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))

    # Group flows by timestamp
    grouped_flows = {}
    for flow in flows_to_plot:
        base_x = mdates.date2num(flow["start_time"])
        grouped_flows.setdefault(base_x, []).append(flow)

    for base_x, flows in grouped_flows.items():
        n = len(flows)
        # Draw arrows with shortest on top
        for i, flow in reversed(list(enumerate(flows))):
            height = ((i + 1) / n) * arrow_max_length if n > 1 else arrow_max_length
            color = universal_color_map.get(flow["key"], "black")

            ax.annotate("",
                        xy=(base_x, height),
                        xytext=(base_x, 0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

            # Stack text labels vertically above arrowhead
            text_y = height + text_spacing
            ax.text(base_x + 0.0001, text_y, flow["key"],
                    rotation=0, va='bottom', ha='left', fontsize=8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, arrow_max_length + n * text_spacing + 1)
    plt.xlabel("Time (Date and Time)")
    plt.title(f"Flow Timeline ({filename})")

    keys_in_plot = {flow["key"] for flow in flows_to_plot}
    legend_handles = [mpatches.Patch(color=universal_color_map[k], label=k) for k in sorted(keys_in_plot)]
    plt.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}_flow_timeline.svg")
    plt.savefig(save_path, format="svg", dpi=100)
    print(f"Plot saved to {os.path.abspath(save_path)}")
    plt.clf()
