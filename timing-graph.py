import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ipaddress

# --- Configuration: adjust the folder path as needed ---
folder_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'

# --- Color choices to assign per protocol ---
color_choices = ["green", "blue", "red"]
protocol_color_map = {}  # Will hold mapping: protocol -> color

# --- Create a new folder in the current directory to save the plots ---
current_directory = os.getcwd()
save_folder = os.path.join(current_directory, "flow_plots")
os.makedirs(save_folder, exist_ok=True)

# --- Loop through all JSON files in the folder ---
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {file_path}")

        # --- Read the file and process the first 100 valid entries ---
        flows_to_plot = []
        with open(file_path, 'r') as f:
            count = 0
            for line in f:
                # Stop after we collect 100 valid flows
                if count >= 100:
                    break

                try:
                    entry = json.loads(line)
                    flow = entry.get("flows", {})

                    # Get source and destination IPs
                    src_ip = flow.get("sourceIPv4Address", "")
                    dst_ip = flow.get("destinationIPv4Address", "")

                    # Check if either source or destination IP is IPv6
                    if ":" in src_ip or ":" in dst_ip:
                        continue  # Skip the flow if it's IPv6

                    # Filter out if both source and destination IPs are private
                    if ipaddress.IPv4Address(src_ip).is_private and ipaddress.IPv4Address(dst_ip).is_private:
                        continue

                    # Parse the start time (assumed format: "YYYY-MM-DD HH:MM:SS.sss")
                    start_time = datetime.strptime(flow["flowStartMilliseconds"], "%Y-%m-%d %H:%M:%S.%f")
                    # Get duration in seconds (flowDurationMilliseconds is in seconds according to sample)
                    duration_secs = float(flow["flowDurationMilliseconds"])

                    # Get protocol number
                    protocol = int(flow["protocolIdentifier"])

                    # Get source and destination ports
                    src_port = flow.get("sourceTransportPort", "")
                    dst_port = flow.get("destinationTransportPort", "")

                    flows_to_plot.append({
                        "start_time": start_time,
                        "duration": duration_secs,
                        "protocol": protocol,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_port": src_port,
                        "dst_port": dst_port
                    })
                    count += 1

                except Exception as e:
                    print("Error processing line:", e)
                    continue

        if not flows_to_plot:
            print("No valid flows to plot.")
            continue

        # --- Build a color map for protocols encountered ---
        for flow in flows_to_plot:
            proto = flow["protocol"]
            if proto not in protocol_color_map:
                # Assign next available color from color_choices, or default to black
                if len(protocol_color_map) < len(color_choices):
                    protocol_color_map[proto] = color_choices[len(protocol_color_map)]
                else:
                    protocol_color_map[proto] = "black"  # fallback

        # --- Create the graph ---
        fig, ax = plt.subplots(figsize=(100, 60))

        # For each flow, plot a horizontal bar on a unique y-level
        for idx, flow in enumerate(flows_to_plot):
            start = flow["start_time"]
            # Duration: convert seconds to days for matplotlib date plotting (1 day = 86400 seconds)
            width = flow["duration"] / 86400
            color = protocol_color_map[flow["protocol"]]
            # Plot a horizontal bar:
            ax.barh(y=idx, width=width, left=start, height=0.4, color=color, edgecolor=color)  # border matches color
            
            # Add text labels with protocol, duration, source/destination IP, and port numbers
            label = f"Proto: {flow['protocol']} | Dur: {flow['duration']:.2f}s | Src: {flow['src_ip']}:{flow['src_port']} | Dst: {flow['dst_ip']}:{flow['dst_port']}"
            ax.text(start, idx - 0.5, label, va='center', fontsize=8, color='black', ha='left')

        # --- Format the x-axis to include both date and time ---
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        plt.xlabel("Time (Date and Time)")
        plt.ylabel("Flow Index")
        plt.title(f"Flow Timeline ({filename})")
        plt.yticks(range(len(flows_to_plot)), [f"Flow {i+1}" for i in range(len(flows_to_plot))])

        # Create a legend for protocols/colors
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=color, label=f"Protocol {proto}") for proto, color in protocol_color_map.items()]
        plt.legend(handles=legend_handles)

        plt.tight_layout()

        # --- Save the plot in the "flow_plots" folder ---
        save_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}_flow_timeline.png")
        plt.savefig(save_path)
        print(f"Plot saved to {os.path.abspath(save_path)}")
        
        # Clear the plot to avoid overlap with the next one
        plt.clf()
