import os
import json
import ipaddress
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def process_flow(flow):
    try:
        src_ip = ipaddress.IPv4Address(flow.get('sourceIPv4Address', ''))
        dst_ip = ipaddress.IPv4Address(flow.get('destinationIPv4Address', ''))

        if src_ip.is_private and dst_ip.is_private:
            return None
        if src_ip.is_private:
            flow['sourceIPv4Address'] = '192.168.0.1'
        if dst_ip.is_private:
            flow['destinationIPv4Address'] = '192.168.0.1'

        return flow
    except ValueError:
        return None

def parse_timestamp_to_ms(ts_str):
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        return int(dt.timestamp() * 1000)
    except Exception:
        return -1

def extract_valid_start_times(filepath):
    chunk_size = 10000
    start_times = []

    try:
        file_size = sum(1 for _ in open(filepath, 'r', encoding='utf-8', errors='ignore'))
        chunk_iter = pd.read_json(filepath, lines=True, chunksize=chunk_size)

        for chunk in tqdm(chunk_iter, total=(file_size // chunk_size) + 1, desc=f"Lines in {os.path.basename(filepath)}", leave=False):
            chunk['flows'] = chunk['flows'].apply(lambda x: x if isinstance(x, dict) else None)
            chunk = chunk.dropna(subset=['flows'])

            chunk['flows'] = chunk['flows'].apply(process_flow)
            chunk = chunk[chunk['flows'].notnull()]

            for flow in chunk['flows']:
                ts = flow.get('flowStartMilliseconds', '')
                start_time = parse_timestamp_to_ms(ts)
                if start_time >= 0:
                    start_times.append(start_time)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return start_times

def compute_average_frequency(start_times):
    if len(start_times) < 2:
        return 0.0
    start_times.sort()
    duration = start_times[-1] - start_times[0]
    if duration == 0:
        return 0.0
    return len(start_times) / duration  # flows per millisecond

def main(directory, exclusion_list, n_smallest=None):
    all_files = [f for f in os.listdir(directory) if f.endswith('.json') and f not in exclusion_list]
    all_paths = [os.path.join(directory, f) for f in all_files]

    if n_smallest:
        all_paths = sorted(all_paths, key=os.path.getsize)[:n_smallest]

    total_start_times = []

    print(f"\nProcessing {len(all_paths)} file(s)...\n")
    for filepath in tqdm(all_paths, desc="Files", unit="file"):
        start_times = extract_valid_start_times(filepath)
        total_start_times.extend(start_times)

    if not total_start_times:
        print("\nNo valid flows found.")
        return

    frequency = compute_average_frequency(total_start_times)
    print(f"\nTotal valid flows: {len(total_start_times)}")
    print(f"Average flow arrival frequency: {frequency:.6f} flows/ms ({frequency * 1000:.3f} flows/sec)")

if __name__ == "__main__":
    # Set your path and exclusions
    target_directory = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'
    exclusions = ['sony_network_camera.json', 'mouse_computer_room_hub.json', 'planex_camera_one_shot!.json']
    
    # Set to None to process all files, or an integer to limit to N smallest
    n_smallest_files = None  # or e.g. 5

    main(target_directory, exclusions, n_smallest=n_smallest_files)
