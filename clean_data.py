import pandas as pd
import ipaddress
import os

def process_flow(flow):
    try:
        src_ip = ipaddress.IPv4Address(flow.get('sourceIPv4Address', ''))
        dst_ip = ipaddress.IPv4Address(flow.get('destinationIPv4Address', ''))

        # Filter condition: drop if both source and dest are private
        if src_ip.is_private and dst_ip.is_private:
            return None

        # Replace either one if it's private
        if src_ip.is_private:
            flow['sourceIPv4Address'] = '192.168.0.1'
        if dst_ip.is_private:
            flow['destinationIPv4Address'] = '192.168.0.1'

        return flow
    except ValueError:
        return None  # Skip invalid IPs

def clean_data(target_file):
    print("Cleaning data... for " + os.path.split(target_file)[1])
    
    # Step 1: Read Data in Chunks
    chunk_size = 10000  # Adjust the chunk size based on your system's memory
    filtered_data_chunks = pd.read_json(target_file, lines=True, chunksize=chunk_size)

    # Step 2: Process Each Chunk
    output = pd.DataFrame()

    for chunk in filtered_data_chunks:
        # Handle cases where 'flows' might contain a float instead of a dictionary
        chunk['destinationIPv4Address'] = chunk['flows'].apply(lambda x: x.get('destinationIPv4Address') if isinstance(x, dict) else None)
        chunk = chunk[chunk['destinationIPv4Address'].apply(lambda x: isinstance(x, str) and ':' not in x)]
        
        destination_ips = chunk['flows'].apply(lambda x: ipaddress.IPv4Address(x['destinationIPv4Address']))
        chunk = chunk[~(destination_ips.apply(lambda x: x.is_multicast) | destination_ips.apply(lambda x: x.is_private))]

        # Apply the process_flow logic to modify source and destination IPs
        chunk['flows'] = chunk['flows'].apply(process_flow)

        # Remove rows where process_flow returned None (i.e., both source and destination were private)
        chunk = chunk[chunk['flows'].notnull()]

        # Remove sourceMacAddress, destinationMacAddress, and sourceIPv4Address from flows
        chunk['flows'] = chunk['flows'].apply(lambda x: {k: v for k, v in x.items() if k not in ['sourceMacAddress', 'destinationMacAddress']})
        
        df1 = pd.DataFrame(chunk)
        df1 = df1[["flows"]]
        output = output._append(df1)

    output1 = output.values.flatten().tolist()

    # Clean the output (convert values to strings, strip symbols)
    output1 = [
        {
            key: str(value).replace(',', '').replace('}', '').replace('{', '').replace("]", '').replace("[", '').replace("'", '') 
            for key, value in item.items()
        } 
        for item in output1
    ]

    num_elements = len(output)
    print(str(num_elements) + ' flows!')
    return output1, num_elements
