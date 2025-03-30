# import pandas as pd
# import ipaddress
# import os
# import time
import re
from datetime import datetime, timedelta

# def group_data(output):

#     return sorted_output


def group_data(output, time_group = 5):
    # print("Cleaning data... for " + os.path.split(target_file)[1])
    # start_time = time.time()
    
    # # Step 1: Read Data in Chunks
    # chunk_size = 10000  # Adjust the chunk size based on your system's memory
    # filtered_data_chunks = pd.read_json(target_file, lines=True, chunksize=chunk_size)

    # # Step 2: Process Each Chunk
    # output = []

    # for chunk in filtered_data_chunks:
    #     # Handle cases where 'flows' might contain a float instead of a dictionary
    #     chunk['destinationIPv4Address'] = chunk['flows'].apply(lambda x: x.get('destinationIPv4Address') if isinstance(x, dict) else None)
    #     chunk = chunk[chunk['destinationIPv4Address'].apply(lambda x: isinstance(x, str) and ':' not in x)]
        
    #     destination_ips = chunk['flows'].apply(lambda x: ipaddress.IPv4Address(x['destinationIPv4Address']))
    #     chunk = chunk[~(destination_ips.apply(lambda x: x.is_multicast) | destination_ips.apply(lambda x: x.is_private))]
        
    #     # Remove source MAC address and source IPv4 address fields
    #     chunk['flows'] = chunk['flows'].apply(lambda x: {key: value for key, value in x.items() if key not in ['sourceMacAddress', 'sourceIPv4Address']})
        
    #     # Convert 'flows' column to the required format
    #     chunk['flows'] = chunk['flows'].apply(lambda x: [str(x).replace(",", " ").replace("}", "").replace("{", "").replace("]", "").replace("[", "").replace("'", "").strip("[]")])
    #     output.extend(chunk['flows'].tolist())

    # num_elements = len([sublist for sublist in output if sublist[0] is not None])
    # print(f"{num_elements} flows!")

    # end_time = time.time()
    # execution_time = end_time - start_time
    # output = group_data(output)

    time_format = '%Y-%m-%d %H:%M:%S.%f'
    # Grouped in intervals. 5 mins by default
    # time_group = 5
    print(f"Grouping data by {time_group} mins")
    i = 0

    sorted_output = []
    line = []
    prev_match = None

    for item in output:
        match = re.match(r'flowStartMilliseconds: [^ ]* [^ ]*', item[0]).group(0).replace('flowStartMilliseconds: ', '')
        match = datetime.strptime(match, time_format)
        if i == 0:
            line.append(output[i][0])
            prev_match = match
            i += 1
            continue

        # print(match)
        
        diff = match - prev_match
        # print(diff)

        if diff <= timedelta(minutes=time_group):
            line.append(output[i][0])
        else:
            sorted_output.append(line)
            # line.clear()
            line = []
            line.append(output[i][0])

        prev_match = match
        # print(output[i][0])
        i += 1
        # print(len(sorted_output))

    sorted_output.append(line)

    return sorted_output

# def print_stats(output, num_elements, execution_time):
#     print(f"Execution Time: {execution_time:.2f} seconds")

# target_file = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data\au_wireless_adapter.json'

# output, num_elements = clean_data(target_file)



# for item1 in sorted_output:
#     for item in item1:
#         match = re.match(r'flowStartMilliseconds: [^ ]* [^ ]*', item).group(0).replace('flowStartMilliseconds: ', '')
#         print(match, end=" ")
#     print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK')


# print(output)
