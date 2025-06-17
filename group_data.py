# import pandas as pd
# import ipaddress
# import os
# import time
import re
from datetime import datetime, timedelta

# def group_data(output):

#     return sorted_output


def group_data_time(output, time_group = 5):
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

def group_data_number(data, window=5, stride=1):
    grouped = []
    for i in range(0, len(data) - window + 1, stride):
        merged = {}
        for j in range(window):
            # Optional: Prefix keys with index if you want to preserve all info
            for k, v in data[i + j].items():
                merged[f"{k}_{j}"] = v  # Disambiguate overlapping keys
        grouped.append(merged)
    return grouped
