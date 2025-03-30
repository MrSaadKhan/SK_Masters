import pandas as pd
import ipaddress
import os
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
     
        # Remove sourceMacAddress, destinationMacAddress, and sourceIPv4Address from flows
        chunk['flows'] = chunk['flows'].apply(lambda x: {k: v for k, v in x.items() if k not in ['sourceMacAddress', 'destinationMacAddress', 'sourceIPv4Address']})
        
        df1 = pd.DataFrame(chunk)
        df1 = df1[["flows"]]
        #output.extend(chunk)
        output = output._append(df1)

    output1 = output.values.flatten().tolist()

    # Change the input correctly
    # print(output1)

    # output1 = [ '[' + str(item).replace(',', ' ').replace('}', '').replace('{', '').replace("]", '').replace("[", '').replace("'", '') + ']' for item in output1]
    output1 = [{key: str(value).replace(',', '').replace('}', '').replace('{', '').replace("]", '').replace("[", '').replace("'", '') 
                for key, value in item.items()} 
                for item in output1]
    # print(output1)
    # output1 = [[s.strip('[]')] for s in output1]
    #num_elements = sum(1 for sublist in output1 if sublist[0] is not None)
    # num_elements = len(list(filter(lambda x: x[0] is not None, output1)))
    
    num_elements = len(output)
    
    print(str(num_elements) +' '+'flows!')
    # print(output1)
    return output1, num_elements

# if __name__ == "__main__":
#     print(clean_data("/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix/planex_smacam_pantilt.json"))
    