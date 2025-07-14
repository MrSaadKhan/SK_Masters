import os
import prepare_data
import group_data

# def save_individual_file(list_of_lists, name, path):
#     path = os.path.join(path, f'{name}.txt')
#     with open(path, 'w') as file:
#         for inner_list in list_of_lists:
#             line = ','.join(map(str, inner_list))
#             file.write(line + '\n')

def save_individual_file(list_of_dicts, name, path):
    path = os.path.join(path, f'{name}.txt')
    with open(path, 'w') as file:
        # Write each dictionary on a new line, without formatting
        for data_dict in list_of_dicts:
            # Convert the dictionary to a custom formatted string (no {}, "", etc.)
            formatted_line = ' '.join(f"{key}: {value}" for key, value in data_dict.items())
            file.write(f"{formatted_line}\n")


   
def save_files(seen, unseen, name, path):
    save_individual_file(seen, name + '_seen', path)
    save_individual_file(unseen, name + '_unseen', path)

def get_data(save_path, device):
    seen = read_file_as_list_of_lists(os.path.join(save_path, device + "_seen.txt"))
    unseen = read_file_as_list_of_lists(os.path.join(save_path, device + "_unseen.txt"))
    return seen, unseen

def read_file_as_list_of_lists(file_path):
    list_of_lists = []
    try:
        with open(file_path, 'r') as file:
            # Read each line, strip newline characters, and keep it as a single string
            list_of_lists = [[line.strip()] for line in file]
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except IOError as e:
        print(f"An I/O error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return list_of_lists


if __name__ == "__main__":
    file_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'

    if not os.path.exists(file_path):
        file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

    cwd = os.getcwd()
    
    group_option=0
    time_group=0
    num2word_option=0   # Unlikely to be implemented
    
    if group_option == 0:
        group_path = 'ungrouped'
        save_path = os.path.join(cwd, 'preprocessed_data', group_path)

    else:
        group_path = 'grouped'
        time_path = str(time_group)
        save_path = os.path.join(cwd, 'preprocessed_data', group_path, time_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # List of files to exclude
    exclusion_list = ['sony_network_camera.json', 'mouse_computer_room_hub.json', 'planex_camera_one_shot!.json']

    device_high = 10#22
    device_low = 0
    num_group = 5
    stride = 1


    all_devices = os.listdir(file_path)
    filtered_devices = [device for device in all_devices if device not in exclusion_list]
    devices_sorted = sorted(filtered_devices, key=lambda device: os.path.getsize(os.path.join(file_path, device)))
    device_list = devices_sorted[device_low:device_high]
    print(device_list)

    lengths = []
    for device in device_list:
        print(os.path.join(file_path, device))

        seen, unseen = prepare_data.prepare_data(os.path.join(file_path, device))
        # print(seen)

        length = ((len(seen), len(unseen)))
        print(f"Device: {device} \nSeen Length: {length[0]}\nUnseen Length: {length[1]}")

        # print(f"{seen}")
        seen = group_data.group_data_number(seen, 5, 1)
        unseen = group_data.group_data_number(unseen, 5, 1)
        save_files(seen, unseen, device, save_path)
### ADD grouping logic HERE!!!!!!!!!!
    print("Complete :)")