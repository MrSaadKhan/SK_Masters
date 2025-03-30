import os

def get_data(save_path, device):
    seen = read_file_as_list_of_lists(os.path.join(save_path, device + "_seen.txt"))
    unseen = read_file_as_list_of_lists(os.path.join(save_path, device + "_unseen.txt"))
    return seen, unseen

def get_seen_data(save_path, device):
    seen = read_file_as_list_of_lists(os.path.join(save_path, device + "_seen.txt"))
    return seen

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