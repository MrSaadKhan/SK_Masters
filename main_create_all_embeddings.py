import os
import time
# import create_fasttext_embeddings
import create_bert_embeddings
import create_gpt_embeddings
import create_mamba_embeddings
import create_plots
import gc

import multiprocessing
import time
import psutil

import matplotlib.pyplot as plt

def memory_monitor(output_dir, stop_event, mem_start):
    output_file = os.path.join(output_dir, "memory_measurements.txt")
    with open(output_file, "a") as f:
        while not stop_event.is_set():  # Continue until the event is set
            mem_usage = psutil.virtual_memory().used / (1024 ** 2)
            mem_usage -= mem_start
            f.write(f"{mem_usage}\n")  # Append each measurement
            time.sleep(0.001)  # 1 millisecond delay
    print(f"Memory measurements saved to {output_file}")

def plot_numbers_from_file(file_path):
    file_path = os.path.join(file_path, "memory_measurements.txt")
    # List to hold the numbers from each line
    numbers = []
    
    # Read numbers from the file
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Convert line to float and add to list
                numbers.append(float(line.strip()))
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
    
    # Plot the numbers
    plt.figure()
    plt.plot(numbers, marker='x', linestyle='-', markersize=5)  # Adjust marker size if needed
    plt.title("Numbers from File")
    plt.xlabel("Milliseconds")
    plt.ylabel("Memory (MB)")
    
    # Set y-axis to start from zero
    # plt.ylim(bottom=0)

    save_path = os.path.join(os.path.dirname(file_path), "plot.png")
    plt.savefig(save_path, dpi=600)  # Set dpi for sharper image
    plt.close()
    print(f"Plot saved at {save_path}")

def highest_value_without_outliers(filename):
    # Read numbers from file
    with open(filename, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    
    if not numbers:
        return None  # Return None if file is empty or contains no valid numbers

    # # Calculate quartiles
    # sorted_numbers = sorted(numbers)
    # q1 = sorted_numbers[int(len(sorted_numbers) * 0.25)]
    # q3 = sorted_numbers[int(len(sorted_numbers) * 0.9)]
    # iqr = q3 - q1

    # # Define outlier range
    # lower_bound = q1 - 1.5 * iqr
    # upper_bound = q3 + 1.5 * iqr

    # # Filter out outliers
    # filtered_numbers = [num for num in sorted_numbers if lower_bound <= num <= upper_bound]

    # Determine the highest value in the filtered list
    highest_value = max(numbers) #if filtered_numbers else None

    # Save the result in the same directory as the input file
    if highest_value is not None:
        output_filename = os.path.join(os.path.dirname(filename), "highest_value.txt")
        with open(output_filename, 'w') as output_file:
            output_file.write(f"Highest value without outliers: {highest_value}")
    
    return highest_value

def save_number_to_text(value, directory, filename="time.txt"):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    
    with open(file_path, 'w') as file:
        file.write(str(value))

def main(device_low, device_high, save_dir, data_path, group_option, word_embedding_option, window_size, slide_length, vector_size = 768):
    # Directory path to read files from
    file_path = r'/home/iotresearch/saad/data/KDDI-IoT-2019/ipfix'

    if not os.path.exists(file_path):
        file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

    # List of files to exclude
    exclusion_list = ['sony_network_camera.json', 'mouse_computer_room_hub.json', 'planex_camera_one_shot!.json']

    # Get a list of all devices in the directory
    all_devices = os.listdir(file_path)
    # Filter out devices that are in the exclusion list
    filtered_devices = [device for device in all_devices if device not in exclusion_list]
    # Sort devices by file size
    devices_sorted = sorted(filtered_devices, key=lambda device: os.path.getsize(os.path.join(file_path, device)))
    # Select the five smallest devices from the sorted list
    device_list = devices_sorted[device_low:device_high]
    print(device_list)

    gc.collect()
    start_time = time.time()

    # new_dir = os.path.join(save_dir, 'FastText')
    # if not os.path.exists(new_dir):
    #     os.mkdir(new_dir)

    # model_filename = create_fasttext_embeddings.train_fasttext_model(file_path, device_list, new_dir, data_path, group_option, word_embedding_option, vector_size)
    # fast_text_training_time = time.time() - start_time
    # fast_text_training_mem_usage = memory_usage(-1, interval=0.1, include_children=True)[0] - start_memory

    # gc.collect()
    # start_memory = memory_usage(-1, interval=0.1, include_children=True)[0]
    # start_time = time.time()

    # seen_ft, unseen_ft = create_fasttext_embeddings.create_embeddings(model_filename, file_path, device_list, data_path, vector_size)
    # fast_text_embeddings_creation_time = time.time() - start_time
    # fast_text_embeddings_creation_mem_usage = memory_usage(-1, interval=0.1, include_children=True)[0] - start_memory

    gc.collect()
    # start_memory = memory_usage(-1, interval=0.1, include_children=True)[0]
    # start_time = time.time()

    # Create BERT embeddings using pretrained model
    # devices_lengths = [seen, unseen]
    seen_ft = 0
    unseen_ft = 0

    # new_dir = os.path.join(save_dir, 'FastText')
    # FastText_path = new_dir
    # if not os.path.exists(new_dir):
    #     os.mkdir(new_dir)
    
    # mem_start_FastText = psutil.virtual_memory().used / (1024 ** 2)
    # stop_event = multiprocessing.Event()  # Create the stop event
    # process = multiprocessing.Process(target=memory_monitor, args=(new_dir, stop_event, mem_start_FastText))
    # process.start()
    # start_time = time.time()
    
    # model_filename = create_fasttext_embeddings.train_fasttext_model(file_path, device_list, new_dir, data_path, group_option, word_embedding_option, window_size, slide_length, vector_size)

    # FastText_training_time = time.time() - start_time
    # print(f"FastText training time: {FastText_training_time}")
    # seen_ft, unseen_ft = create_fasttext_embeddings.create_embeddings(model_filename, file_path, device_list, data_path, window_size, slide_length, vector_size)
    # print(f"FastText embedding time: {(time.time() - FastText_training_time)/(seen_ft+unseen_ft)}")
    # print(f"FastText TOTAL time per flow: {(time.time() - start_time)/(seen_ft+unseen_ft)}")
    # print(f"FastText TOTAL memory per flow: {highest_value_without_outliers(os.path.join(FastText_path, 'memory_measurements.txt'))/(seen_ft+unseen_ft)}")

    
    # stop_event.set()  # Signal the memory monitor to stop
    # process.join()
    # plot_numbers_from_file(FastText_path)

    # new_dir = os.path.join(save_dir, 'BERT')
    # bert_path = new_dir
    # if not os.path.exists(new_dir):
    #     os.mkdir(new_dir)
    
    # mem_start_BERT = psutil.virtual_memory().used / (1024 ** 2)
    # stop_event = multiprocessing.Event()  # Create the stop event
    # process = multiprocessing.Process(target=memory_monitor, args=(new_dir, stop_event, mem_start_BERT))
    # process.start()
    # start_time = time.time()
    # seen, unseen, temp = create_bert_embeddings.create_embeddings(file_path, device_list, new_dir, data_path, group_option, word_embedding_option, window_size, slide_length, vector_size)
    # bert_time = time.time() - start_time
    # stop_event.set()  # Signal the memory monitor to stop
    # process.join()
    # plot_numbers_from_file(bert_path)

    # new_dir = os.path.join(save_dir, 'GPT2')
    # GPT_path = new_dir
    # if not os.path.exists(new_dir):
    #     os.mkdir(new_dir)

    # mem_start_GPT = psutil.virtual_memory().used / (1024 ** 2)
    # stop_event = multiprocessing.Event()  # Create the stop event
    # process = multiprocessing.Process(target=memory_monitor, args=(new_dir, stop_event, mem_start_GPT))
    # process.start()
    # start_time = time.time()
    # seen, unseen, temp = create_gpt_embeddings.create_embeddings(file_path, device_list, new_dir, data_path, group_option, word_embedding_option, window_size, slide_length, vector_size)
    # gpt_time = time.time() - start_time
    # stop_event.set()  # Signal the memory monitor to stop
    # process.join()
    # plot_numbers_from_file(GPT_path)

    new_dir = os.path.join(save_dir, 'MAMBA')
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    seen, unseen = create_mamba_embeddings.create_embeddings(file_path, device_list, new_dir, data_path, group_option, word_embedding_option, window_size, slide_length, vector_size)

    temp = None

    if temp is not None:
        bert_embeddings_creation_time = bert_time
        bert_embeddings_creation_mem_usage = highest_value_without_outliers(os.path.join(bert_path, "memory_measurements.txt"))
    else:
        bert_embeddings_creation_time = 0
        bert_embeddings_creation_mem_usage = 0

    total = seen + unseen
    if total == 0:
        total = seen_ft + unseen_ft
        unseen = unseen_ft
        seen = seen_ft

    # Per flow!
    if total != 0:
        # times = (fast_text_training_time/unseen, fast_text_embeddings_creation_time/total, bert_embeddings_creation_time/total)
        # memories = (fast_text_training_mem_usage/unseen, fast_text_embeddings_creation_mem_usage/total, bert_embeddings_creation_mem_usage/total)
        times = (0, 0, bert_embeddings_creation_time/total)
        memories = (0, 0, bert_embeddings_creation_mem_usage/total)

    else:
        times = (0, 0, 0)
        memories = times

    return times, memories

def print_stats(stats_list, vector_list):
    print("-----------------------")

    # Define descriptions for each item in times and memories
    time_descriptions = ["FastText Training",
                         "FastText",
                         "BERT"]

    memory_descriptions = ["FastText Training",
                           "FastText",
                           "BERT"]

    # Printing the stats
    for vector, (times, memories) in zip(vector_list, stats_list):
        print(f"Stats for category: {vector}")

        # Print times with descriptions
        print("Times (sec):")
        for desc, item in zip(time_descriptions, times):
            print(f"{desc}: {item}")

        # Print memories with descriptions
        print("Memories (MB):")
        for desc, item in zip(memory_descriptions, memories):
            print(f"{desc}: {item}")

        print("-----------------------")

# if __name__ == "__main__":
def main_ext(vector_list, device_low, device_high, group_option, time_group, num2word_option, window_group, window_size, slide_length):
    # vector_list = [768, 512, 256, 128, 64, 32, 15, 5]
    # vector_list = [128, 256, 512, 768]
    # vector_list = [128, 256]
    stats_list = []

    time_descriptions = ["FastText Training",
                         "FastText",
                         "BERT"]

    memory_descriptions = ["FastText Training",
                           "FastText",
                           "BERT"]

    # # Analyzes devices device_low - device_high
    # device_high = 5
    # device_low = 0

    cwd = os.getcwd()

    # group_option     = 0

    # time_group       = 0
    # num2word_option  = 0   # Unlikely to be implemented

    # window_group     = 1
    # window_size      = 10
    # slide_length     = 1


    if group_option == 0:
        group_path = 'ungrouped'
        data_path = os.path.join(cwd, 'preprocessed_data', group_path)
 

    elif time_group != 0:
        group_path = 'grouped'
        time_path = str(time_group)
        data_path = os.path.join(cwd, 'preprocessed_data', group_path, time_path)


    elif window_group != 0:
        group_path = 'grouped'
        data_path = os.path.join(cwd, 'preprocessed_data', group_path, f"{window_size}_{slide_length}")

    if not os.path.exists(data_path):
        print(f"Error: The path {data_path} does not exit!")
        exit(1)

    print(f"Processing for input directory: {data_path}")

    for vector in vector_list:
        print(f"Creating embeddings at vector size: {vector}")
        
        save_dir = str(device_low) + "-" + str(device_high)
        save_dir = os.path.join(cwd, save_dir, str(vector))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        times, memories = main(device_low, device_high, save_dir, data_path, group_option, num2word_option, window_size, slide_length, vector)
        stats_list.append((times, memories))
    print(stats_list)
    print_stats(stats_list, vector_list)
    create_plots.plot_graphs_embedder(stats_list, vector_list, time_descriptions, memory_descriptions)
