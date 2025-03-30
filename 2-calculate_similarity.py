from result_builder import build_results
from tqdm import tqdm
import numpy as np
import itertools
import random
import os

def generate_device_combinations(device_list):
    combinations = list(itertools.combinations(device_list, 2))
    combinations_self = [(device, device) for device in device_list]
    total_combinations = combinations + combinations_self
    total_combinations.sort(key=lambda combo: (device_list.index(combo[0]), device_list.index(combo[1])))
    return total_combinations

def count_lines(file_path, folder_name, device, seen_or_unseen):
    # file = os.path.join(file_path, folder_name, device + "_" + seen_or_unseen + "_" + folder_name + ".txt")
    file = file_path
    # print(file)
    with open(file, 'r') as fp:
        lines = sum(1 for line in fp)
        return lines

def read_specific_line(file_path, line_number):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for current_line_number, line in enumerate(file, start=1):
                if current_line_number == line_number:
                    vector = line.strip().split()  # Split the line into a list of values
                    vector = [float(value) for value in vector]  # Convert values to float
                    return np.array(vector)  # Convert to numpy array and return
        print(f"Line {line_number} does not exist in the file.")
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def calculate_similarity(dev1, dev2, dev1_path, dev2_path):
    length_dev_1 = count_lines(dev1_path, "fast_text_embeddings", dev1, "seen")
    length_dev_2 = count_lines(dev2_path, "fast_text_embeddings", dev2, "seen")

    index_1 = random.randint(1, length_dev_1)
    index_2 = random.randint(1, length_dev_2)

    vector_1 = read_specific_line(dev1_path, index_1)
    vector_2 = read_specific_line(dev2_path, index_2)

    similarity = cosine_similarity(vector_1, vector_2)

    return similarity

# def main():
#     iterations = 10000
#     file_path = r'/home/iotresearch/saad/FastTextExp/thesis_b' 
#     if not os.path.exists(file_path):
#         file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T2\ELEC 4952 - Thesis B\python\thesis_b'

#     device_list = ['nature_remo.json', 'xiaomi_mijia_led.json', 'irobot_roomba.json', 'planex_smacam_pantilt.json', 'jvc_kenwood_hdtv_ip_camera.json']
#     total_combinations = generate_device_combinations(device_list)
#     embedder = ["fast_text_embeddings", "bert_embeddings"]
#     seen_option = ["seen", "unseen"]

#     device_indices = {dev: i for i, dev in enumerate(device_list)}
#     print(device_indices)

#     for embed_option in embedder:

#         for dev1, dev2 in total_combinations:

#             for seen_data_option in seen_option:

#                 similarity_score_list = []
#                 mu_device_seen = sigma_device_seen = mu_device_unseen = sigma_device_unseen = 0

#                 for _ in range(iterations):
#                     file1 = os.path.join(file_path, embed_option, dev1 + "_" + seen_data_option + "_" + embed_option + ".txt")
#                     file2 = os.path.join(file_path, embed_option, dev1 + "_" + seen_data_option + "_" + embed_option + ".txt")
#                     similarity_score = calculate_similarity(dev1, dev2, file1, file2)
#                     similarity_score_list.append(similarity_score)

#                 if seen_data_option == "seen":
#                     mu_device_seen = np.mean(similarity_score_list)
#                     sigma_device_seen = np.std(similarity_score_list)

#                 if seen_data_option == "unseen":
#                     mu_device_unseen = np.mean(similarity_score_list)
#                     sigma_device_unseen = np.std(similarity_score_list)

#                 build_results(device_indices, dev1, dev2, mu_device_seen, sigma_device_seen, mu_device_unseen, sigma_device_unseen, embed_option)

# if __name__ == "__main__":
#     main()


def main():
    iterations = 10000
    file_path = r'/home/iotresearch/saad/FastTextExp/thesis_b' 
    if not os.path.exists(file_path):
        file_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T2\ELEC 4952 - Thesis B\python\thesis_b'

    device_list = ['nature_remo.json', 'xiaomi_mijia_led.json', 'irobot_roomba.json', 'planex_smacam_pantilt.json', 'jvc_kenwood_hdtv_ip_camera.json']
    total_combinations = generate_device_combinations(device_list)
    embedder = ["fast_text_embeddings", "bert_embeddings"]
    seen_option = ["seen", "unseen"]

    device_indices = {dev: i for i, dev in enumerate(device_list)}
    print(device_indices)

    for embed_option in tqdm(embedder, desc="Embedding Options"):  # Wrap outer loop with tqdm

        for dev1, dev2 in tqdm(total_combinations, desc="Device Combinations", leave=False):  # Wrap middle loop with tqdm
            
            mu_device_seen = sigma_device_seen = mu_device_unseen = sigma_device_unseen = 0

            for seen_data_option in tqdm(seen_option, desc="Seen/Unseen Options", leave=False):  # Wrap inner loop with tqdm

                similarity_score_list = []

                for _ in tqdm(range(iterations), desc="Iterations", leave=False):  # Wrap innermost loop with tqdm
                    file1 = os.path.join(file_path, embed_option, dev1 + "_" + seen_data_option + "_" + embed_option + ".txt")
                    file2 = os.path.join(file_path, embed_option, dev2 + "_" + seen_data_option + "_" + embed_option + ".txt")
                    similarity_score = calculate_similarity(dev1, dev2, file1, file2)
                    similarity_score_list.append(similarity_score)

                if seen_data_option == "seen":
                    mu_device_seen = np.mean(similarity_score_list)
                    sigma_device_seen = np.std(similarity_score_list)

                if seen_data_option == "unseen":
                    mu_device_unseen = np.mean(similarity_score_list)
                    sigma_device_unseen = np.std(similarity_score_list)

            build_results(device_indices, dev1, dev2, mu_device_seen, sigma_device_seen, mu_device_unseen, sigma_device_unseen, embed_option)

if __name__ == "__main__":
    main()