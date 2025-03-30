import os
from gensim.models import FastText
from gensim.utils import simple_preprocess
# import prepare_data
import numpy as np
import get_data

def train_fasttext_model(file_path, device_list, save_dir, data_path, group_option, word_embedding_option, window_size, slide_length, embedding_size=768):

    if group_option == 0:
        word_embed = "Ungrouped"
    else:
        word_embed = "Grouped"

    # Create the filename
    model_filename = "fasttext_model" + "_" + str(embedding_size)

    save_dir = os.path.join(save_dir, word_embed, f"{window_size}_{slide_length}")

    # Check if the model already exists
    if os.path.exists(os.path.join(save_dir, "model.model")):
        print(f'\033[92mModel already exists: {model_filename} ✔\033[0m')
        return os.path.join(save_dir, "model.model")
    else:
        os.makedirs(save_dir)
    
    dev1_seen = []
    dev1_seen_word = []

    for device in device_list:
        # device_file_path = os.path.join(file_path, device)
        # seen, _ = prepare_data.prepare_data(device_file_path)
        seen = get_data.get_seen_data(data_path, device)
        
        dev1_seen.extend(seen)

        if word_embedding_option == 1:
            # Tokenize the seen sentences for word embeddings
            seen_word = [simple_preprocess(sentence[0]) for sentence in seen]
            dev1_seen_word.extend(seen_word)

    print('Creating FastText model')
    if word_embedding_option == 1:
        model = FastText(sentences=dev1_seen_word, vector_size=embedding_size, window=5, min_count=1, workers=4)
    else:
        model = FastText(sentences=[sentence[0] for sentence in dev1_seen], vector_size=embedding_size, window=5, min_count=1, workers=4)
    print('\033[92mFastText model created ✔\033[0m')

    model.save(os.path.join(save_dir, "model.model"))
    print(f'\033[92mFastText model saved as {model_filename} ✔\033[0m')

    return os.path.join(save_dir, "model.model")

def create_device_embedding(model, file_path, device, model_dir, data_path, window_size, slide_length, vector_size=768):
    
    save_dir = os.path.dirname(model_dir)

    # Check if the embeddings text file already exists
    embeddings_folder = os.path.join(save_dir, "fast_text_embeddings")
    # Define filenames for seen and unseen embeddings
    seen_embeddings_filename = os.path.join(embeddings_folder, device + "_seen_fast_text_embeddings.txt")
    unseen_embeddings_filename = os.path.join(embeddings_folder, device + "_unseen_fast_text_embeddings.txt")
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    
    if os.path.exists(seen_embeddings_filename) and os.path.exists(unseen_embeddings_filename):
        print(f'\033[92mEmbeddings already exist for {device} ✔\033[0m')
        return 0,0
    
    # Initialize lists for storing seen and unseen embeddings
    seen_embeddings = []
    unseen_embeddings = []

    # Function to get the average embedding of a sentence
    def get_sentence_embedding(sentence, model, vector_size):
        embeddings = []
        for word in sentence:
            if word in model.wv:
                embeddings.append(model.wv[word])
            else:
                # Append a zero vector for words not in the vocabulary
                embeddings.append(np.zeros(vector_size))
        if embeddings:
            return sum(embeddings) / len(embeddings)
        else:
            return np.zeros(vector_size)

    # Prepare data for the device
    # device_file_path = os.path.join(file_path, device)
    # seen, unseen = prepare_data.prepare_data(device_file_path)
    seen, unseen = get_data.get_data(data_path, device)

    # Create embeddings for seen data
    for sentence in seen:
        embedding = get_sentence_embedding(sentence[0], model, vector_size)
        seen_embeddings.append(embedding)

    # Create embeddings for unseen data
    for sentence in unseen:
        embedding = get_sentence_embedding(sentence[0], model, vector_size)
        unseen_embeddings.append(embedding)

    # Save the embeddings to separate text files for seen and unseen data
    with open(seen_embeddings_filename, 'w') as f:
        for embedding in seen_embeddings:
            f.write(' '.join(map(str, embedding.tolist())) + '\n')

    with open(unseen_embeddings_filename, 'w') as f:
        for embedding in unseen_embeddings:
            f.write(' '.join(map(str, embedding.tolist())) + '\n')

    # Print the number of embeddings created
    print(f'Number of seen embeddings created: {len(seen_embeddings)}')
    print(f'Number of unseen embeddings created: {len(unseen_embeddings)}')
    return len(seen_embeddings), len(unseen_embeddings)

def create_embeddings(model_filename, file_path, device_list, data_path, window_size, slide_length, vector_size = 768):
    seen_count = 0
    unseen_count = 0

    # Load the trained FastText model
    model = FastText.load(model_filename)
    for device in device_list:
        seen, unseen = create_device_embedding(model, file_path, device, model_filename, data_path, window_size, slide_length, vector_size)
        seen_count += seen
        unseen_count += unseen

    return seen_count, unseen_count