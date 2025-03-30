import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, TFBertModel, AutoModel, AutoTokenizer
# import prepare_data
import get_data
from tqdm import tqdm


def create_device_embedding(model, tokenizer, file_path, device, save_dir, data_dir, vector_size=768):
    # embeddings_folder = "bert_embeddings" + "_" + str(vector_size)
    # embeddings_folder = os.path.join(save_dir, embeddings_folder)
    embeddings_folder = save_dir
    seen_embeddings_filename = os.path.join(embeddings_folder, device + "_seen_bert_embeddings.txt")
    unseen_embeddings_filename = os.path.join(embeddings_folder, device + "_unseen_bert_embeddings.txt")
    
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    
    if os.path.exists(seen_embeddings_filename) and os.path.exists(unseen_embeddings_filename):
        print(f'\033[92mEmbeddings already exist for {device} ✔\033[0m')
        print(f'Paths checked: {seen_embeddings_filename} and {unseen_embeddings_filename}')
        return 0, 0
    
    # seen_embeddings = []
    # unseen_embeddings = []

    def get_sentence_embedding(sentence, model, tokenizer, vector_size):
        sentence_str = ' '.join(sentence)
        inputs = tokenizer(sentence_str, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
        return cls_embedding

    # device_file_path = os.path.join(file_path, device)
    # seen, unseen = prepare_data.prepare_data(device_file_path)
    seen, unseen = get_data.get_data(data_dir, device)

    total_sentences = len(seen) + len(unseen)

    with open(seen_embeddings_filename, 'w') as f_seen, tqdm(total=total_sentences, desc=f'Processing {device}', unit='sentence') as pbar:
        for sentence in seen:
            embedding = get_sentence_embedding(sentence[0], model, tokenizer, vector_size)
            f_seen.write(' '.join(map(str, embedding.tolist())) + '\n')
            pbar.update(1)

    with open(unseen_embeddings_filename, 'w') as f_unseen, tqdm(total=total_sentences, desc=f'Processing {device}', unit='sentence', initial=len(seen)) as pbar:
        for sentence in unseen:
            embedding = get_sentence_embedding(sentence[0], model, tokenizer, vector_size)
            f_unseen.write(' '.join(map(str, embedding.tolist())) + '\n')
            pbar.update(1)
    print(f'Number of seen embeddings created: {len(seen)}')
    print(f'Number of unseen embeddings created: {len(unseen)}')
    return len(seen), len(unseen)

def create_embeddings(file_path, device_list, save_dir, data_dir, group_option, word_embedding_option, window_size, slide_length, vector_size = 768):
    def load_bert_model(model_name):
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        return tokenizer, model


    if group_option == 0:
        word_embed = "Ungrouped"
    else:
        word_embed = "Grouped"
    
    model_dir = os.path.join(save_dir, word_embed, f"{window_size}_{slide_length}")
    save_dir = os.path.join(model_dir, "bert_embeddings")
    os.makedirs(save_dir)
    # List of pretrained models of [128, 256, 512, 768]
    model_list = ["prajjwal1/bert-tiny", "prajjwal1/bert-mini", "prajjwal1/bert-medium", "bert-base-uncased"]
    model_lengths = [128, 256, 512, 768]
    # Create a dictionary to map vector_size to model names
    model_dict = dict(zip(model_lengths, model_list))

     # Check if the provided vector_size is valid
    if vector_size not in model_dict:
        print(f"Invalid vector_size. Please choose from {model_lengths}.")
        return 0, 0, None

    # Get the model name based on vector_size
    model_name = model_dict[vector_size]

    tokenizer, model = load_bert_model(model_name)

    if model and tokenizer:
        print("Model and tokenizer loaded successfully.")
        flag = 0
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

    else:
        print("Failed to load model and tokenizer.")
        flag = None
        return 0, 0, None

    seen_count = 0
    unseen_count = 0

    for device in device_list:
        seen, unseen = create_device_embedding(model, tokenizer, file_path, device, save_dir, data_dir, vector_size)
        seen_count += seen
        unseen_count += unseen

    if seen_count + unseen_count == 0:
        flag = None
    else:
        flag = 0

    return seen_count, unseen_count, flag