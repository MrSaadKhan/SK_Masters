import os
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
import get_data
from tqdm import tqdm

def create_device_embedding(model, tokenizer, file_path, device, save_dir, data_dir, vector_size=768):
    embeddings_folder = save_dir
    seen_embeddings_filename = os.path.join(embeddings_folder, device + "_seen_gpt2_embeddings.txt")
    unseen_embeddings_filename = os.path.join(embeddings_folder, device + "_unseen_gpt2_embeddings.txt")
    
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    
    if os.path.exists(seen_embeddings_filename) and os.path.exists(unseen_embeddings_filename):
        print(f'\033[92mEmbeddings already exist for {device} ✔\033[0m')
        print(f'Paths checked: {seen_embeddings_filename} and {unseen_embeddings_filename}')
        return 0, 0

    def get_sentence_embedding(sentence, model, tokenizer, vector_size):
        sentence_str = ' '.join(sentence)
        inputs = tokenizer(sentence_str, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Average the hidden states across tokens to get sentence embedding
        hidden_states = outputs.last_hidden_state[0]  # Shape: (sequence_length, hidden_size)
        sentence_embedding = hidden_states.mean(dim=0).numpy()  # Average across the sequence length (tokens)
        return sentence_embedding

    seen, unseen = get_data.get_data(data_dir, device)

    total_sentences = len(seen) + len(unseen)

    with open(seen_embeddings_filename, 'w') as f_seen, tqdm(total=total_sentences, desc=f'Processing {device}', unit='sentence') as pbar:
        for sentence in seen:
            # print(sentence)
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

def create_embeddings(file_path, device_list, save_dir, data_dir, group_option, word_embedding_option, window_size, slide_length, vector_size=768):
    def load_gpt2_model(model_name):
        # Load GPT-2 tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        tokenizer.add_special_tokens({'pad_token': '[PAD]'})


        model = GPT2Model.from_pretrained(model_name)
        return tokenizer, model

    if group_option == 0:
        word_embed = "Ungrouped"
    else:
        word_embed = "Grouped"
    
    model_dir = os.path.join(save_dir, word_embed, f"{window_size}_{slide_length}")
    save_dir = os.path.join(model_dir, "gpt2_embeddings")
    os.makedirs(save_dir)
    
    # GPT-2 models typically use a hidden size of 768
    model_name = "gpt2"  # You can adjust to other sizes like "gpt2-medium", etc.
    
    tokenizer, model = load_gpt2_model(model_name)

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
