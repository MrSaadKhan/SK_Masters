import os
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import get_data
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

def fine_tune_model(model, tokenizer, combined_seen_data, device, vector_size, learning_rate, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer([sentence[0] for sentence in combined_seen_data], 
                       padding=True, truncation=True, 
                       return_tensors="pt", max_length=vector_size)
    dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f'Fine-tuning epoch {epoch + 1}/{epochs}'):
            input_ids, attention_mask = [t.to(device) for t in batch]
            
            # Masking the input for MLM objective; note: labels=input_ids for MLM.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")
    
    print(f"Unsupervised Fine-tuning complete for {device}")
    return model

def create_device_embedding(model, tokenizer, file_path, device, save_dir, data_dir, vector_size=768, fine_tuned=False):
    os.makedirs(save_dir, exist_ok=True)
    suffix = "_fine_tuned" if fine_tuned else ""
    seen_embeddings_filename = os.path.join(save_dir, device + f"_seen_bert_embeddings{suffix}.txt")
    unseen_embeddings_filename = os.path.join(save_dir, device + f"_unseen_bert_embeddings{suffix}.txt")
    
    if os.path.exists(seen_embeddings_filename) and os.path.exists(unseen_embeddings_filename):
        print(f'\033[92mEmbeddings already exist for {device} ✔\033[0m')
        return 0, 0
    
    def get_sentence_embedding(sentence, model, tokenizer, vector_size):
        # Pass output_hidden_states=True to get hidden states from the model.
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Extract the last layer's hidden states; take the [CLS] token embedding.
        return outputs.hidden_states[-1][0, 0, :].cpu().numpy()
    
    seen, unseen = get_data.get_data(data_dir, device)
    
    with open(seen_embeddings_filename, 'w') as f_seen, open(unseen_embeddings_filename, 'w') as f_unseen:
        for sentence in tqdm(seen, desc=f'Processing {device} (Seen)'):
            f_seen.write(' '.join(map(str, get_sentence_embedding(sentence[0], model, tokenizer, vector_size))) + '\n')
        for sentence in tqdm(unseen, desc=f'Processing {device} (Unseen)'):
            f_unseen.write(' '.join(map(str, get_sentence_embedding(sentence[0], model, tokenizer, vector_size))) + '\n')
    
    return len(seen), len(unseen)

def create_embeddings(file_path, device_list, save_dir, data_dir, group_option, word_embedding_option, window_size, slide_length, vector_size=768, fine_tune_percent=0.9):
    
    fine_tune_option = False
    
    def load_bert_model(model_name):
        # Load model for masked language modeling with output_hidden_states enabled.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        return tokenizer, model
    
    word_embed = "Grouped" if group_option else "Ungrouped"
    model_dir = os.path.join(save_dir, word_embed, f"{window_size}_{slide_length}")
    
    # Always create the main model directory and the base save_dir
    os.makedirs(model_dir, exist_ok=True)

    save_dir = os.path.join(model_dir, "bert_embeddings")
    os.makedirs(save_dir, exist_ok=True)

    # Only create the fine-tuned directory if fine_tune_option is True
    if fine_tune_option:
        fine_tuned_save_dir = os.path.join(model_dir, "bert_embeddings_finetuned")
        os.makedirs(fine_tuned_save_dir, exist_ok=True)


    # model_dict = {128: "prajjwal1/bert-tiny", 256: "prajjwal1/bert-mini", 512: "prajjwal1/bert-medium", 768: "bert-base-uncased"}

    model_dict = {
        128: "google/bert_uncased_L-2_H-128_A-2",   # tiny
        256: "google/bert_uncased_L-4_H-256_A-4",   # mini
        512: "google/bert_uncased_L-8_H-512_A-8",   # medium
        768: "google-bert/bert-base-uncased"        # base
    }

    if vector_size not in model_dict:
        print(f"Invalid vector_size. Choose from {list(model_dict.keys())}.")
        return 0, 0, None
    
    tokenizer, model = load_bert_model(model_dict[vector_size])
    # Clone the model for fine-tuning so that the pre-trained model remains unchanged.
    fine_tuned_tokenizer, fine_tuned_model = load_bert_model(model_dict[vector_size])
    fine_tuned_model.load_state_dict(model.state_dict())
    
    print("Model and tokenizer loaded successfully.")
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
    
    combined_seen_data = []
    for device in device_list:
        seen, _ = get_data.get_data(data_dir, device)
        combined_seen_data.extend(seen[:int(len(seen) * fine_tune_percent)])
    
    # fine_tuned_model = fine_tune_model(fine_tuned_model, tokenizer, combined_seen_data, "combined_data", vector_size, learning_rate=4e-5, epochs=3)
    
    seen_count = 0
    unseen_count = 0
    # Generate embeddings using the non-fine-tuned (pre-trained) model.
    for device in device_list:
        seen, unseen = create_device_embedding(model, tokenizer, file_path, device, save_dir, data_dir, vector_size, fine_tuned=False)
        seen_count += seen
        unseen_count += unseen
    # Generate embeddings using the fine-tuned model.
    # for device in device_list:
    #     seen, unseen = create_device_embedding(fine_tuned_model, tokenizer, file_path, device, fine_tuned_save_dir, data_dir, vector_size, fine_tuned=True)
    #     seen_count += seen
    #     unseen_count += unseen
    
    return seen_count, unseen_count, 0 if seen_count + unseen_count > 0 else None