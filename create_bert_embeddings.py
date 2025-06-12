import os
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import get_data
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")
    
    print(f"Unsupervised Fine-tuning complete for {device}")
    return model

# Global model/tokenizer for worker processes
global_model = None
global_tokenizer = None
def _init_worker(model_name, vector_size):
    global global_model, global_tokenizer
    global_tokenizer = AutoTokenizer.from_pretrained(model_name)
    global_model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    global_model.eval()

def _embed_sentence(sentence):
    inputs = global_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512).to(global_model.device)
    with torch.no_grad():
        outputs = global_model(**inputs)
    return outputs.hidden_states[-1][0, 0, :].cpu().numpy()

# def _embed_sentence(sentence):
#     inputs = global_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512).to(global_model.device)
#     with torch.no_grad():
#         outputs = global_model(**inputs)
#     # Average the last 3 hidden states instead of using just the last one
#     last_3_hidden_states = outputs.hidden_states[-3:]
#     avg_hidden_state = torch.stack(last_3_hidden_states).mean(dim=0)
#     return avg_hidden_state[0, 0, :].cpu().numpy()

def create_device_embedding(model, tokenizer, file_path, device, save_dir, data_dir, vector_size=768, fine_tuned=False):
    os.makedirs(save_dir, exist_ok=True)
    suffix = "_fine_tuned" if fine_tuned else ""
    seen_embeddings_filename = os.path.join(save_dir, device + f"_seen_bert_embeddings{suffix}.txt")
    unseen_embeddings_filename = os.path.join(save_dir, device + f"_unseen_bert_embeddings{suffix}.txt")
    
    if os.path.exists(seen_embeddings_filename) and os.path.exists(unseen_embeddings_filename):
        print(f'\033[92mEmbeddings already exist for {device} ✔\033[0m')
        return 0, 0
    
    seen, unseen = get_data.get_data(data_dir, device)
    seen_texts   = [s[0] for s in seen]
    unseen_texts = [s[0] for s in unseen]

    model_name = model.config._name_or_path
    # Use Pool with initializer to load model/tokenizer once per process
    with open(seen_embeddings_filename, 'w') as f_seen:
        with Pool(processes=cpu_count(), initializer=_init_worker, initargs=(model_name, vector_size)) as pool:
            for vec in tqdm(pool.imap(_embed_sentence, seen_texts), total=len(seen_texts), desc=f'Processing {device} (Seen)'):
                f_seen.write(' '.join(map(str, vec)) + '\n')

    with open(unseen_embeddings_filename, 'w') as f_unseen:
        with Pool(processes=cpu_count(), initializer=_init_worker, initargs=(model_name, vector_size)) as pool:
            for vec in tqdm(pool.imap(_embed_sentence, unseen_texts), total=len(unseen_texts), desc=f'Processing {device} (Unseen)'):
                f_unseen.write(' '.join(map(str, vec)) + '\n')
    
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


    model_dict = {128: "prajjwal1/bert-tiny", 256: "prajjwal1/bert-mini", 512: "prajjwal1/bert-medium", 768: "bert-base-uncased"}
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
    for dev in device_list:
        s_cnt, u_cnt = create_device_embedding(model, tokenizer, file_path, dev, save_dir, data_dir, vector_size, fine_tuned=False)
        seen_count += s_cnt
        unseen_count += u_cnt
    
    return seen_count, unseen_count, 0 if seen_count + unseen_count > 0 else None
