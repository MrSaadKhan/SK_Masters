import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import get_data
from tqdm import tqdm
from transformers.utils import logging

# Suppress unnecessary HF warnings
logging.set_verbosity_error()

# Globals for model and tokenizer
_global_model = None
_global_tokenizer = None
_global_max_length = None

def _init_model(model_name: str, max_length: int, hf_token: str = None):
    global _global_model, _global_tokenizer, _global_max_length
    _global_max_length = max_length
    tok_kwargs = {}
    if hf_token: tok_kwargs['use_auth_token'] = hf_token
    _global_tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.output_hidden_states = True
    mdl_kwargs = {'config': cfg, 'torch_dtype': torch.float16}
    if hf_token: mdl_kwargs['use_auth_token'] = hf_token
    _global_model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs).to('cpu')
    _global_model.eval()

def _embed_sentence(sentence: str) -> np.ndarray:
    inputs = _global_tokenizer(
        sentence,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=_global_max_length
    ).to(_global_model.device)
    with torch.no_grad():
        outputs = _global_model(**inputs)
    last_hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden)
    mask = inputs['attention_mask'][0].unsqueeze(-1)  # (seq_len, 1)
    summed = (last_hidden * mask).sum(dim=0)  # (hidden,)
    counts = mask.sum(dim=0)  # (1,)
    embedding = (summed / counts).cpu().numpy()  # (hidden,)
    return embedding

def create_device_mamba_embeddings(
    model_name: str,
    device: str,
    save_dir: str,
    data_dir: str,
    vector_size: int = 512,
    hf_token: str = None,
    max_total_embeddings: int = 0
) -> tuple[int, int]:
    if _global_model is None:
        _init_model(model_name, vector_size, hf_token)
    os.makedirs(save_dir, exist_ok=True)
    seen_file = os.path.join(save_dir, f"{device}_seen_mamba_embeddings.txt")
    unseen_file = os.path.join(save_dir, f"{device}_unseen_mamba_embeddings.txt")
    if os.path.exists(seen_file) and os.path.exists(unseen_file):
        return 0, 0
    seen, unseen = get_data.get_data(data_dir, device)
    seen_texts = [s[0] for s in seen]
    unseen_texts = [s[0] for s in unseen]

    if max_total_embeddings == 0:
        seen_limit, unseen_limit = len(seen_texts), len(unseen_texts)
    else:
        seen_target = int(max_total_embeddings * 0.7)
        unseen_target = max_total_embeddings - seen_target

        seen_limit = min(len(seen_texts), seen_target)
        unseen_limit = min(len(unseen_texts), unseen_target)

    seen_count = 0
    with open(seen_file, 'w') as sf:
        for sentence in tqdm(seen_texts[:seen_limit], desc=f"{device} (Seen)"):
            vec = _embed_sentence(sentence)
            sf.write(' '.join(map(str, vec)) + '\n')
            seen_count += 1
    unseen_count = 0
    with open(unseen_file, 'w') as uf:
        for sentence in tqdm(unseen_texts[:unseen_limit], desc=f"{device} (Unseen)"):
            vec = _embed_sentence(sentence)
            uf.write(' '.join(map(str, vec)) + '\n')
            unseen_count += 1
    return seen_count, unseen_count

def create_embeddings(
    file_path: str,
    device_list: list[str],
    save_dir: str,
    data_dir: str,
    group_option: int,
    word_embedding_option: int,
    window_size: int,
    slide_length: int,
    vector_size: int = 512,
    hf_token: str = None,
    max_total_embeddings: int = 860
) -> tuple[int, int]:
    model_name = "state-spaces/mamba-130m-hf"
    _init_model(model_name, vector_size, hf_token)
    grouping = "Grouped" if group_option else "Ungrouped"
    embeddings_dir = os.path.join(save_dir, grouping, f"{window_size}_{slide_length}", "mamba_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    total_seen = 0
    total_unseen = 0
    for device in device_list:
        s, u = create_device_mamba_embeddings(model_name, device, embeddings_dir, data_dir, vector_size, hf_token, max_total_embeddings)
        total_seen += s
        total_unseen += u
    return total_seen, total_unseen
