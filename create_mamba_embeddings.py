import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import get_data
from tqdm import tqdm

# Globals for model and tokenizer
_global_model = None
_global_tokenizer = None
_global_max_length = None

def _init_model(model_name: str, max_length: int, hf_token: str = None):
    """
    Initialize global model, tokenizer, and max_length.
    """
    global _global_model, _global_tokenizer, _global_max_length
    print(f"[INIT] Loading model and tokenizer: {model_name} with max_length={max_length}")
    _global_max_length = max_length
    tokenizer_kwargs = {}
    if hf_token:
        tokenizer_kwargs['use_auth_token'] = hf_token
    _global_tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model_kwargs = {'output_hidden_states': True, 'torch_dtype': torch.float16}
    if hf_token:
        model_kwargs['use_auth_token'] = hf_token
    _global_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    _global_model.eval()
    print("[INIT] Model and tokenizer ready.")


def _embed_sentence(sentence: str) -> np.ndarray:
    """
    Tokenize and get the last hidden state for the BOS token of the sentence.
    Uses globals initialized in _init_model.
    """
    print(f"[EMBED] Tokenizing sentence of length {len(sentence)}")
    inputs = _global_tokenizer(
        sentence,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=_global_max_length
    ).to(_global_model.device)
    with torch.no_grad():
        outputs = _global_model(**inputs)
    last_hidden = outputs.hidden_states[-1][0, 0, :].cpu().numpy()
    print("[EMBED] Obtained embedding vector.")
    return last_hidden


def create_device_mamba_embeddings(
    model_name: str,
    device: str,
    save_dir: str,
    data_dir: str,
    vector_size: int = 512,
    hf_token: str = None
) -> tuple[int, int]:
    """
    Create seen and unseen embeddings for a single device using Mamba, sequentially (no multiprocessing).
    """
    print(f"[DEVICE] Starting embeddings for device: {device}")
    os.makedirs(save_dir, exist_ok=True)
    seen_file = os.path.join(save_dir, f"{device}_seen_mamba_embeddings.txt")
    unseen_file = os.path.join(save_dir, f"{device}_unseen_mamba_embeddings.txt")

    # Skip if already exists
    if os.path.exists(seen_file) and os.path.exists(unseen_file):
        print(f"[DEVICE] Skipping {device}, embeddings already exist.")
        return 0, 0

    # Load data
    print(f"[DEVICE] Loading data for {device} from {data_dir}")
    seen, unseen = get_data.get_data(data_dir, device)
    seen_texts = [s[0] for s in seen]
    unseen_texts = [s[0] for s in unseen]
    print(f"[DEVICE] Loaded {len(seen_texts)} seen and {len(unseen_texts)} unseen texts.")

    # Initialize model and tokenizer once
    _init_model(model_name, vector_size, hf_token)

    # Process seen embeddings sequentially
    seen_count = 0
    print(f"[DEVICE] Processing seen embeddings for {device}")
    with open(seen_file, 'w') as sf:
        for idx, sentence in enumerate(tqdm(seen_texts, desc=f"{device} (Seen)")):
            print(f"[DEVICE][SEEN] Embedding {idx+1}/{len(seen_texts)}")
            vec = _embed_sentence(sentence)
            sf.write(' '.join(map(str, vec)) + '\n')
            seen_count += 1

    # Process unseen embeddings sequentially
    unseen_count = 0
    print(f"[DEVICE] Processing unseen embeddings for {device}")
    with open(unseen_file, 'w') as uf:
        for idx, sentence in enumerate(tqdm(unseen_texts, desc=f"{device} (Unseen)")):
            print(f"[DEVICE][UNSEEN] Embedding {idx+1}/{len(unseen_texts)}")
            vec = _embed_sentence(sentence)
            uf.write(' '.join(map(str, vec)) + '\n')
            unseen_count += 1

    print(f"[DEVICE] Completed embeddings for {device}: {seen_count} seen, {unseen_count} unseen")
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
    hf_token: str = None
) -> tuple[int, int]:
    """
    Generate Mamba embeddings for all devices in device_list sequentially.

    Returns:
        seen_count (int), unseen_count (int)
    """
    print(f"[RUN] Starting full embedding run for {len(device_list)} devices")
    model_name = "state-spaces/mamba-2.8b-hf"
    grouping = "Grouped" if group_option else "Ungrouped"
    model_dir = os.path.join(save_dir, grouping, f"{window_size}_{slide_length}")
    embeddings_dir = os.path.join(model_dir, "mamba_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    total_seen = 0
    total_unseen = 0
    for device in device_list:
        seen_cnt, unseen_cnt = create_device_mamba_embeddings(
            model_name,
            device,
            embeddings_dir,
            data_dir,
            vector_size,
            hf_token
        )
        total_seen += seen_cnt
        total_unseen += unseen_cnt

    print(f"[RUN] Completed all devices: Total seen={total_seen}, Total unseen={total_unseen}")
    return total_seen, total_unseen