#!/usr/bin/env python3
"""
Demo: generate a Mamba embedding for a single sentence
(minimal edits so mamba outputs match the first script)
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM

# ------------------------------------------------------------------
# Config – pick any pre-trained Mamba model you have locally.
#   • `mamba-130m-hf` → hidden_size = 768
#   • `mamba-370m-hf` → hidden_size = 1024
# ------------------------------------------------------------------
MODEL_NAME = "state-spaces/mamba-130m-hf"
# MODEL_NAME = "google-bert/bert-base-uncased"
MAX_TOKENS  = 512          # truncate / pad to this many tokens

def get_sentence_embedding(sentence: str, tokenizer, model, model_name: str = MODEL_NAME) -> torch.Tensor:
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
    )

    # move input tensors to the model device (important for consistency)
    device = next(model.parameters()).device
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        # models were loaded with config.output_hidden_states = True
        outputs = model(**inputs)

    # --- For Mamba (causal LM style) use mean-pooled last hidden states like the first script ---
    if "mamba" in model_name.lower() or model_name.startswith("state-spaces/mamba"):
        # outputs.hidden_states[-1] shape: (batch, seq_len, hidden)
        last_hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden)
        mask = inputs["attention_mask"][0].unsqueeze(-1)  # (seq_len, 1)
        summed = (last_hidden * mask).sum(dim=0)  # (hidden,)
        counts = mask.sum(dim=0)  # (1,)
        vec = (summed / counts)  # mean-pooled embedding (torch.Tensor)
    else:
        # Keep original behaviour for BERT-like models: take CLS token embedding
        cls_emb = outputs.hidden_states[-1][0][0]
        vec = cls_emb

    return vec  # torch.Tensor on CPU (or model.device)


def main() -> None:
    print(f"Loading model “{MODEL_NAME}” …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cfg        = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.output_hidden_states = True  # ensure hidden states are returned

    # Minimal branching: if Mamba, load same model class & dtype as in the first script
    if "mamba" in MODEL_NAME.lower() or MODEL_NAME.startswith("state-spaces/mamba"):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=cfg, torch_dtype=torch.float16)
    else:
        model = AutoModel.from_pretrained(MODEL_NAME, config=cfg)

    model.eval().to("cpu")  # keep on CPU like the first script

    sentence = input("\nEnter a sentence: ").strip()
    vec = get_sentence_embedding(sentence, tokenizer, model, MODEL_NAME)

    vec_np = vec.cpu().numpy()
    print("\nEmbedding (first 10 values):")
    print(vec_np[:10])
    print(f"\nVector dimensionality: {vec_np.shape[0]}")

if __name__ == "__main__":
    main()
