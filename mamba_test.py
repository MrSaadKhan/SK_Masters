#!/usr/bin/env python3
"""
Demo: generate a Mamba embedding for a single sentence
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ------------------------------------------------------------------
# Config – pick any pre-trained Mamba model you have locally.
#   • `mamba-130m-hf` → hidden_size = 768
#   • `mamba-370m-hf` → hidden_size = 1024
#   • `mamba-790m-hf` → hidden_size = ???
# ------------------------------------------------------------------
MODEL_NAME = "state-spaces/mamba-130m-hf"
MAX_TOKENS  = 2000          # truncate / pad to this many tokens

def get_sentence_embedding(sentence: str, tokenizer, model) -> torch.Tensor:
    """Average-pool the last hidden state over the *non-padding* tokens."""
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
    )

    with torch.no_grad():
        # `output_hidden_states=True` can be set here or in the config
        outputs = model(**inputs, output_hidden_states=True)

    last = outputs.hidden_states[-1][0]            # (seq_len, hidden)
    mask = inputs["attention_mask"][0].unsqueeze(-1)  # (seq_len, 1)

    summed  = (last * mask).sum(dim=0)             # (hidden,)
    count   = mask.sum()                           # scalar
    return summed / count                          # (hidden,)

def main() -> None:
    print(f"Loading model “{MODEL_NAME}” …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cfg        = AutoConfig.from_pretrained(MODEL_NAME)
    model      = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=cfg)
    model.eval().to("cpu")                         # stays on CPU

    sentence = input("\nEnter a sentence: ").strip()
    vec = get_sentence_embedding(sentence, tokenizer, model)

    print("\nEmbedding (first 10 values):")
    print(vec[:10])
    print(f"\nVector dimensionality: {vec.shape[0]}")

if __name__ == "__main__":
    main()
