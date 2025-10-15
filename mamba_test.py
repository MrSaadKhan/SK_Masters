#!/usr/bin/env python3
"""
Demo: generate a Mamba embedding for a single sentence
"""
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ------------------------------------------------------------------
# Config – pick any pre-trained Mamba model you have locally.
#   • `mamba-130m-hf` → hidden_size = 768
#   • `mamba-370m-hf` → hidden_size = 1024
#   • `mamba-790m-hf` → hidden_size = ???
# ------------------------------------------------------------------
# MODEL_NAME = "state-spaces/mamba-130m-hf"
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_TOKENS  = 512          # truncate / pad to this many tokens

def get_sentence_embedding(sentence: str, tokenizer, model) -> torch.Tensor:
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # CLS token embedding (index 0)
    cls_emb = outputs.hidden_states[-1][0][0]
    return cls_emb


def main() -> None:
    print(f"Loading model “{MODEL_NAME}” …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cfg        = AutoConfig.from_pretrained(MODEL_NAME)
    model      = AutoModel.from_pretrained(MODEL_NAME, config=cfg)
    model.eval().to("cpu")                         # stays on CPU

    sentence = input("\nEnter a sentence: ").strip()
    vec = get_sentence_embedding(sentence, tokenizer, model)

    print("\nEmbedding (first 10 values):")
    print(vec[:10])
    print(f"\nVector dimensionality: {vec.shape[0]}")

if __name__ == "__main__":
    main()
