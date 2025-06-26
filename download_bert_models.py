from transformers import AutoTokenizer, AutoModelForMaskedLM

for model_name in [
    "google/bert_uncased_L-2_H-128_A-2",
    "google/bert_uncased_L-4_H-256_A-4",
    "google/bert_uncased_L-8_H-512_A-8",
    "google-bert/bert-base-uncased",
]:
    print(f"Downloading and caching {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForMaskedLM.from_pretrained(model_name)
print("All models cached.")
