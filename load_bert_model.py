import os
import subprocess
import torch
from transformers import AutoConfig, BertModel, BertTokenizer

def convert_and_get_model(embedding_size):
    # Define model directories and expected embedding sizes
    model_info = {
        128: "BERT-Tiny (2-128)",
        256: "BERT-Mini (4-256)",
        512: "BERT-Medium (8-512)",
        768: "BERT-Base (12-768)"
    }

    # Check if the embedding size is valid
    if embedding_size not in model_info:
        print(f"No model found for embedding size {embedding_size}")
        return None, None

    # Define paths
    model_dir = rf"C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T2\ELEC 4952 - Thesis B\python\thesis_b\models\bert\{model_info[embedding_size]}"
    if not os.path.exists(model_dir):
        model_dir = rf"/home/iotresearch/saad/FastTextExp/thesis_b/models/bert/{model_info[embedding_size]}"

    tf_checkpoint_path = os.path.join(model_dir, "bert_model.ckpt")
    bert_config_file = os.path.join(model_dir, "bert_config.json")
    pytorch_dump_path = os.path.join(model_dir, "pytorch_model.bin")

    # Check if the PyTorch model already exists
    # if not os.path.exists(pytorch_dump_path):
    #     # Convert TensorFlow checkpoint to PyTorch
    #     subprocess.run([
    #         'python', 'convert_bert_original_tf_checkpoint_to_pytorch.py',
    #         '--tf_checkpoint_path', tf_checkpoint_path,
    #         '--bert_config_file', bert_config_file,
    #         '--pytorch_dump_path', pytorch_dump_path
    #     ], check=True)

    # Check if the PyTorch dump path exists
    if not os.path.exists(pytorch_dump_path):
        import convert_bert_original_tf_checkpoint_to_pytorch

        # Convert TensorFlow checkpoint to PyTorch
        convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
            tf_checkpoint_path=tf_checkpoint_path,
            bert_config_file=bert_config_file,
            pytorch_dump_path=pytorch_dump_path
        )
    else:
        print(f"PyTorch dump path '{pytorch_dump_path}' already exists. Skipping conversion.")

    # Load the model configuration
    config = AutoConfig.from_pretrained(bert_config_file)

    # Print the embedding size
    print("Embedding size:", config.hidden_size)

    if config.hidden_size != embedding_size:
        print("Converted model embedding size does not match the expected size.")
        return None, None

    try:
        model = BertModel(config)
        state_dict = torch.load(pytorch_dump_path)
        state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        return None, None

# # Example usage
# embedding_size = 256
# model, tokenizer = convert_and_get_model(embedding_size)

# if model and tokenizer:
#     print("Model and tokenizer loaded successfully.")
# else:
#     print("Failed to load model and tokenizer.")
