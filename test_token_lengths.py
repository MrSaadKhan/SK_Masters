import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import logging

# ── Configuration ────────────────────────────────────────────────────────────────
sentence = (
    # "flow Start Milliseconds_0: 2019-06-25 08:36:28.378 flow End Milliseconds_0: 2019-06-25 09:06:19.433 flow Duration Milliseconds_0: 1791.055 reverse Flow Delta Milliseconds_0: 0.084 protocol Identifier_0: 6 source IPv4 Address_0: 3.86.45.125 source Transport Port_0: 443 packet Total Count_0: 41 octet Total Count_0: 5478 flow Attributes_0: 01 destination IPv4 Address_0: 192.168.0.1 destination Transport Port_0: 4162 reverse Packet Total Count_0: 41 reverse Octet Total Count_0: 1991 reverse Flow Attributes_0: 01 initial TCP Flags_0: AP union TCP Flags_0: AP reverse Initial TCP Flags_0: AP reverse Union TCP Flags_0: AP tcp Sequence Number_0: 0x7229139c reverse Tcp Sequence Number_0: 0x006a64d7 ingress Interface_0: 0 egress Interface_0: 0 vlan Id_0: 0x000 silkApp Label_0: 0 ip Class Of Service_0: 0x00 flow End Reason_0: active collector Name_0: C1 observation Domain Id_0: 0 tcp Urgent Total Count_0: 0 small Packet Count_0: 0 non Empty Packet Count_0: 38 data Byte Count_0: 3838 average Interarrival Time_0: 44774 first Non Empty Packet Size_0: 101 large Packet Count_0: 0 maximum Packet Size_0: 101 first Eight Non Empty Packet Directions_0: 50 standard Deviation Payload Length_0: 0 standard Deviation Interarrival Time_0: 7185 bytes Per Packet_0: 101 reverse Tcp Urgent Total Count_0: 0 reverse Small Packet Count_0: 0 reverse Non Empty Packet Count_0: 3 reverse Data Byte Count_0: 351 reverse Average Interarrival Time_0: 44774 reverse First Non Empty Packet Size_0: 117 reverse Large Packet Count_0: 0 reverse Maximum Packet Size_0: 117 reverse Standard Deviation Payload Length_0: 0 reverse Standard Deviation Interarrival Time_0: 7250 reverse Bytes Per Packet_0: 117 SEP_0: .  flow Start Milliseconds_1: 2019-06-25 09:07:08.474 flow End Milliseconds_1: 2019-06-25 09:36:25.400 flow Duration Milliseconds_1: 1756.926 reverse Flow Delta Milliseconds_1: 0.10200000000000001 protocol Identifier_1: 6 source IPv4 Address_1: 3.86.45.125 source Transport Port_1: 443 packet Total Count_1: 40 octet Total Count_1: 5337 flow Attributes_1: 01 destination IPv4 Address_1: 192.168.0.1 destination Transport Port_1: 4162 reverse Packet Total Count_1: 40 reverse Octet Total Count_1: 1951 reverse Flow Attributes_1: 01 initial TCP Flags_1: AP union TCP Flags_1: AP reverse Initial TCP Flags_1: AP reverse Union TCP Flags_1: AP tcp Sequence Number_1: 0x7229229a reverse Tcp Sequence Number_1: 0x006a6636 ingress Interface_1: 0 egress Interface_1: 0 vlan Id_1: 0x000 silkApp Label_1: 0 ip Class Of Service_1: 0x00 flow End Reason_1: active collector Name_1: C1 observation Domain Id_1: 0 tcp Urgent Total Count_1: 0 small Packet Count_1: 0 non Empty Packet Count_1: 37 data Byte Count_1: 3737 average Interarrival Time_1: 45047 first Non Empty Packet Size_1: 101 large Packet Count_1: 0 maximum Packet Size_1: 101 first Eight Non Empty Packet Directions_1: 45 standard Deviation Payload Length_1: 0 standard Deviation Interarrival Time_1: 3010 bytes Per Packet_1: 101 reverse Tcp Urgent Total Count_1: 0 reverse Small Packet Count_1: 0 reverse Non Empty Packet Count_1: 3 reverse Data Byte Count_1: 351 reverse Average Interarrival Time_1: 45046 reverse First Non Empty Packet Size_1: 117 reverse Large Packet Count_1: 0 reverse Maximum Packet Size_1: 117 reverse Standard Deviation Payload Length_1: 0 reverse Standard Deviation Interarrival Time_1: 3003 reverse Bytes Per Packet_1: 117 SEP_1: .  flow Start Milliseconds_2: 2019-06-25 09:37:11.391 flow End Milliseconds_2: 2019-06-25 10:06:51.310 flow Duration Milliseconds_2: 1779.919 reverse Flow Delta Milliseconds_2: 0.063 protocol Identifier_2: 6 source IPv4 Address_2: 3.86.45.125 source Transport Port_2: 443 packet Total Count_2: 39 octet Total Count_2: 5297 flow Attributes_2: 01 destination IPv4 Address_2: 192.168.0.1 destination Transport Port_2: 4162 reverse Packet Total Count_2: 39 reverse Octet Total Count_2: 1794 reverse Flow Attributes_2: 01 initial TCP Flags_2: AP union TCP Flags_2: AP reverse Initial TCP Flags_2: AP reverse Union TCP Flags_2: AP tcp Sequence Number_2: 0x72293133 reverse Tcp Sequence Number_2: 0x006a6795 ingress Interface_2: 0 egress Interface_2: 0 vlan Id_2: 0x000 silkApp Label_2: 0 ip Class Of Service_2: 0x00 flow End Reason_2: active collector Name_2: C1 observation Domain Id_2: 0 tcp Urgent Total Count_2: 0 small Packet Count_2: 0 non Empty Packet Count_2: 37 data Byte Count_2: 3737 average Interarrival Time_2: 46837 first Non Empty Packet Size_2: 101 large Packet Count_2: 0 maximum Packet Size_2: 101 first Eight Non Empty Packet Directions_2: c0 standard Deviation Payload Length_2: 0 standard Deviation Interarrival Time_2: 5839 bytes Per Packet_2: 101 reverse Tcp Urgent Total Count_2: 0 reverse Small Packet Count_2: 0 reverse Non Empty Packet Count_2: 2 reverse Data Byte Count_2: 234 reverse Average Interarrival Time_2: 46838 reverse First Non Empty Packet Size_2: 117 reverse Large Packet Count_2: 0 reverse Maximum Packet Size_2: 117 reverse Standard Deviation Payload Length_2: 0 reverse Standard Deviation Interarrival Time_2: 5872 reverse Bytes Per Packet_2: 117 SEP_2: .  flow Start Milliseconds_3: 2019-06-25 10:07:40.918 flow End Milliseconds_3: 2019-06-25 10:37:35.181 flow Duration Milliseconds_3: 1794.263 reverse Flow Delta Milliseconds_3: 0.043000000000000003 protocol Identifier_3: 6 source IPv4 Address_3: 3.86.45.125 source Transport Port_3: 443 packet Total Count_3: 39 octet Total Count_3: 5499 flow Attributes_3: 01 destination IPv4 Address_3: 192.168.0.1 destination Transport Port_3: 4162 reverse Packet Total Count_3: 39 reverse Octet Total Count_3: 1560 reverse Flow Attributes_3: 00 initial TCP Flags_3: AP union TCP Flags_3: AP reverse Initial TCP Flags_3: A reverse Union TCP Flags_3: A tcp Sequence Number_3: 0x72293fcc reverse Tcp Sequence Number_3: 0x006a687f ingress Interface_3: 0 egress Interface_3: 0 vlan Id_3: 0x000 silkApp Label_3: 0 ip Class Of Service_3: 0x00 flow End Reason_3: active collector Name_3: C1 observation Domain Id_3: 0 tcp Urgent Total Count_3: 0 small Packet Count_3: 0 non Empty Packet Count_3: 39 data Byte Count_3: 3939 average Interarrival Time_3: 47214 first Non Empty Packet Size_3: 101 large Packet Count_3: 0 maximum Packet Size_3: 101 first Eight Non Empty Packet Directions_3: 00 standard Deviation Payload Length_3: 0 standard Deviation Interarrival Time_3: 1622 bytes Per Packet_3: 101 reverse Tcp Urgent Total Count_3: 0 reverse Small Packet Count_3: 0 reverse Non Empty Packet Count_3: 0 reverse Data Byte Count_3: 0 reverse Average Interarrival Time_3: 47216 reverse First Non Empty Packet Size_3: 0 reverse Large Packet Count_3: 0 reverse Maximum Packet Size_3: 0 reverse Standard Deviation Payload Length_3: 0 reverse Standard Deviation Interarrival Time_3: 1626 SEP_3: .  flow Start Milliseconds_4: 2019-06-25 10:38:23.698 flow End Milliseconds_4: 2019-06-25 11:07:37.490 flow Duration Milliseconds_4: 1753.792 reverse Flow Delta Milliseconds_4: 0.057 protocol Identifier_4: 6 source IPv4 Address_4: 3.86.45.125 source Transport Port_4: 443 packet Total Count_4: 38 octet Total Count_4: 5358 flow Attributes_4: 01 destination IPv4 Address_4: 192.168.0.1 destination Transport Port_4: 4162 reverse Packet Total Count_4: 38 reverse Octet Total Count_4: 1520 reverse Flow Attributes_4: 00 initial TCP Flags_4: AP union TCP Flags_4: AP reverse Initial TCP Flags_4: A reverse Union TCP Flags_4: A tcp Sequence Number_4: 0x72294f2f reverse Tcp Sequence Number_4: 0x006a687f ingress Interface_4: 0 egress Interface_4: 0 vlan Id_4: 0x000 silkApp Label_4: 0 ip Class Of Service_4: 0x00 flow End Reason_4: active collector Name_4: C1 observation Domain Id_4: 0 tcp Urgent Total Count_4: 0 small Packet Count_4: 0 non Empty Packet Count_4: 38 data Byte Count_4: 3838 average Interarrival Time_4: 47396 first Non Empty Packet Size_4: 101 large Packet Count_4: 0 maximum Packet Size_4: 101 first Eight Non Empty Packet Directions_4: 00 standard Deviation Payload Length_4: 0 standard Deviation Interarrival Time_4: 1295 bytes Per Packet_4: 101 reverse Tcp Urgent Total Count_4: 0 reverse Small Packet Count_4: 0 reverse Non Empty Packet Count_4: 0 reverse Data Byte Count_4: 0 reverse Average Interarrival Time_4: 47398 reverse First Non Empty Packet Size_4: 0 reverse Large Packet Count_4: 0 reverse Maximum Packet Size_4: 0 reverse Standard Deviation Payload Length_4: 0 reverse Standard Deviation Interarrival Time_4: 1299 SEP_4: . "
    "flow Start Milliseconds: 2019-06-25 08:36:28.378 flow End Milliseconds: 2019-06-25 09:06:19.433 flow Duration Milliseconds: 1791.055 reverse Flow Delta Milliseconds: 0.084 protocol Identifier: 6 source IPv4 Address: 3.86.45.125 source Transport Port: 443 packet Total Count: 41 octet Total Count: 5478 flow Attributes: 01 destination IPv4 Address: 192.168.0.1 destination Transport Port: 4162 reverse Packet Total Count: 41 reverse Octet Total Count: 1991 reverse Flow Attributes: 01 initial TCP Flags: AP union TCP Flags: AP reverse Initial TCP Flags: AP reverse Union TCP Flags: AP tcp Sequence Number: 0x7229139c reverse Tcp Sequence Number: 0x006a64d7 ingress Interface: 0 egress Interface: 0 vlan Id: 0x000 silkApp Label: 0 ip Class Of Service: 0x00 flow End Reason: active collector Name: C1 observation Domain Id: 0 tcp Urgent Total Count: 0 small Packet Count: 0 non Empty Packet Count: 38 data Byte Count: 3838 average Interarrival Time: 44774 first Non Empty Packet Size: 101 large Packet Count: 0 maximum Packet Size: 101 first Eight Non Empty Packet Directions: 50 standard Deviation Payload Length: 0 standard Deviation Interarrival Time: 7185 bytes Per Packet: 101 reverse Tcp Urgent Total Count: 0 reverse Small Packet Count: 0 reverse Non Empty Packet Count: 3 reverse Data Byte Count: 351 reverse Average Interarrival Time: 44774 reverse First Non Empty Packet Size: 117 reverse Large Packet Count: 0 reverse Maximum Packet Size: 117 reverse Standard Deviation Payload Length: 0 reverse Standard Deviation Interarrival Time: 7250 reverse Bytes Per Packet: 117"
)
output_path = "mamba_embeddings_smaller.txt"
model_name = "state-spaces/mamba-130m-hf"


def _init_model(model_name: str):
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.output_hidden_states = True
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=cfg, torch_dtype=torch.float16
    ).to("cpu")
    model.eval()
    return tokenizer, model


def _pool_outputs(outputs, attention_mask):
    last_hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden)
    mask = attention_mask[0].unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=0)
    count = mask.sum(dim=0).clamp(min=1)
    return summed.div(count).detach().cpu().numpy()


def main():
    tokenizer, model = _init_model(model_name)

    raw_ids = tokenizer.encode(sentence, add_special_tokens=True)
    raw_len = len(raw_ids)

    # exact
    inputs_exact = tokenizer(sentence, return_tensors="pt", truncation=True,
                             padding="max_length", max_length=raw_len).to(model.device)
    emb_exact = _pool_outputs(model(**inputs_exact), inputs_exact['attention_mask'])

    # fixed
    inputs_fixed = tokenizer(sentence, return_tensors="pt", truncation=True,
                             padding="max_length", max_length=768).to(model.device)
    emb_fixed = _pool_outputs(model(**inputs_fixed), inputs_fixed['attention_mask'])

    # truncated: slice same inputs_fixed
    emb_trunc = None
    trunc_sentence = None
    if raw_len > 768:
        ids_trunc = inputs_fixed['input_ids'][:, :768]
        mask_trunc = inputs_fixed['attention_mask'][:, :768]
        # decode to get the actual truncated text
        trunc_sentence = tokenizer.decode(ids_trunc[0], skip_special_tokens=True)
        emb_trunc = _pool_outputs(
            model(input_ids=ids_trunc, attention_mask=mask_trunc),
            mask_trunc
        )

    # write results
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Sentence:\n{sentence}\n\n")
        f.write(f"## exact (max_length={raw_len})\n")
        f.write(" ".join(f"{x:.6f}" for x in emb_exact) + "\n\n")
        f.write("## fixed (max_length=768)\n")
        f.write(" ".join(f"{x:.6f}" for x in emb_fixed) + "\n\n")

        if emb_trunc is not None:
            f.write("## truncated (first 768 tokens)\n")
            f.write(f"# Truncated sentence used:\n{trunc_sentence}\n\n")
            f.write(" ".join(f"{x:.6f}" for x in emb_trunc) + "\n")

    print(f"✅ Embeddings saved to {output_path}")

if __name__ == "__main__":
    main()
