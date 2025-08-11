import os
import shutil

def concatenate_embeddings(original_dir, seen_first=True):
    """
    Merges seen/unseen .txt files in all subfolders of original_dir,
    ignoring any blank lines at the end of each input file.
    
    Args:
      original_dir (str): Name of the directory to process.
      seen_first (bool): 
        - True  → write seen then unseen.
        - False → write unseen then seen.
    """
    new_dir = f"{original_dir}_merged"

    print(f"Original directory: {original_dir}")
    print(f"New (merged) directory: {new_dir}")

    if not os.path.exists(original_dir):
        print(f"❌ Error: Directory '{original_dir}' does not exist.")
        return

    if os.path.exists(new_dir):
        print(f"⚠️ Warning: '{new_dir}' already exists. Deleting and recreating.")
        shutil.rmtree(new_dir)

    print("📁 Copying directory...")
    shutil.copytree(original_dir, new_dir)

    print("🔍 Recursively searching for .txt files to merge...")
    for root, _, files in os.walk(new_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        if not txt_files:
            continue

        device_file_map = {}
        for fname in txt_files:
            # base-style: device.json_seen.txt / device.json_unseen.txt
            if fname.endswith("_seen.txt"):
                device = fname[:-9]
                key = (device, "base")
                device_file_map.setdefault(key, {})["seen"] = fname
            elif fname.endswith("_unseen.txt"):
                device = fname[:-11]
                key = (device, "base")
                device_file_map.setdefault(key, {})["unseen"] = fname
            else:
                # extended: device_seen_bert_embeddings.txt, etc.
                parts = fname.split('_')
                if 'seen' in parts:
                    idx = parts.index('seen')
                    su = 'seen'
                elif 'unseen' in parts:
                    idx = parts.index('unseen')
                    su = 'unseen'
                else:
                    continue

                emb = parts[idx+1] if idx+1 < len(parts) and parts[idx+2]=="embeddings.txt" else "base"
                device = '_'.join(parts[:idx])
                key = (device, emb)
                device_file_map.setdefault(key, {})[su] = fname

        for (device, emb), pair in device_file_map.items():
            seen_file  = pair.get("seen")
            unseen_file = pair.get("unseen")
            if not (seen_file and unseen_file):
                print(f"⛔ Skipping {device} ({emb}): missing seen/unseen")
                continue

            seen_path   = os.path.join(root, seen_file)
            unseen_path = os.path.join(root, unseen_file)

            # determine output name
            if emb == "base":
                out_name = f"{device}.txt"
            else:
                out_name = f"{device}_{emb}_embeddings.txt"
            out_path = os.path.join(root, out_name)

            print(f"\n🔧 Merging in {root}:")
            order = ("seen", "unseen") if seen_first else ("unseen", "seen")
            print(f"    {order[0]} → {order[1]} → {out_name}")

            # read & trim trailing blank lines
            def read_and_trim(path):
                lines = open(path, 'r', encoding='utf-8').read().splitlines()
                # pop empty strings off end
                while lines and lines[-1].strip() == "":
                    lines.pop()
                return lines

            seen_lines  = read_and_trim(seen_path)
            unseen_lines = read_and_trim(unseen_path)

            with open(out_path, 'w', encoding='utf-8') as fo:
                if seen_first:
                    fo.write("\n".join(seen_lines))
                    fo.write("\n")
                    fo.write("\n".join(unseen_lines))
                else:
                    fo.write("\n".join(unseen_lines))
                    fo.write("\n")
                    fo.write("\n".join(seen_lines))

            print(f"  🧹 Deleting source files:")
            print(f"     - {seen_file}")
            print(f"     - {unseen_file}")
            os.remove(seen_path)
            os.remove(unseen_path)

    print("\n✅ Complete :)")
