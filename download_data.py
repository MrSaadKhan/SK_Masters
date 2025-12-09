#!/usr/bin/env python3
"""
download_data_extract_json.py

- Clones (or downloads) KDDI-IoT-2019 repo
- Joins split parts (supports both `.gz.001` and `.gz00` style)
- Decompresses .gz -> .tar (streamed) into ./data/tars/
- Extracts members from each .tar -> writes member files into ./data/json/
  (streamed, chunked; no full-member in-memory reads)
- Verbose logging so you can see progress.

Run:
    python3.10 download_data_extract_json.py
"""
from pathlib import Path
import subprocess, shutil, gzip, tarfile, tempfile, urllib.request, zipfile, os, re, sys, time
from math import ceil

REPO_URL = "https://github.com/nokuik/KDDI-IoT-2019.git"
ZIP_URL = "https://github.com/nokuik/KDDI-IoT-2019/archive/refs/heads/master.zip"
REPO_DIRNAME = "KDDI-IoT-2019"
DATA_DIR = Path.cwd() / "data"
TARS_DIR = DATA_DIR / "tars"
JSON_DIR = DATA_DIR / "json"

CHUNK = 1024 * 1024  # 1 MiB chunk for streaming copy

def v(msg=""):
    print(msg, flush=True)

def try_git_clone(dest: Path) -> bool:
    try:
        v(f"[CLONE] checking git ...")
        subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        v(f"[CLONE] git clone --depth 1 {REPO_URL} -> {dest}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(dest)], check=True)
        return True
    except Exception as e:
        v(f"[CLONE] git clone failed or not installed: {e}")
        return False

def download_and_unzip(dest_parent: Path) -> Path:
    v("[ZIP] downloading repo zip...")
    tmpfd, tmpzip = tempfile.mkstemp(suffix=".zip")
    os.close(tmpfd)
    try:
        urllib.request.urlretrieve(ZIP_URL, tmpzip)
        v("[ZIP] extracting zip...")
        with zipfile.ZipFile(tmpzip, "r") as zf:
            zf.extractall(dest_parent)
        for p in dest_parent.iterdir():
            if p.is_dir() and p.name.lower().startswith("kddi-iot-2019"):
                final = dest_parent / REPO_DIRNAME
                if final.exists():
                    shutil.rmtree(final)
                p.rename(final)
                v(f"[ZIP] repo extracted -> {final}")
                return final
        raise FileNotFoundError("Couldn't find extracted repo directory after unzip")
    finally:
        try:
            os.remove(tmpzip)
        except Exception:
            pass

def find_ipfix(repo_root: Path) -> Path:
    v("[SEARCH] looking for ipfix/ directory...")
    for p in repo_root.rglob("ipfix"):
        if p.is_dir():
            v(f"[SEARCH] found ipfix at: {p}")
            return p
    raise FileNotFoundError("ipfix directory not found")

# Enhanced joining: match .gz.* (like .gz.001) AND .gzNN (like .gz00 .gz01) patterns
def join_split_parts(ipfix_dir: Path):
    v("[JOIN] scanning for split parts ('.gz.*' and '.gzNN' styles)...")
    all_files = list(ipfix_dir.iterdir())
    # 1) handle .gz.<suffix> (dot before suffix)
    dot_parts = sorted(ipfix_dir.glob("*.gz.*"))
    grouped = {}
    for p in dot_parts:
        idx = p.name.find(".gz")
        if idx == -1: continue
        base = p.name[:idx+3]  # include .gz
        grouped.setdefault(base, []).append(p)
    # 2) handle .gzNN or .gzNNN (no dot) where suffix is numeric right after .gz
    #    e.g. google_home_gen1.tar.gz00
    pattern = re.compile(r"^(?P<base>.+?\.gz)(?P<part>\d+)$")
    for p in all_files:
        m = pattern.match(p.name)
        if m:
            base = m.group("base")
            grouped.setdefault(base, []).append(p)

    if not grouped:
        v("[JOIN] no split parts found")
        return

    for base, parts in grouped.items():
        dest = ipfix_dir / base
        if dest.exists():
            v(f"[JOIN] joined target {dest.name} already exists; skipping")
            continue
        # natural sort: numeric suffix detection
        parts_sorted = sorted(parts, key=lambda p: [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', p.name)])
        v(f"[JOIN] joining {len(parts_sorted)} parts -> {base}")
        with dest.open("wb") as w:
            for pf in parts_sorted:
                v(f"  [JOIN] appending {pf.name} (size={pf.stat().st_size})")
                with pf.open("rb") as r:
                    shutil.copyfileobj(r, w)
        v(f"[JOIN] created {dest.name} (size={dest.stat().st_size})")

def decompress_gz_to_tar(ipfix_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_files = sorted(ipfix_dir.glob("*.gz"))
    if not gz_files:
        v("[DECOMPRESS] no .gz files found")
        return
    v(f"[DECOMPRESS] found {len(gz_files)} .gz files; streaming decompress to {out_dir}")
    for gz in gz_files:
        tar_name = gz.name[:-3]
        out_path = out_dir / tar_name
        if out_path.exists():
            v(f"[DECOMPRESS] {out_path.name} exists; skipping decompress")
            continue
        v(f"[DECOMPRESS] {gz.name} -> {out_path.name}")
        try:
            with gzip.open(gz, "rb") as f_in, out_path.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out, CHUNK)
            v(f"[DECOMPRESS] wrote {out_path} (size={out_path.stat().st_size})")
        except Exception as e:
            v(f"[DECOMPRESS] ERROR decompressing {gz.name}: {e}")
            try:
                out_path.unlink()
            except Exception:
                pass

def ensure_unique_path(dirpath: Path, name: str):
    candidate = dirpath / name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    i = 1
    while True:
        alt = dirpath / f"{stem}_{i}{suffix}"
        if not alt.exists():
            return alt
        i += 1

def stream_extract_tar_members(tar_path: Path, out_json_dir: Path):
    v(f"[EXTRACT] opening tar: {tar_path.name}")
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            if not members:
                v(f"[EXTRACT] {tar_path.name} contains no members; skipping")
                return
            v(f"[EXTRACT] {tar_path.name} contains {len(members)} member(s).")
            for i, m in enumerate(members):
                if not m.isfile():
                    v(f"  [EXTRACT] member {m.name} is not a file (type={m.type}); skipping")
                    continue
                member_name = Path(m.name).name
                # prefer the tar filename as prefix so names remain unique and understood
                dest_name = f"{tar_path.stem}__{member_name}"
                dest_path = ensure_unique_path(out_json_dir, dest_name)
                v(f"  [EXTRACT] ({i+1}/{len(members)}) extracting member -> {dest_path.name}  size={m.size}")
                out_json_dir.mkdir(parents=True, exist_ok=True)
                # extract streamed: tf.extractfile returns a file-like
                fobj = tf.extractfile(m)
                if fobj is None:
                    v(f"    [EXTRACT] WARNING: could not extract member {m.name}; skipped")
                    continue
                # write in chunks
                written = 0
                t0 = time.time()
                try:
                    with dest_path.open("wb") as fout:
                        while True:
                            chunk = fobj.read(CHUNK)
                            if not chunk:
                                break
                            fout.write(chunk)
                            written += len(chunk)
                    elapsed = time.time() - t0
                    v(f"    [EXTRACT] wrote {written} bytes to {dest_path.name} (elapsed {elapsed:.1f}s)")
                except Exception as e:
                    v(f"    [EXTRACT] ERROR writing member to {dest_path}: {e}")
                    try:
                        dest_path.unlink()
                    except Exception:
                        pass
                finally:
                    try:
                        fobj.close()
                    except Exception:
                        pass
    except Exception as e:
        v(f"[EXTRACT] ERROR opening tar {tar_path.name}: {e}")

def main():
    v(f"[MAIN] cwd: {Path.cwd()}")
    DATA_DIR.mkdir(exist_ok=True)
    TARS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    repo_dir = Path.cwd() / REPO_DIRNAME
    if repo_dir.exists():
        v(f"[MAIN] using existing repo at {repo_dir}")
    else:
        ok = try_git_clone(repo_dir)
        if not ok:
            repo_dir = download_and_unzip(Path.cwd())

    try:
        ipfix = find_ipfix(repo_dir)
    except FileNotFoundError as e:
        v(f"[MAIN] ERROR: {e}")
        sys.exit(1)

    # show ipfix quick list
    v("[MAIN] ipfix contents:")
    for f in sorted(ipfix.iterdir()):
        if f.is_file():
            v(f"  - {f.name}  size={f.stat().st_size}")

    join_split_parts(ipfix)
    decompress_gz_to_tar(ipfix, TARS_DIR)

    # Extract members from each tar to JSON_DIR (streamed)
    tar_files = sorted(TARS_DIR.glob("*.tar"))
    if not tar_files:
        v("[MAIN] no .tar files found in data/tars; nothing to extract")
        return
    v(f"[MAIN] extracting members from {len(tar_files)} .tar files into {JSON_DIR}")
    for t in tar_files:
        stream_extract_tar_members(t, JSON_DIR)

    v("[MAIN] Done. .tar files are in: " + str(TARS_DIR))
    v("[MAIN] Extracted member files are in: " + str(JSON_DIR))
    v("[MAIN] If you want cleanup (remove repo / gz / tar), tell me and I can add options.")

if __name__ == "__main__":
    main()
