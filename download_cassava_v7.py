import os
import sys
import time
import zipfile
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def _get_api_key():
    key = os.getenv("ROBOFLOW_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ROBOFLOW_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

ROBOFLOW_API_KEY = _get_api_key()
DATASET_URL = "https://api.roboflow.com/cassava-disease/cassava-zpvrj/7/yolov8?api_key=" + ROBOFLOW_API_KEY
TARGET_ZIP = Path("d:/UNDIP/SKRIPSI/cassava-v7.zip")
TARGET_DIR = Path("d:/UNDIP/SKRIPSI/cassava-v7")
NUM_CHUNKS = 8

def get_download_url():
    print("[1/4] Fetching Roboflow export link...")
    resp = requests.get(DATASET_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    link = data.get("export", {}).get("link")
    if not link:
        raise RuntimeError("No export link found in response: " + str(data))
    
    print("[2/4] Resolving signed download URL...")
    head_resp = requests.head(link, allow_redirects=True, timeout=30)
    final_url = head_resp.url
    content_length = int(head_resp.headers.get("content-length", 0))
    print(f"      Content-Length: {content_length:,} bytes ({content_length / (1024*1024):.2f} MB)")
    return final_url, content_length

def download_chunk(url, start_byte, end_byte, chunk_idx, part_file):
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    for attempt in range(5):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(part_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return chunk_idx, True
        except Exception as e:
            time.sleep(2 * (attempt + 1))
            if attempt == 4:
                raise e

def parallel_download(url, total_size):
    print(f"[3/4] Downloading {total_size / (1024*1024):.2f} MB with {NUM_CHUNKS} parallel streams...")
    t0 = time.time()
    chunk_size = total_size // NUM_CHUNKS
    part_files = []
    ranges = []

    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = total_size - 1 if i == NUM_CHUNKS - 1 else (i + 1) * chunk_size - 1
        part_file = TARGET_ZIP.parent / f"cassava-v7.part_{i}"
        part_files.append(part_file)
        ranges.append((start, end, i, part_file))

    with ThreadPoolExecutor(max_workers=NUM_CHUNKS) as executor:
        futures = [executor.submit(download_chunk, url, s, e, idx, p) for s, e, idx, p in ranges]
        for f in futures:
            f.result()

    dt = time.time() - t0
    print(f"      Download completed in {dt:.2f}s ({total_size / (1024*1024) / dt:.2f} MB/s)")

    print("      Assembling chunks into single zip...")
    with open(TARGET_ZIP, "wb") as outfile:
        for p in part_files:
            with open(p, "rb") as infile:
                while True:
                    buf = infile.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    outfile.write(buf)
            p.unlink()

def extract_zip():
    print(f"[4/4] Extracting {TARGET_ZIP} to {TARGET_DIR}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(TARGET_ZIP, 'r') as z:
        z.extractall(TARGET_DIR)
    dt = time.time() - t0
    print(f"      Extracted in {dt:.2f}s")
    
    # Verify image count
    images = []
    for split in ["train", "valid", "test"]:
        img_dir = TARGET_DIR / split / "images"
        if img_dir.exists():
            cnt = len([p for p in img_dir.iterdir() if p.is_file()])
            print(f"      {split}/images: {cnt:,} files")
            images.extend(list(img_dir.iterdir()))
    print(f"      TOTAL images extracted: {len(images):,}")
    
    # Remove zip to clean up
    TARGET_ZIP.unlink(missing_ok=True)
    print("      Removed temporary zip file.")

if __name__ == "__main__":
    url, total_size = get_download_url()
    parallel_download(url, total_size)
    extract_zip()
