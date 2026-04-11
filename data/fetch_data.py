import os
import requests
    
    
def fetch_data():
    BASE_DIR = os.path.dirname(__file__)
    SAVE_DIR = os.path.join(BASE_DIR, "zenodo")
    
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    urls = [
        "https://zenodo.org/api/records/15968495/files/avg_item.tsv/content",
        "https://zenodo.org/api/records/15968495/files/avg_user.tsv/content",
        "https://zenodo.org/api/records/15968495/files/avg_query.tsv/content",
        "https://zenodo.org/api/records/15968495/files/cross_user.tsv/content",
        "https://zenodo.org/api/records/15968495/files/cross_query.tsv/content",
        "https://zenodo.org/api/records/15968495/files/cross_item.tsv/content",
    ]
    
    
    CHUNK_SIZE = 1024 * 1024  # 1MB
    
    for url in urls:
        filename = url.split("/")[-2]
        filepath = os.path.join(SAVE_DIR, filename)
    
        print(f"\nDownloading {filename}...")
    
        # get server file size
        head = requests.head(url)
        total_size = int(head.headers.get("content-length", 0))
    
        existing_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    
        # ✅ CASE 1: already fully downloaded
        if existing_size == total_size and total_size > 0:
            print(f"✅ {filename} already complete ({existing_size//(1024*1024)} MB)")
            continue
    
        # ❌ CASE 2: local file bigger than server → corrupted
        if existing_size > total_size:
            print(f"⚠️ Corrupted file detected. Deleting {filename}")
            os.remove(filepath)
            existing_size = 0
    
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
    
        with requests.get(url, stream=True, headers=headers) as r:
            # 🔥 Handle 416 explicitly
            if r.status_code == 416:
                print(f"✅ {filename} already fully downloaded (server confirmed)")
                continue
    
            r.raise_for_status()
    
            mode = "ab" if existing_size > 0 else "wb"
            downloaded = existing_size
    
            with open(filepath, mode) as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
    
                        percent = (downloaded / total_size) * 100 if total_size else 0
                        print(f"\r{percent:.2f}% ({downloaded//(1024*1024)} MB)", end="")
    
        print("\nDone.")
    