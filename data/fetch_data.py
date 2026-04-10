import os
import requests

BASE_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(BASE_DIR, "zenodo")

os.makedirs(SAVE_DIR, exist_ok=True)

urls = [
    "https://zenodo.org/records/15968495/files/avg_item.tsv",
    "https://zenodo.org/records/15968495/files/avg_user.tsv",
    "https://zenodo.org/records/15968495/files/avg_query.tsv",
    "https://zenodo.org/records/15968495/files/cross_user.tsv",
    "https://zenodo.org/records/15968495/files/cross_query.tsv",
    "https://zenodo.org/records/15968495/files/cross_item.tsv",
]

headers = {"User-Agent": "Mozilla/5.0"}

for url in urls:
    filename = url.split("/")[-1]
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath):
        print(f"Skipping {filename}, already exists.")
        continue

    print(f"Downloading {filename}...")
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

print("All files downloaded.")