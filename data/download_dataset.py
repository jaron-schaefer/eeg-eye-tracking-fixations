import os
import zipfile
import requests
from tqdm import tqdm

def download_dataset():
    """
    Downloads all files from a Figshare dataset article, unzips them if needed,
    and removes the original zip files.
    """
    out_dir = "data/data/raw_data"
    os.makedirs(out_dir, exist_ok=True)
    api_url = f"https://api.figshare.com/v2/articles/26943565/files"
    files = requests.get(api_url).json()

    for f in files:
        file_path = os.path.join(out_dir, f["name"])

        # Download with progress bar
        with requests.get(f["download_url"], stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            chunk_size = 8192

            with open(file_path, "wb") as out, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=f["name"]
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    out.write(chunk)
                    pbar.update(len(chunk))

        # Unzip if needed and remove zip
        if file_path.lower().endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(out_dir)
            os.remove(file_path)