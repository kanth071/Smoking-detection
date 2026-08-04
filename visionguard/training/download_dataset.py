"""
Download a pre-labelled cigarette dataset for YOLOv11 training.
"""
import os
import sys
import yaml
import zipfile
import urllib.request

DATASET_URL = "https://github.com/kanth071/Smoking-detection/releases/download/v1.0/cigarette_dataset.zip"

def download_and_extract(dataset_dir="dataset"):
    os.makedirs(dataset_dir, exist_ok=True)
    zip_path = os.path.join(dataset_dir, "dataset.zip")

    print(f"[download] Fetching cigarette dataset into {dataset_dir}...")
    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("[download] Extracting dataset archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        os.remove(zip_path)
        print(f"[download] Successfully set up dataset in {dataset_dir}")
        return True
    except Exception as e:
        print(f"[download] Download error: {e}")
        return False

if __name__ == "__main__":
    download_and_extract()
