import os
import requests
import tarfile
import zipfile
import io

# Configuration
BASE_DIR = r"D:\model\assistant\src\wake_word_project\data"
NEGATIVE_DIR = os.path.join(BASE_DIR, "negative")
BACKGROUND_DIR = os.path.join(BASE_DIR, "background")

# CORRECT DATASET URLS
# Google Speech Commands v0.02 (approx 2.4 GB)
SPEECH_COMMANDS_URL = "http://tensorflow.org"
# ESC-50 Environmental Sound Classification (approx 600 MB)
# Using the correct direct master zip download link
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

def download_and_extract(url, target_dir, is_tar=True):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    print(f"Starting download from: {url}")
    # Using stream=True is critical for the 2.4GB Speech Commands file
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        
        if is_tar:
            print("Extracting Tarball (streaming mode)...")
            # We open from the raw response stream directly for memory efficiency
            with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                tar.extractall(path=target_dir)
        else:
            print("Downloading and Extracting Zip...")
            try:
                # Zips generally need to be fully downloaded before extraction
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(path=target_dir)
            except zipfile.BadZipFile:
                print(f"Error: The file at {url} is not a valid zip file or was corrupted.")
                
    print(f"Successfully populated {target_dir}")

if __name__ == "__main__":
    # 1. Download Negative Speech (Google Speech Commands)
    # This dataset contains ~105,000 utterances for training negatives
    #download_and_extract(SPEECH_COMMANDS_URL, NEGATIVE_DIR, is_tar=True)

    # 2. Download Background Noise (ESC-50)
    # This dataset contains 2,000 environmental recordings across 50 classes
    download_and_extract(ESC50_URL, BACKGROUND_DIR, is_tar=False)