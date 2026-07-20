import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Configuration
BASE_DIR = r"D:\model\assistant\src\wake_word_project\data"
OUT_BASE_DIR = r"D:\model\assistant\src\wake_word_project\data_standardized"

POS_DIR = os.path.join(BASE_DIR, "positive")
NEG_DIR = os.path.join(BASE_DIR, "negative")
BACK_DIR = os.path.join(BASE_DIR, "background")


def standardize_folder(folder_path, label_name, out_root):
    print(f"Standardizing {label_name} files...")

    files = []
    for root, _, filenames in os.walk(folder_path):
        for f in filenames:
            if f.endswith(".wav"):
                files.append(os.path.join(root, f))

    for file_path in tqdm(files, desc=f"Processing {label_name}"):
        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)

            target_samples = int(16000 * 1.5)
            if len(y) > target_samples:
                y = y[:target_samples]
            else:
                y = np.pad(y, (0, target_samples - len(y)))

            # Mirror the relative path into the output folder
            rel_path = os.path.relpath(file_path, folder_path)
            out_path = os.path.join(out_root, os.path.basename(folder_path), rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            sf.write(out_path, y, 16000, subtype='PCM_16')

        except Exception as e:
            print(f"\nSkipping {file_path} due to error: {e}")


if __name__ == "__main__":
    for path, label in [(POS_DIR, "Positive"), (NEG_DIR, "Negative"), (BACK_DIR, "Background")]:
        if os.path.exists(path):
            standardize_folder(path, label, OUT_BASE_DIR)
    print("All datasets successfully standardized!")