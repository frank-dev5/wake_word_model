import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import soundfile as sf

# ─────────────────────────────────────────────────────────────
# 1. THE DATASET CLASS (The Brain)
# ─────────────────────────────────────────────────────────────
class WakeWordDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, bg_dir, target_sr=16000, n_mels=128, max_negatives=None):
        """
        max_negatives: if set, only the first N negative files are used.
                       Handy for quick visualization/sanity-check runs.
                       Leave as None (default) for real training so the
                       full negative set is used.
        """
        self.target_sr = target_sr
        self.n_mels = n_mels

        # Collect all file paths and labels
        self.samples = []
        if os.path.exists(pos_dir):
            for f in os.listdir(pos_dir):
                if f.endswith(".wav"):
                    self.samples.append((os.path.join(pos_dir, f), 1))

        if os.path.exists(neg_dir):
            neg_files = []
            for root, _, files in os.walk(neg_dir):
                for f in files:
                    if f.endswith(".wav"):
                        neg_files.append(os.path.join(root, f))
            if max_negatives is not None:
                random.shuffle(neg_files)
                neg_files = neg_files[:max_negatives]
            for f in neg_files:
                self.samples.append((f, 0))

        # Collect background noise for augmentation
        self.bg_files = []
        if os.path.exists(bg_dir):
            for root, _, files in os.walk(bg_dir):
                for f in files:
                    if f.endswith(".wav"):
                        self.bg_files.append(os.path.join(root, f))

        # Define transformations
        self.melspec_transform = T.MelSpectrogram(
            sample_rate=target_sr, n_mels=n_mels, n_fft=1024, hop_length=512
        )
        self.db_transform = T.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load directly with soundfile, bypass torchaudio's backend issues
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
            
        waveform = torch.from_numpy(audio).unsqueeze(0)  # add channel dim: (1, num_samples)

        # Data Augmentation: Mix background noise (50% chance for positive samples)
        if label == 1 and random.random() > 0.5 and self.bg_files:
            bg_path = random.choice(self.bg_files)
            bg_audio, _ = sf.read(bg_path, dtype="float32")
            bg_waveform = torch.from_numpy(bg_audio).unsqueeze(0)

            if bg_waveform.shape[1] >= waveform.shape[1]:
                bg_waveform = bg_waveform[:, :waveform.shape[1]]
                snr = random.uniform(0.1, 0.3)
                waveform = waveform + (bg_waveform * snr)

        # Transform to Log-Mel Spectrogram
        mel_spec = self.melspec_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)

        return log_mel_spec, label


# ─────────────────────────────────────────────────────────────
# 2. THE EXECUTION BLOCK (The Action) — quick visual sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Define your paths
    POS_DIR = r"D:\model\assistant\src\wake_word_project\data_standardized\positive"
    NEG_DIR = r"D:\model\assistant\src\wake_word_project\data_standardized\negative"
    BG_DIR = r"D:\model\assistant\src\wake_word_project\data_standardized\background"

    # Initialize Dataset — cap negatives for a fast test run only
    dataset = WakeWordDataset(POS_DIR, NEG_DIR, BG_DIR, max_negatives=100)

    if len(dataset) == 0:
        print("❌ Error: No files found. Check your paths!")
    else:
        # Grab the very first sample (usually a positive one)
        mel_spec, label = dataset[0]
        print(f"✅ Successfully loaded sample. Matrix Shape: {mel_spec.shape}")

        # 3. PLOT THE VISUAL
        plt.figure(figsize=(10, 4))
        # mel_spec is [1, 128, T]; squeeze() makes it [128, T] for the plot
        plt.imshow(mel_spec.squeeze().numpy(), origin='lower', aspect='auto', cmap='magma')
        plt.title(f"Visual Fingerprint of: {'Wake Word' if label == 1 else 'Background/Other'}")
        plt.ylabel("Frequency (Mel Bins)")
        plt.xlabel("Time (Frames)")
        plt.colorbar(label='Decibels (dB)')
        plt.tight_layout()
        plt.show()