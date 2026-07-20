import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
WAKE_WORD_LABEL = "J-Dek"

RAW_OUT_DIR = r"D:\model\assistant\src\wake_word_project\data\real_world_raw"
STD_OUT_DIR = r"D:\model\assistant\src\wake_word_project\data_standardized\positive"

RECORD_SECONDS = 2.5      # raw capture window (gives buffer for trimming)
TARGET_SR = 16000         # must match training pipeline
TARGET_DURATION = 1.5     # seconds -- must match standardize_folder.py
TRIM_SILENCE = True       # trims leading/trailing silence before padding

os.makedirs(RAW_OUT_DIR, exist_ok=True)
os.makedirs(STD_OUT_DIR, exist_ok=True)


def record_clip(filename, duration=RECORD_SECONDS, sr=TARGET_SR):
    print(f"\n  Recording in...")
    for i in [3, 2, 1]:
        print(f"    {i}...")
        time.sleep(0.5)
    print(f"  🎤 Say '{WAKE_WORD_LABEL}' now!")

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("  ✅ Captured.")

    sf.write(filename, audio, sr, subtype="PCM_16")
    return filename


def standardize_clip(in_path, out_path, target_sr=TARGET_SR, target_duration=TARGET_DURATION):
    y, sr = librosa.load(in_path, sr=target_sr, mono=True)

    if TRIM_SILENCE:
        y, _ = librosa.effects.trim(y, top_db=25)

    target_samples = int(target_sr * target_duration)
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, y, target_sr, subtype="PCM_16")


def main():
    print(f"=== Real Voice Recorder: '{WAKE_WORD_LABEL}' ===")
    try:
        num_samples = int(input("How many takes do you want to record? (e.g. 20): ").strip())
    except ValueError:
        num_samples = 20
        print(f"Invalid input, defaulting to {num_samples}.")

    # Optional: list input devices if the default mic isn't the one you want
    print("\nUsing default input device:", sd.query_devices(kind="input")["name"])
    print("(If this is wrong, run `python -m sounddevice` to list devices "
          "and set sd.default.device manually.)\n")

    input("Press Enter when ready to start recording...")

    for i in range(1, num_samples + 1):
        print(f"\n--- Take {i}/{num_samples} ---")
        raw_path = os.path.join(RAW_OUT_DIR, f"real_{i:03d}.wav")
        record_clip(raw_path)

        # Playback-free quick check: let user redo a bad take
        redo = input("  Keep this take? (Enter = yes, 'r' = redo): ").strip().lower()
        while redo == "r":
            record_clip(raw_path)
            redo = input("  Keep this take? (Enter = yes, 'r' = redo): ").strip().lower()

        std_path = os.path.join(STD_OUT_DIR, f"real_{i:03d}.wav")
        standardize_clip(raw_path, std_path)
        print(f"  → Standardized and saved to {std_path}")

    print(f"\n✅ Done! {num_samples} real samples recorded and standardized.")
    print(f"   Raw files:          {RAW_OUT_DIR}")
    print(f"   Standardized files: {STD_OUT_DIR}")


if __name__ == "__main__":
    main()