import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
RAW_OUT_DIR = r"D:\model\assistant\src\wake_word_project\data\negative_raw"
STD_OUT_DIR = r"D:\model\assistant\src\wake_word_project\data_standardized\negative"

TARGET_SR = 16000          # must match training pipeline
TARGET_DURATION = 1.5      # seconds -- must match standardize_folder.py / positive recorder
WORD_TRIM_TOP_DB = 25       # only used in "word" mode -- matches positive recorder
CHUNK_OVERLAP = 0.0        # 0.0-0.9: fraction of overlap between consecutive
                            # chopped windows. 0 = no overlap (max diversity
                            # per second recorded); higher = more clips per
                            # session at the cost of more redundancy.

# Suggested categories -- these become subfolder names under both
# RAW_OUT_DIR and STD_OUT_DIR. WakeWordDataset already walks neg_dir
# recursively (os.walk), so any subfolder structure here works
# directly as training data with no extra steps.
CATEGORY_SUGGESTIONS = [
    "similar_sounding",   # words that sound like the wake word (jay, deck,
                           # take, etc.) -- WORD MODE: isolated utterances,
                           # trimmed + padded like the positive recorder.
    "general_speech",     # normal conversation, unrelated sentences
    "background_noise",   # TV, music, appliances, traffic, etc.
    "silence_room_tone",  # near-silent room ambience -- IMPORTANT: this is
                           # what teaches the model (and the energy gate)
                           # not to false-trigger on a quiet room.
    "other_people",       # other household members / voices speaking naturally
]

os.makedirs(RAW_OUT_DIR, exist_ok=True)
os.makedirs(STD_OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Recording
# ─────────────────────────────────────────────────────────────
def record_word(filename, duration=2.5, sr=TARGET_SR, label=""):
    print(f"\n  Recording in...")
    for i in [3, 2, 1]:
        print(f"    {i}...")
        time.sleep(0.5)
    print(f"  🎤 Say '{label}' now!" if label else "  🎤 Speak now!")

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("  ✅ Captured.")

    sf.write(filename, audio, sr, subtype="PCM_16")
    return filename


def record_continuous(filename, duration, sr=TARGET_SR, instructions=""):
    print(f"\n  Recording {duration:.0f}s of audio in...")
    for i in [3, 2, 1]:
        print(f"    {i}...")
        time.sleep(0.5)
    print(f"  🎙️  Recording — {instructions or 'do the thing this category needs.'}")

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("  ✅ Captured.")

    sf.write(filename, audio, sr, subtype="PCM_16")
    return filename


# ─────────────────────────────────────────────────────────────
# Standardization
# ─────────────────────────────────────────────────────────────
def standardize_word_clip(in_path, out_path, target_sr=TARGET_SR,
                           target_duration=TARGET_DURATION, top_db=WORD_TRIM_TOP_DB):
    """Same trim+pad convention as the positive-sample recorder -- for
    isolated-word negatives (similar-sounding words)."""
    y, sr = librosa.load(in_path, sr=target_sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)

    target_samples = int(target_sr * target_duration)
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, y, target_sr, subtype="PCM_16")


def chunk_continuous(in_path, out_dir, prefix, target_sr=TARGET_SR,
                      target_duration=TARGET_DURATION, overlap=CHUNK_OVERLAP):
    """
    Splits a longer continuous recording into consecutive fixed-length
    windows. Deliberately does NOT trim silence -- natural pauses,
    quiet moments, and room tone within the recording are legitimate,
    valuable negative examples (exactly what teaches the energy gate
    and the model not to false-trigger on a quiet room).
    """
    y, sr = librosa.load(in_path, sr=target_sr, mono=True)

    window_samples = int(target_sr * target_duration)
    step_samples = max(1, int(window_samples * (1 - overlap)))

    os.makedirs(out_dir, exist_ok=True)

    chunk_idx = 0
    start = 0
    saved_paths = []
    while start < len(y):
        chunk = y[start:start + window_samples]

        if len(chunk) < window_samples:
            if len(chunk) < window_samples * 0.3:
                # trailing scrap too short to be a useful sample -- discard
                break
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))

        out_path = os.path.join(out_dir, f"{prefix}_{chunk_idx:03d}.wav")
        sf.write(out_path, chunk, target_sr, subtype="PCM_16")
        saved_paths.append(out_path)

        chunk_idx += 1
        start += step_samples

    return saved_paths


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=== Real Negative Sample Recorder ===")
    print("\nSuggested categories (you can also type your own):")
    for c in CATEGORY_SUGGESTIONS:
        print(f"  - {c}")
    category = input("\nCategory name for this session: ").strip() or "general_speech"

    is_word_mode = category == "similar_sounding"
    # similar_sounding negatives are short isolated utterances (like the
    # positive recorder). Everything else is continuous audio that gets
    # auto-chopped into multiple fixed-length training windows.

    print("\nUsing default input device:", sd.query_devices(kind="input")["name"])
    print("(If this is wrong, run `python -m sounddevice` to list devices "
          "and set sd.default.device manually.)\n")

    raw_dir = os.path.join(RAW_OUT_DIR, category)
    std_dir = os.path.join(STD_OUT_DIR, category)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(std_dir, exist_ok=True)

    if is_word_mode:
        try:
            num_samples = int(input("How many takes do you want to record? (e.g. 20): ").strip())
        except ValueError:
            num_samples = 20
            print(f"Invalid input, defaulting to {num_samples}.")

        input("Press Enter when ready to start recording...")

        for i in range(1, num_samples + 1):
            print(f"\n--- Take {i}/{num_samples} ---")
            raw_path = os.path.join(raw_dir, f"{category}_{i:03d}.wav")
            record_word(raw_path, label="a similar-sounding word/phrase (e.g. jay, deck, take)")

            redo = input("  Keep this take? (Enter = yes, 'r' = redo): ").strip().lower()
            while redo == "r":
                record_word(raw_path, label="a similar-sounding word/phrase (e.g. jay, deck, take)")
                redo = input("  Keep this take? (Enter = yes, 'r' = redo): ").strip().lower()

            std_path = os.path.join(std_dir, f"{category}_{i:03d}.wav")
            standardize_word_clip(raw_path, std_path)
            print(f"  → Standardized and saved to {std_path}")

        print(f"\n✅ Done! {num_samples} negative word samples recorded and standardized.")

    else:
        try:
            num_sessions = int(input("How many recording sessions? (e.g. 5): ").strip())
        except ValueError:
            num_sessions = 5
            print(f"Invalid input, defaulting to {num_sessions}.")
        try:
            session_seconds = float(input("Seconds per session? (e.g. 20): ").strip())
        except ValueError:
            session_seconds = 20.0
            print(f"Invalid input, defaulting to {session_seconds}s.")

        instructions = {
            "background_noise": "play TV/music, run appliances, let ambient noise happen",
            "silence_room_tone": "stay quiet -- just capture the room as it normally is",
            "other_people": "have someone else talk naturally, not to the assistant",
            "general_speech": "talk naturally about anything, not directed at the assistant",
        }.get(category, "do whatever fits this category")

        print(f"\nEach {session_seconds:.0f}s session will be auto-chopped into "
              f"~{int(session_seconds // TARGET_DURATION)} negative training clips.")
        input("Press Enter when ready to start recording...")

        total_chunks = 0
        for i in range(1, num_sessions + 1):
            print(f"\n--- Session {i}/{num_sessions} ({category}) ---")
            raw_path = os.path.join(raw_dir, f"{category}_session{i:02d}.wav")
            record_continuous(raw_path, duration=session_seconds, instructions=instructions)

            redo = input("  Keep this session? (Enter = yes, 'r' = redo): ").strip().lower()
            while redo == "r":
                record_continuous(raw_path, duration=session_seconds, instructions=instructions)
                redo = input("  Keep this session? (Enter = yes, 'r' = redo): ").strip().lower()

            chunks = chunk_continuous(raw_path, std_dir, prefix=f"{category}_s{i:02d}")
            total_chunks += len(chunks)
            print(f"  → Chopped into {len(chunks)} standardized clips in {std_dir}")

        print(f"\n✅ Done! {total_chunks} negative samples standardized across "
              f"{num_sessions} sessions.")

    print(f"   Raw files:          {raw_dir}")
    print(f"   Standardized files: {std_dir}")


if __name__ == "__main__":
    main()