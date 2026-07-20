import asyncio
import os
import itertools
import edge_tts

WAKE_WORD = "J-Dek"
OUTPUT_DIR = r"D:\model\assistant\src\wake_word_project\data\positive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prosody variations: each combo of (rate, pitch, volume) applied per voice.
# Keep it modest so the wake word stays recognizable, but diverse enough
# to meaningfully augment your training set.
RATES = ["-15%", "+0%", "+15%"]
PITCHES = ["-30Hz", "+0Hz", "+30Hz"]
VOLUMES = ["+10%"]  # add "-10%", "+10%" here too if you want even more

# How many prosody variants to generate per voice.
# len(RATES) * len(PITCHES) * len(VOLUMES) = 9 combos available;
# we just take a subset per voice to control total output count.
VARIANTS_PER_VOICE = 8

MAX_CONCURRENT = 16  # be polite to the service / avoid rate limiting


def build_variant_combos():
    all_combos = list(itertools.product(RATES, PITCHES, VOLUMES))
    return all_combos


async def generate_one(sem, voice_short_name, voice_name, idx, rate, pitch, volume):
    filename = os.path.join(
        OUTPUT_DIR,
        f"voice_{idx}_{voice_short_name}_r{rate.replace('%','pct')}_p{pitch.replace('Hz','hz')}.wav".replace(
            "+", "pos"
        ).replace("-", "neg"),
    )

    async with sem:
        try:
            communicate = edge_tts.Communicate(
                WAKE_WORD,
                voice_name,
                rate=rate,
                pitch=pitch,
                volume=volume,
            )
            await communicate.save(filename)
            return True
        except Exception as e:
            print(f"  [skip] {voice_short_name} ({rate}, {pitch}): {e}")
            return False


async def generate_samples():
    all_voices = await edge_tts.VoicesManager.create()
    voices = all_voices.voices

    english_voices = [v for v in voices if "en-" in v["Locale"]]
    print(f"Found {len(voices)} total voices, {len(english_voices)} English voices.")

    combos = build_variant_combos()
    combos = combos[:VARIANTS_PER_VOICE]  # limit per-voice variant count

    total_planned = len(english_voices) * len(combos)
    print(f"Planning {len(english_voices)} voices x {len(combos)} variants = {total_planned} files.")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    for i, v in enumerate(english_voices):
        for rate, pitch, volume in combos:
            tasks.append(
                generate_one(sem, v["ShortName"], v["Name"], i, rate, pitch, volume)
            )

    results = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        ok = await coro
        results.append(ok)
        completed += 1
        if completed % 20 == 0:
            print(f"Progress: {completed}/{len(tasks)}")

    count = sum(1 for r in results if r)
    print(f"Done! Saved {count}/{len(tasks)} positive samples to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(generate_samples())