import os
import subprocess

# Paths
input_dir = "examples/input_voices"
output_dir = "examples/output_voice"

combined_mp3 = os.path.join(output_dir, "combined_temp.mp3")
final_mp3 = os.path.join(output_dir, "narration.mp3")
final_wav = os.path.join(output_dir, "narration.wav")

file_list_path = "mp3_list.txt"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Input MP3 files (order matters)
mp3_files = [
    "sample_1.mp3",
    "sample_2.mp3",
    "sample_3.mp3"
]

# Create concat list
with open(file_list_path, "w", encoding="utf-8") as f:
    for file in mp3_files:
        full_path = os.path.abspath(os.path.join(input_dir, file))
        f.write(f"file '{full_path}'\n")

# 1️⃣ Lossless MP3 concat (no re-encode)
subprocess.run([
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", file_list_path,
    "-c", "copy",
    combined_mp3
], check=True)

# 2️⃣ Convert to WAV (uncompressed)
''' 
subprocess.run([
    "ffmpeg", "-y",
    "-i", combined_mp3,
    "-acodec", "pcm_s16le",
    "-ar", "44100",
    "-ac", "2",
    final_wav
], check=True)
'''

# 3️⃣ Normalize audio to -14 dB (web video standard) and convert to MP3
subprocess.run([
    "ffmpeg", "-y",
    "-i", combined_mp3,
    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,acompressor=threshold=-20dB:ratio=4:attack=5:release=50",
    "-c:a", "libmp3lame",
    "-b:a", "192k",
    final_mp3
], check=True)

# Cleanup temp files
os.remove(file_list_path)
os.remove(combined_mp3)

print("✅ Audio processing complete!")
# print(f"🎧 WAV saved at: {final_wav}")
print(f"🎧 MP3 saved at: {final_mp3}")