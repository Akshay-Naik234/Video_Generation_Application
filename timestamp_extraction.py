import whisper
import json
import sys
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

def extract_timestamps(audio_file, model_size="base"):
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing {audio_file}...")
    result = model.transcribe(audio_file, word_timestamps=True)
    
    # Extract segments with timestamps
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip()
        })
    
    # Get total duration
    duration = round(result["segments"][-1]["end"], 3) if result["segments"] else 0
    
    output = {
        "duration": duration,
        "segments": segments
    }
    
    return output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whisper_timestamps.py <audio_file.mp3> [model_size]")
        print("Model sizes: tiny, base, small, medium, large")
        sys.exit(1)
    
    audio_file = "examples/audio/narration.mp3"
    model_size = sys.argv[1] if len(sys.argv) > 2 else "base"
    
    result = extract_timestamps(audio_file, model_size)
    
    # Save to JSON file
    output_file = audio_file.rsplit(".", 1)[0] + "_timestamps.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nTimestamps saved to: {output_file}")
    print(f"Duration: {result['duration']} seconds")
    print(f"Segments: {len(result['segments'])}")