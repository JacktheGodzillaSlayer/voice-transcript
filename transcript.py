import argparse
import whisper
import torch
import os
import sys
import re
from tqdm import tqdm

class ProgressWatcher:
    """Helper class to intercept Whisper's verbose output for progress tracking"""
    def __init__(self, progress_bar, original_stdout):
        self.progress_bar = progress_bar
        self.original_stdout = original_stdout
        self.re_segment = re.compile(r'\d+:\d+\.\d+ --> (\d+:\d+\.\d+)')

    def write(self, text):
        # Update progress from Whisper's verbose output
        for line in text.split('\n'):
            match = self.re_segment.search(line)
            if match:
                end_time_str = match.group(1)
                minutes, seconds = map(float, end_time_str.split(':'))
                end_time = minutes * 60 + seconds
                self.progress_bar.n = min(end_time, self.progress_bar.total)
                self.progress_bar.refresh()
        self.original_stdout.write(text)

    def flush(self):
        self.original_stdout.flush()

def transcribe_audio(file_path, language, output_file=None, 
                    model_name="base", model_file=None, device="cuda"):
    # Device setup and memory management
    if device == "cuda":
        torch.cuda.empty_cache()
        if not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
            device = "cpu"

    # Model loading
    model = whisper.load_model(
        model_file or model_name,
        device=device
    )
    print(f"Using language code: {language}")

    # Audio duration calculation
    audio = whisper.load_audio(file_path)
    duration = len(audio) / 16000  # Whisper resamples to 16kHz
    print(f"Audio duration: {duration:.2f} seconds")

    # Transcription with progress bar
    with tqdm(total=duration, unit='s', desc='Transcribing') as pbar:
        original_stdout = sys.stdout
        sys.stdout = ProgressWatcher(pbar, original_stdout)
        
        # Actual transcription
        result = model.transcribe(
            file_path,
            language=language,
            verbose=True  # Required for progress tracking
        )
        
        # Restore stdout and finalize progress
        sys.stdout = original_stdout
        pbar.n = duration
        pbar.refresh()

    # Output file handling
    if output_file is None:
        base_name = os.path.splitext(file_path)[0]
        output_file = base_name + ".txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"\nTranscription saved to: {output_file}")

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    parser = argparse.ArgumentParser(description="Speech-to-text transcription with progress")
    parser.add_argument("--lang", type=str, default="yue", help="Language code (e.g., 'en')")
    parser.add_argument("--voicesource", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--model_file", type=str, help="Custom model file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    transcribe_audio(
        args.voicesource,
        args.lang,
        args.output,
        args.model,
        args.model_file,
        args.device
    )

if __name__ == "__main__":
    main()