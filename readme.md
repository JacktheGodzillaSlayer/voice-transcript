# Voice Transcription Tool

This tool transcribes or translates audio/video files into text using OpenAI's Whisper model. It supports various languages and can run on both CPU and GPU (CUDA).

## Prerequisites

- Python 3.9 or higher
- OpenAI Whisper (`openai-whisper`)
- PyTorch (`torch`)

## Installation

1. Install Python 3.9 or higher from [python.org](https://www.python.org/).
2. Install the required dependencies:

   ```bash
   pip install openai-whisper torch
   ```

3. (Optional) For GPU support, ensure you have the appropriate version of PyTorch with CUDA installed. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Usage

### Basic Command

```bash
python transcript.py \
    --model turbo \
    --voicesource ./source/source.mp4 \
    --lang en \
    --output ./gpt-prompt.txt
```

### Arguments

| Argument        | Description                                                                 | Default Value | Required |
|-----------------|-----------------------------------------------------------------------------|---------------|----------|
| `--lang`        | Language code of the audio (e.g., `en` for English, `yue` for Cantonese).   | `yue`         | No       |
| `--voicesource` | Path to the audio/video file to transcribe.                                 | -             | Yes      |
| `--output`      | Path to save the transcription text file. If not provided, the output file will be saved in the same directory as the input file with a `.txt` extension. | -             | No       |
| `--model`       | Whisper model size (e.g., `base`, `small`, `medium`, `large`, `turbo`).     | `base`        | No       |
| `--model_file`  | Path to a custom Whisper model file.                                        | -             | No       |
| `--device`      | Device to use (`cuda` for GPU or `cpu` for CPU).                            | `cuda`        | No       |

### Examples

1. **Transcribe an English audio file:**
   ```bash
   python transcript.py \
       --voicesource ./audio/english.mp3 \
       --lang en \
       --output ./transcriptions/english.txt
   ```

2. **Translate a Cantonese audio file to English:**
   ```bash
   python transcript.py \
       --voicesource ./audio/cantonese.mp3 \
       --lang yue \
       --translate \
       --output ./translations/cantonese_english.txt
   ```

3. **Use a custom model:**
   ```bash
   python transcript.py \
       --voicesource ./audio/spanish.mp3 \
       --lang es \
       --model_file ./custom_models/spanish_model.pt \
       --output ./transcriptions/spanish.txt
   ```

4. **Run on CPU:**
   ```bash
   python transcript.py \
       --voicesource ./audio/french.mp3 \
       --lang fr \
       --device cpu \
       --output ./transcriptions/french.txt
   ```

## Output

The tool generates a text file containing the transcription or translation of the input audio/video file. If the `--output` argument is not provided, the output file will be saved in the same directory as the input file with a `.txt` extension.

For example:
- Input: `./source/source.mp4`
- Output: `./source/source.txt`

## Supported Models

The following Whisper models are supported:
- `tiny`
- `base`
- `small`
- `medium`
- `large`
- `turbo`

## Supported Languages

The tool supports all languages supported by Whisper. Common language codes include:
- English: `en`
- Cantonese: `yue`
- Mandarin: `zh`
- Spanish: `es`
- French: `fr`
- German: `de`

For a full list of supported languages, refer to the [Whisper documentation](https://github.com/openai/whisper).

## Notes

- Ensure your system has sufficient memory (RAM/VRAM) for larger models like `large` or `turbo`.
- For GPU usage, ensure CUDA is properly installed and configured.
- If the `--translate` flag is used, the output will always be in English.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
