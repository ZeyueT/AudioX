# AudioX ComfyUI Extension

This package provides a simple ComfyUI node for running the AudioX model.

## Installation

1. Clone the AudioX repository or install it via `pip`:
   ```bash
   git clone https://github.com/ZeyueT/AudioX.git
   ```
2. Copy the `audiox_comfyui` folder into ComfyUI's `custom_nodes` directory or install the repository with pip so that the package is available in Python's path.

## Usage

After copying the node into your ComfyUI installation, restart ComfyUI. A new node **AudioX Generator** appears under the `audiox` category. The node accepts:

- `text_prompt`: text description of the desired audio or music.
- `video_path`: optional path to a video file used as conditioning.
- `audio_path`: optional audio file for additional conditioning.
- `seed`: random seed (use `-1` for random).

The node outputs paths to the generated audio file and, when a video path is provided, a merged video with the generated audio.

## License

This extension is distributed under the same [CC‑BY‑NC](../LICENSE) license as the rest of AudioX.
