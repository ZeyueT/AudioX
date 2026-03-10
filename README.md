# 🎧 AudioX: A Unified Framework for Anything-to-Audio Generation

[![arXiv](https://img.shields.io/badge/arXiv-2503.10522-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2503.10522)
[![Project Page](https://img.shields.io/badge/GitHub.io-Project-blue?logo=Github&style=flat-square)](https://zeyuet.github.io/AudioX/)
[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/HKUSTAudio/audiox)
[![🤗 Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/HKUSTAudio/AudioX-IFcaps)
[![🤗 Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Zeyue7/AudioX)

---

**This is the official repository for "[AudioX: A Unified Framework for Anything-to-Audio Generation](https://arxiv.org/pdf/2503.10522)" (Accepted to ICLR 2026).**

## 📺 Demo Video

https://github.com/user-attachments/assets/a065b764-3ea4-4e30-aad5-c8cec51c2f0d

---

## ✨ Abstract

Audio and music generation based on flexible multimodal control signals is a widely applicable topic, with the following key challenges: 1) a unified multimodal modeling framework, and 2) large-scale, high-quality training data.
As such, we propose AudioX, a unified framework for anything-to-audio generation that integrates varied multimodal conditions (\ie, text, video, and audio signals) in this work. The core design in this framework is a Multimodal Adaptive Fusion module, which enables the effective fusion of diverse multimodal inputs, enhancing cross-modal alignment and improving overall generation quality.
To train this unified model, we construct a large-scale, high-quality dataset, IF-caps, comprising over 7 million samples curated through a structured data annotation pipeline. This dataset provides comprehensive supervision for multimodal-conditioned audio generation. We benchmark AudioX against state-of-the-art methods across a wide range of tasks, finding that our model achieves superior performance, especially in text-to-audio and text-to-music generation. These results demonstrate our method is capable of audio generation under multimodal control signals, showing powerful instruction-following potential.


## ✨ Teaser

<p align="center">
  <img width="1819" height="783" alt="teaser" src="https://github.com/user-attachments/assets/ca7e768a-a113-423d-ac7e-40b7437c5538" />
</p>
<p style="text-align: left;">Performance comparison of AudioX against baselines. (a) Comprehensive comparison across multiple benchmarks via Inception Score. (b) Results on instruction-following benchmarks.</p>

## ✨ Method

<p align="center">
  <img width="1813" height="658" alt="method" src="https://github.com/user-attachments/assets/f040df46-faf2-4d0e-82f4-65dc2f625558" />
</p>
<p align="center">Overview of the AudioX Framework.</p>

---

## 🛠️ Environment Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg and libsndfile

### Installation

```bash
# Clone the repository
git clone https://github.com/ZeyueT/AudioX.git
cd AudioX

# Create conda environment
conda create -n AudioX python=3.8.20
conda activate AudioX

# Install dependencies
pip install git+https://github.com/ZeyueT/AudioX.git
conda install -c conda-forge ffmpeg libsndfile
```

---

## 🪄 Pretrained Checkpoints

We provide three pretrained models on 🤗 [Hugging Face](https://huggingface.co/HKUSTAudio):

1. **AudioX** - Base model for general audio and music generation
2. **AudioX-MAF** - Model with Multi-modal Adaptive Fusion (MAF) module
3. **AudioX-MAF-MMDiT** - Model with MAF and MMDiT

### Quick Download

You can download models using the Gradio interface (see below) or manually:

```bash
# Create model directory
mkdir -p model

# Download AudioX
wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt -O model/AudioX/model.ckpt
wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json -O model/AudioX/config.json

# Download AudioX-MAF
wget https://huggingface.co/HKUSTAudio/AudioX-MAF/resolve/main/model.ckpt -O model/AudioX-MAF/model.ckpt
wget https://huggingface.co/HKUSTAudio/AudioX-MAF/resolve/main/config.json -O model/AudioX-MAF/config.json

# Download AudioX-MAF-MMDiT
wget https://huggingface.co/HKUSTAudio/AudioX-MAF-MMDiT/resolve/main/model.ckpt -O model/AudioX-MAF-MMDiT/model.ckpt
wget https://huggingface.co/HKUSTAudio/AudioX-MAF-MMDiT/resolve/main/config.json -O model/AudioX-MAF-MMDiT/config.json


# Download synchformer
wget https://huggingface.co/HKUSTAudio/AudioX-MAF/resolve/main/synchformer_state_dict.pth -O model/synchformer_state_dict.pth

# Download VAE
wget https://huggingface.co/HKUSTAudio/AudioX-MAF-MMDiT/resolve/main/VAE.ckpt -O model/VAE.ckpt


```

---

## 🤗 Gradio Demo

### Quick Start (Recommended)

The easiest way to launch the demo is using the `--model` argument, which automatically downloads and loads the specified model:

```bash
# Use AudioX model (automatically downloads if not present)
python3 run_gradio.py --model "AudioX" --share

# Use AudioX-MAF model
python3 run_gradio.py --model "AudioX-MAF" --share

# Use AudioX-MAF-MMDiT model
python3 run_gradio.py --model "AudioX-MAF-MMDiT" --share
```

Available model names:
- `"AudioX"` - Base AudioX model
- `"AudioX-MAF"` - AudioX with MAF mechanism
- `"AudioX-MAF-MMDiT"` - AudioX with MAF and MMDiT

### Custom Model Configuration

If you have custom model files, you can specify them directly:

```bash
python3 run_gradio.py \
    --model-config model/config.json \
    --ckpt-path model/model.ckpt \
    --share
```

### Command Line Arguments

```bash
python3 run_gradio.py [OPTIONS]

Options:
  --model MODEL_NAME          Predefined model name (AudioX, AudioX-MAF, AudioX-MAF-MMDiT)
  --model-config PATH         Path to custom model config.json
  --ckpt-path PATH            Path to custom model checkpoint (.ckpt)
  --share                     Create a public Gradio share link
  --server-name ADDRESS       Server address (default: 127.0.0.1)
  --server-port PORT          Server port (default: 7860)
```

**Note:** If `--model` is specified, `--model-config` and `--ckpt-path` will be ignored.

---

## 🎯 Usage Examples

### Supported Tasks

AudioX supports various generation tasks with different input combinations:

| Task                 | `video_path`       | `text_prompt`                                 | `audio_path` |
|:---------------------|:-------------------|:----------------------------------------------|:-------------|
| Text-to-Audio (T2A)  | `None`             | `"Typing on a keyboard"`                      | `None`       |
| Text-to-Music (T2M)  | `None`             | `"A music with piano and violin"`             | `None`       |
| Video-to-Audio (V2A) | `"video_path.mp4"` | `"Generate general audio for the video"`      | `None`       |
| Video-to-Music (V2M) | `"video_path.mp4"` | `"Generate music for the video"`              | `None`       |
| TV-to-Audio (TV2A)   | `"video_path.mp4"` | `"Ocean waves crashing with people laughing"` | `None`       |
| TV-to-Music (TV2M)   | `"video_path.mp4"` | `"Generate music with piano instrument"`      | `None`       |


---

## 🖥️ Script Inference

For programmatic usage, you can use the Python API:

```python
import torch
import torchaudio
from einops import rearrange
from audiox import get_pretrained_model
from audiox.inference.generation import generate_diffusion_cond
from audiox.data.utils import read_video, merge_video_audio, load_and_process_audio, encode_video_with_synchformer
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained model
# Choose one: "HKUSTAudio/AudioX", "HKUSTAudio/AudioX-MAF", or "HKUSTAudio/AudioX-MAF-MMDiT"
model_name = "HKUSTAudio/AudioX-MAF"
model, model_config = get_pretrained_model(model_name)
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
target_fps = model_config["video_fps"]
seconds_start = 0
seconds_total = 10

model = model.to(device)

# Example: Video-to-Music generation
video_path = "example/V2M_sample-1.mp4"
text_prompt = "Generate music for the video" 
audio_path = None

# Prepare inputs
video_tensor = read_video(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=target_fps)
if audio_path:
    audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)
else:
    # Use zero tensor when no audio is provided (following gradio implementation)
    audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))

# For AudioX-MAF and AudioX-MAF-MMDiT: encode video with synchformer
video_sync_frames = None
if "MAF" in model_name:
    video_sync_frames = encode_video_with_synchformer(
        video_path, model_name, seconds_start, seconds_total, device
    )

# Create conditioning (always include audio_prompt, using zero tensor if no audio)
conditioning = [{
    "video_prompt": {"video_tensors": video_tensor.unsqueeze(0), "video_sync_frames": video_sync_frames},        
    "text_prompt": text_prompt,
    "audio_prompt": audio_tensor.unsqueeze(0),
    "seconds_start": seconds_start,
    "seconds_total": seconds_total
}]
    
# Generate audio
output = generate_diffusion_cond(
    model,
    steps=250,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Post-process audio
output = rearrange(output, "b d n -> d (b n)")
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)

# Merge with video (optional)
if video_path is not None and os.path.exists(video_path):
    merge_video_audio(video_path, "output.wav", "output.mp4", seconds_start, seconds_total)
```



---

## 🚀 Citation

If you find our work useful, please consider citing:

```bibtex
@article{tian2025audiox,
  title={AudioX: Diffusion Transformer for Anything-to-Audio Generation},
  author={Tian, Zeyue and Jin, Yizhu and Liu, Zhaoyang and Yuan, Ruibin and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2503.10522},
  year={2025}
}

@inproceedings{tian2025vidmuse,
  title={Vidmuse: A simple video-to-music generation framework with long-short-term modeling},
  author={Tian, Zeyue and Liu, Zhaoyang and Yuan, Ruibin and Pan, Jiahao and Liu, Qifeng and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18782--18793},
  year={2025}
}
```

---

## 📭 Contact

If you have any comments or questions, feel free to contact:
- **Zeyue Tian**: ztianad@connect.ust.hk

---

## 📄 License

Please follow [CC-BY-NC](./LICENSE).

**Note:** The models are watermarked and are strictly for non-commercial use only.

---

## 🙏 Acknowledgments

We thank [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools), [VidMuse](https://github.com/ZeyueT/VidMuse), and [MMAudio](https://github.com/hkchengrex/MMAudio) for their valuable contributions.
