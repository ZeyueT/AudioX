# 🎧 AudioX: Diffusion Transformer for Anything-to-Audio Generation

[![arXiv](https://img.shields.io/badge/arXiv-2503.10522-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2503.10522)
[![Project Page](https://img.shields.io/badge/GitHub.io-Project-blue?logo=Github&style=flat-square)](https://zeyuet.github.io/AudioX/)
[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/HKUSTAudio/AudioX)
[![🤗 Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Zeyue7/AudioX)

---

**This is the official repository for "[AudioX: Diffusion Transformer for Anything-to-Audio Generation](https://arxiv.org/pdf/2503.10522)".**


## 📺 Demo Video

https://github.com/user-attachments/assets/0d8dd927-ff0f-4b35-ab1f-b3c3915017be

---


## ✨ Abstract

Audio and music generation have emerged as crucial tasks in many applications, yet existing approaches face significant limitations: they operate in isolation without unified capabilities across modalities, suffer from scarce high-quality, multi-modal training data, and struggle to effectively integrate diverse inputs. In this work, we propose AudioX, a unified Diffusion Transformer model for Anything-to-Audio and Music Generation. Unlike previous domain-specific models, AudioX can generate both general audio and music with high quality, while offering flexible natural language control and seamless processing of various modalities including text, video, image, music, and audio. Its key innovation is a multi-modal masked training strategy that masks inputs across modalities and forces the model to learn from masked inputs, yielding robust and unified cross-modal representations. To address data scarcity, we curate two comprehensive datasets: vggsound-caps with 190K audio captions based on the VGGSound dataset, and V2M-caps with 6 million music captions derived from the V2M dataset. Extensive experiments demonstrate that AudioX not only matches or outperforms state-of-the-art specialized models, but also offers remarkable versatility in handling diverse input modalities and generation tasks within a unified architecture.


## ✨ Teaser

<p align="center">
  <img src="https://github.com/user-attachments/assets/ea723225-f9c8-4ca2-8837-2c2c08189bdd" alt="method">
</p>
<p style="text-align: left;">(a) Overview of AudioX, illustrating its capabilities across various tasks. (b) Radar chart comparing the performance of different methods across multiple benchmarks. AudioX demonstrates superior Inception Scores (IS) across a diverse set of datasets in audio and music generation tasks.</p>


## ✨ Method

<p align="center">
  <img src="https://github.com/user-attachments/assets/94ea3df0-8c66-4259-b681-791ee41bada8" alt="method">
</p>
<p align="center">Overview of the AudioX Framework.</p>



## Code


### 🛠️ Environment Setup

```bash
git clone https://github.com/ZeyueT/AudioX.git
cd AudioX
conda create -n AudioX python=3.8.20
conda activate AudioX
pip install git+https://github.com/ZeyueT/AudioX.git
conda install -c conda-forge ffmpeg libsndfile

```

## 🪄 Pretrained Checkpoints

Download the pretrained model from 🤗 [AudioX on Hugging Face](https://huggingface.co/HKUSTAudio/AudioX):

```bash
mkdir -p model
wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt -O model/model.ckpt
wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json -O model/config.json
```

### 🤗 Gradio Demo

To launch the Gradio demo locally, run:

```bash
python3 run_gradio.py \
    --model-config model/config.json \
    --share
```

### 🎯 VRAM Optimization for Low-Memory GPUs

AudioX includes automatic VRAM optimization for systems with limited GPU memory (e.g., 8GB VRAM). The system uses:

- **FP16 (Half Precision)**: Reduces VRAM usage by ~40% with minimal quality impact
- **CPU Offloading**: Automatically distributes model layers between GPU and CPU using Hugging Face Accelerate
- **Automatic Memory Detection**: Detects available VRAM and RAM to optimize allocation
- **Balanced Device Mapping**: Intelligently splits the model across GPU and CPU for optimal performance

By default, the system uses **90% of available VRAM** and **80% of available RAM** for CPU offloading.

#### Memory Detection Output
When starting the demo, you'll see detailed memory information:
```
==================================================
VRAM Optimization Settings:
  - FP16 (half precision): Enabled by default
  - CPU offloading: Enabled by default

System Memory Detection:
  - VRAM: 7.6GB total, 7.2GB available
  - RAM: 61.3GB total, 41.9GB available

Planned Memory Allocation:
  - Using: 6.5GB VRAM (90% of available)
  - Using: 33.5GB RAM (80% of available)
==================================================
```

#### Disabling Optimization Features
If you have sufficient VRAM or want to disable optimizations:

```bash
# Disable FP16 (uses more VRAM but may improve quality)
python3 run_gradio.py --no-half

# Disable CPU offloading (keeps entire model on GPU)
python3 run_gradio.py --no-cpu-offload

# Disable both optimizations
python3 run_gradio.py --no-half --no-cpu-offload
```

#### Manual Configuration
For advanced users, you can manually set VRAM limits:
```bash
# Limit VRAM usage to specific GB (default: auto-detected)
python3 run_gradio.py --max-vram 8
```

#### Requirements
The VRAM optimization requires `accelerate` and `psutil`, which are automatically installed when installing AudioX via `pip`. If you installed manually, install them with:
```bash
pip install accelerate psutil
```

### 🎯 Prompt Configuration Examples

| Task                 | `video_path`       | `text_prompt`                                 | `audio_path` |
|:---------------------|:-------------------|:----------------------------------------------|:-------------|
| Text-to-Audio (T2A)  | `None`             | `"Typing on a keyboard"`                      | `None`       |
| Text-to-Music (T2M)  | `None`             | `"A music with piano and violin"`             | `None`       |
| Video-to-Audio (V2A) | `"video_path.mp4"` | `"Generate general audio for the video"`      | `None`       |
| Video-to-Music (V2M) | `"video_path.mp4"` | `"Generate music for the video"`              | `None`       |
| TV-to-Audio (TV2A)   | `"video_path.mp4"` | `"Ocean waves crashing with people laughing"` | `None`       |
| TV-to-Music (TV2M)   | `"video_path.mp4"` | `"Generate music with piano instrument"`      | `None`       |

### 🖥️ Script Inference

```python
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import read_video, merge_video_audio
from stable_audio_tools.data.utils import load_and_process_audio
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("HKUSTAudio/AudioX")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
target_fps = model_config["video_fps"]
seconds_start = 0
seconds_total = 10

model = model.to(device)

# for video-to-music generation
video_path = "example/V2M_sample-1.mp4"
text_prompt = "Generate music for the video" 
audio_path = None

video_tensor = read_video(video_path, seek_time=0, duration=seconds_total, target_fps=target_fps)
audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)

conditioning = [{
    "video_prompt": [video_tensor.unsqueeze(0)],        
    "text_prompt": text_prompt,
    "audio_prompt": audio_tensor.unsqueeze(0),
    "seconds_start": seconds_start,
    "seconds_total": seconds_total
}]
    
# Generate stereo audio
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

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)

if video_path is not None and os.path.exists(video_path):
    merge_video_audio(video_path, "output.wav", "output.mp4", 0, seconds_total)

```


## 🚀 Citation

If you find our work useful, please consider citing:

```
@article{tian2025audiox,
  title={AudioX: Diffusion Transformer for Anything-to-Audio Generation},
  author={Tian, Zeyue and Jin, Yizhu and Liu, Zhaoyang and Yuan, Ruibin and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2503.10522},
  year={2025}
}
```

## 📭 Contact

If you have any comments or questions, feel free to contact Zeyue Tian(ztianad@connect.ust.hk).

## License

Please follow [CC-BY-NC](./LICENSE).
