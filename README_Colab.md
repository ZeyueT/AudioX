# 🎧 AudioX Google Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wamp1re-Ai/AudioX/blob/main/AudioX_Colab.ipynb)

This repository provides a Google Colab version of **AudioX: Diffusion Transformer for Anything-to-Audio Generation** with an optimized Gradio interface for easy use in cloud environments.

## 🚀 Quick Start

### Option 1: One-Click Colab (Recommended)
1. Click the "Open in Colab" badge above
2. Run all cells in order
3. Wait for the Gradio interface to load
4. Use the public URL to access the demo

### Option 2: Manual Setup
1. Open Google Colab
2. Upload `AudioX_Colab.ipynb` to your Colab environment
3. Run all cells sequentially
4. Access the demo via the generated public URL

## 📋 Features

### 🎯 Supported Tasks
- **📝 Text-to-Audio**: Generate audio from text descriptions
- **🎶 Text-to-Music**: Create music from text prompts
- **🎬 Video-to-Audio**: Generate audio for video content
- **🎵 Video-to-Music**: Create background music for videos

### ⚡ Colab Optimizations
- **Memory Management**: Automatic GPU memory clearing
- **Model Caching**: Efficient model loading and caching
- **Queue Management**: Optimized for Colab's resource limits
- **Error Handling**: Robust error recovery and reporting
- **Mobile-Friendly**: Responsive interface design

## 🛠️ Files Overview

| File | Description |
|------|-------------|
| `AudioX_Colab.ipynb` | Main Colab notebook with complete setup |
| `colab_setup.py` | Setup utilities and environment configuration |
| `colab_gradio_interface.py` | Colab-optimized Gradio interface |
| `launch_colab.py` | Simple launcher script |
| `requirements_colab.txt` | Colab-specific dependencies |

## 📖 Usage Guide

### 🎯 Text-to-Audio Generation
```
Prompt: "Ocean waves crashing on a rocky shore"
Steps: 100-150
CFG Scale: 7.0
```

### 🎶 Music Generation
```
Prompt: "Upbeat jazz piano music in a cozy cafe"
Steps: 150-200
CFG Scale: 8.0
```

### 🎬 Video-to-Audio
1. Upload your video file (MP4, AVI, MOV)
2. Use prompt: "Generate audio for this video"
3. Adjust settings as needed

## ⚙️ Settings Guide

### Basic Settings
- **Steps**: Number of diffusion steps (10-250)
  - Lower: Faster generation, lower quality
  - Higher: Slower generation, better quality
  - Recommended: 100-150

- **CFG Scale**: Classifier-free guidance scale (1.0-15.0)
  - Lower: More creative, less prompt adherence
  - Higher: More prompt adherence, less creative
  - Recommended: 6.0-8.0

- **Seed**: Random seed for reproducible results
  - -1: Random seed each time
  - Fixed number: Reproducible results

### Advanced Settings
- **Sampler Type**: Different sampling algorithms
  - `dpmpp-3m-sde`: Balanced quality and speed (default)
  - `dpmpp-2m-sde`: Faster, slightly lower quality
  - `k-heun`: Higher quality, slower

- **Sigma Min/Max**: Noise schedule parameters
  - Usually best to keep at defaults

## 💡 Tips for Best Results

### 🎯 Prompt Writing
- **Be Specific**: "Jazz piano in a smoky bar" vs "music"
- **Include Context**: "Footsteps on gravel at night"
- **Mention Style**: "Classical orchestra", "electronic dance music"
- **Add Atmosphere**: "Peaceful", "energetic", "mysterious"

### 🎬 Video Processing
- **File Size**: Keep videos under 100MB for best performance
- **Duration**: 10-30 seconds work best
- **Format**: MP4 is most compatible
- **Quality**: Lower resolution videos process faster

### ⚡ Performance Tips
- Start with default settings
- Use lower steps (50-100) for testing
- Clear memory between generations if needed
- Avoid very long prompts (>100 words)

## 🚨 Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Click "Clear Memory" button
- Reduce steps to 50-100
- Try generating shorter audio
- Restart the runtime if needed

**"Model not loading"**
- Check internet connection
- Re-run the model download cell
- Verify model files exist in `/content/AudioX/model/`

**"Generation taking too long"**
- Reduce number of steps
- Use faster sampler (dpmpp-2m-sde)
- Check if using GPU (not CPU)

**"Interface not accessible"**
- Check if the public URL is working
- Try refreshing the Gradio interface
- Re-run the launch cell

### Memory Management
```python
# Clear GPU memory manually
import torch
import gc
torch.cuda.empty_cache()
gc.collect()
```

## 🔧 System Requirements

### Google Colab
- **GPU**: T4, V100, or A100 (recommended)
- **RAM**: 12GB+ (High-RAM runtime recommended)
- **Storage**: 5GB+ free space
- **Runtime**: Python 3.10+

### Local Environment
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Python**: 3.8-3.11
- **CUDA**: 11.8 or 12.1

## 📚 Examples

### Text-to-Audio Examples
```
"Typing on a mechanical keyboard"
"Ocean waves crashing"
"Rain falling on leaves"
"Footsteps in snow"
"Coffee machine brewing"
```

### Music Examples
```
"Peaceful acoustic guitar melody"
"Upbeat electronic dance music"
"Classical piano sonata"
"Jazz saxophone solo"
"Lo-fi hip hop beat"
```

## 🔗 Links

- **Original Repository**: [AudioX GitHub](https://github.com/ZeyueT/AudioX)
- **Paper**: [AudioX: Diffusion Transformer for Anything-to-Audio Generation](https://arxiv.org/abs/2503.10522)
- **Project Page**: [AudioX Project](https://zeyuet.github.io/AudioX/)
- **Hugging Face**: [AudioX Model](https://huggingface.co/HKUSTAudio/AudioX)

## 📄 License

This project follows the same license as the original AudioX repository.

## 🤝 Contributing

Contributions to improve the Colab experience are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test in Google Colab
4. Submit a pull request

## 🙏 Acknowledgments

- Original AudioX team at HKUST
- Google Colab for providing the platform
- Gradio team for the interface framework
- Hugging Face for model hosting
