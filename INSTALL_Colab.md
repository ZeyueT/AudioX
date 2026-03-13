# 🚀 AudioX Google Colab Installation Guide

This guide provides step-by-step instructions for setting up AudioX in Google Colab with Gradio live integration.

## 📋 Prerequisites

- Google account with access to Google Colab
- Stable internet connection
- Recommended: Colab Pro for better GPU access and longer runtimes

## 🎯 Quick Start (Recommended)

### Method 1: One-Click Setup

1. **Open the Colab Notebook**
   - Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wamp1re-Ai/AudioX/blob/main/AudioX_Colab.ipynb)

2. **Enable GPU Runtime**
   - Go to `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Choose `T4`, `V100`, or `A100` if available
   - Click `Save`

3. **Run Setup Cells**
   - Run the "Automated Setup" cell (Option A)
   - Wait for installation to complete (~5-10 minutes)
   - Skip to "Launch Gradio Interface" section

4. **Launch Demo**
   - Run the Gradio interface cell
   - Wait for the public URL to appear
   - Click the URL to access the demo

## 🔧 Manual Setup (Alternative)

If the automated setup fails, use this manual approach:

### Step 1: Environment Setup

```python
# Check Colab environment
try:
    import google.colab
    print("✅ Running in Google Colab")
except ImportError:
    print("❌ Not in Colab")

# Enable GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Step 2: Install System Dependencies

```bash
# Update system packages
!apt-get update -qq

# Install required system packages
!apt-get install -y ffmpeg libsndfile1 git-lfs

# Verify installations
!ffmpeg -version | head -1
!which git-lfs
```

### Step 3: Clone Repository

```bash
# Clone the AudioX repository
!git clone https://github.com/Wamp1re-Ai/AudioX.git
%cd AudioX

# Verify repository structure
!ls -la
```

### Step 4: Install Python Dependencies

```bash
# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install AudioX package
!pip install -e .

# Install additional requirements
!pip install gradio==4.44.1
```

### Step 5: Download Models

```python
import os
import urllib.request

# Create model directory
os.makedirs('model', exist_ok=True)

# Download model files
model_files = {
    'model.ckpt': 'https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt',
    'config.json': 'https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json'
}

for filename, url in model_files.items():
    if not os.path.exists(f'model/{filename}'):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, f'model/{filename}')
        print(f"✅ {filename} downloaded")
    else:
        print(f"✅ {filename} already exists")
```

### Step 6: Launch Interface

```python
from stable_audio_tools.interface.gradio import create_ui

# Create and launch interface
interface = create_ui(
    model_config_path='./model/config.json',
    ckpt_path='./model/model.ckpt'
)

interface.queue(max_size=10)
interface.launch(share=True, debug=False)
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. "CUDA out of memory"
```python
# Clear GPU memory
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

# Reduce model precision
interface = create_ui(model_half=True)
```

#### 2. "Model files not found"
```bash
# Check if files exist
!ls -la model/

# Re-download if missing
!wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt -O model/model.ckpt
!wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json -O model/config.json
```

#### 3. "Import errors"
```bash
# Reinstall dependencies
!pip install --force-reinstall -e .
!pip install --upgrade gradio
```

#### 4. "Runtime disconnected"
```python
# Reconnect and run this cell to restore session
%cd AudioX
import torch
print(f"GPU Available: {torch.cuda.is_available()}")

# Re-launch interface
from stable_audio_tools.interface.gradio import create_ui
interface = create_ui()
interface.launch(share=True)
```

### Performance Optimization

#### For Free Colab Users
```python
# Use smaller models and reduced settings
interface = create_ui(model_half=True)

# Reduce generation steps
# In the interface, use:
# - Steps: 50-100 (instead of 150-250)
# - CFG Scale: 6.0-7.0
# - Shorter audio duration
```

#### For Colab Pro Users
```python
# Use full precision for better quality
interface = create_ui(model_half=False)

# Enable high-quality settings:
# - Steps: 150-250
# - CFG Scale: 7.0-8.0
# - Longer audio duration
```

## 📊 System Requirements

### Minimum Requirements
- **GPU**: T4 (15GB VRAM)
- **RAM**: 12GB
- **Storage**: 5GB free space
- **Runtime**: Standard GPU runtime

### Recommended Requirements
- **GPU**: V100 or A100
- **RAM**: High-RAM runtime (25GB+)
- **Storage**: 10GB free space
- **Runtime**: Colab Pro with premium GPUs

## 🎯 Usage Tips

### Best Practices
1. **Start Simple**: Begin with default settings
2. **Monitor Memory**: Clear memory between generations
3. **Save Work**: Download generated audio files
4. **Use Examples**: Try the provided example prompts
5. **Experiment**: Adjust parameters gradually

### Prompt Writing Tips
- **Be Specific**: "Jazz piano in a smoky bar" vs "music"
- **Include Context**: "Footsteps on gravel at night"
- **Mention Style**: "Classical orchestra", "electronic music"
- **Add Atmosphere**: "Peaceful", "energetic", "mysterious"

## 🔗 Additional Resources

- **Original Repository**: [AudioX GitHub](https://github.com/ZeyueT/AudioX)
- **Paper**: [AudioX: Diffusion Transformer](https://arxiv.org/abs/2503.10522)
- **Hugging Face**: [AudioX Model](https://huggingface.co/HKUSTAudio/AudioX)
- **Gradio Documentation**: [Gradio Docs](https://gradio.app/docs/)

## 🆘 Getting Help

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Review the error messages carefully**
3. **Try restarting the runtime**
4. **Use the manual setup method**
5. **Check the original repository for updates**

## 📝 Notes

- **Session Limits**: Free Colab sessions have time limits
- **File Persistence**: Files are deleted when session ends
- **GPU Availability**: GPUs may not always be available
- **Network Speed**: Download times depend on connection
- **Model Size**: Model files are ~2GB total

---

**Happy audio generation! 🎵**
