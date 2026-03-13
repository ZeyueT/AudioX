# 🔧 AudioX Colab Troubleshooting Guide

This guide helps resolve common issues when running AudioX in Google Colab.

## 🚨 Common Issues and Solutions

### 1. ModuleNotFoundError: No module named 'colab_gradio_interface'

**Symptoms:**
```
ModuleNotFoundError: No module named 'colab_gradio_interface'
```

**Solution:**
- This is normal! The notebook will automatically fall back to the standard interface
- You'll see: "⚠️ Colab interface not found, using standard interface..."
- The standard interface works just as well

**If you want to fix it:**
1. Make sure you're in the AudioX directory: `%cd AudioX`
2. The notebook should handle this automatically now

### 2. Dependency Conflict Warnings

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account...
google-colab 1.0.0 requires pandas==2.2.2, but you have pandas 2.0.2...
```

**Solution:**
- **These warnings are NORMAL in Colab** ✅
- They don't prevent AudioX from working
- Colab has many pre-installed packages with specific versions
- AudioX is designed to work with these conflicts

**What NOT to do:**
- Don't try to "fix" these by upgrading/downgrading packages
- Don't restart and reinstall - it will create more conflicts

### 3. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Clear GPU memory:**
   ```python
   import torch
   import gc
   torch.cuda.empty_cache()
   gc.collect()
   ```

2. **Use model half precision:**
   - In the interface, the model will automatically use half precision if needed

3. **Reduce generation settings:**
   - Steps: Use 50-100 instead of 150-250
   - Batch size: Keep at 1
   - Audio length: Generate shorter clips

4. **Restart runtime:**
   - Runtime → Restart runtime
   - Re-run all cells

### 4. Model Files Not Found

**Symptoms:**
```
FileNotFoundError: model/model.ckpt not found
```

**Solutions:**
1. **Check if model directory exists:**
   ```python
   import os
   print(os.listdir('.'))
   print(os.path.exists('model'))
   if os.path.exists('model'):
       print(os.listdir('model'))
   ```

2. **Re-download models:**
   ```python
   !mkdir -p model
   !wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt -O model/model.ckpt
   !wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json -O model/config.json
   ```

3. **Check file sizes:**
   ```python
   import os
   if os.path.exists('model/model.ckpt'):
       size = os.path.getsize('model/model.ckpt') / 1024 / 1024
       print(f"model.ckpt size: {size:.1f}MB")
       if size < 100:  # Should be ~2GB
           print("File seems incomplete, re-downloading...")
   ```

### 5. Interface Won't Load

**Symptoms:**
- Gradio interface doesn't appear
- No public URL generated

**Solutions:**
1. **Check for errors in the cell output**
2. **Try restarting the interface:**
   ```python
   # Stop any running interfaces
   import gradio as gr
   gr.close_all()
   
   # Re-run the Gradio launch cell
   ```

3. **Use alternative launch:**
   ```python
   from stable_audio_tools.interface.gradio import create_ui
   interface = create_ui()
   interface.launch(share=True, debug=True)
   ```

### 6. Runtime Disconnected

**Symptoms:**
- "Runtime disconnected" message
- Session lost

**Solutions:**
1. **Reconnect and restore:**
   ```python
   # Check current directory
   import os
   print(f"Current directory: {os.getcwd()}")
   
   # If not in AudioX directory
   if not os.path.exists('stable_audio_tools'):
       %cd AudioX
   
   # Check GPU
   import torch
   print(f"GPU available: {torch.cuda.is_available()}")
   ```

2. **Quick restart:**
   - Re-run the Gradio launch cell
   - Models should still be cached

### 7. Slow Generation

**Symptoms:**
- Audio generation takes very long
- Interface seems frozen

**Solutions:**
1. **Check GPU usage:**
   ```python
   !nvidia-smi
   ```

2. **Reduce settings:**
   - Steps: 50-100 (instead of 150-250)
   - CFG Scale: 6-7 (instead of 8-10)
   - Audio length: 5-10 seconds

3. **Free Colab limitations:**
   - Free tier has slower GPUs
   - Consider Colab Pro for faster generation

## 🛠️ Diagnostic Commands

### Check System Status
```python
import torch
import os
import sys
from pathlib import Path

print("=== System Diagnostics ===")
print(f"Current directory: {Path.cwd()}")
print(f"Python path: {sys.path[:3]}...")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

print("\n=== File Check ===")
print(f"AudioX directory exists: {Path('stable_audio_tools').exists()}")
print(f"Model directory exists: {Path('model').exists()}")
if Path('model').exists():
    for file in ['model.ckpt', 'config.json']:
        path = Path(f'model/{file}')
        if path.exists():
            size = path.stat().st_size / 1024 / 1024
            print(f"{file}: {size:.1f}MB")
        else:
            print(f"{file}: Missing")
```

### Test Import
```python
print("=== Import Test ===")
try:
    from stable_audio_tools.interface.gradio import create_ui
    print("✅ AudioX modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

try:
    from colab_gradio_interface import launch_colab_demo
    print("✅ Colab interface available")
except ImportError:
    print("⚠️ Colab interface not found (will use standard)")
```

## 🆘 When All Else Fails

### Nuclear Option: Complete Reset
```python
# 1. Clear everything
import shutil
if os.path.exists('AudioX'):
    shutil.rmtree('AudioX')

# 2. Re-clone and setup
!git clone https://github.com/Wamp1re-Ai/AudioX.git
%cd AudioX

# 3. Minimal installation
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q gradio aeiou einops safetensors transformers huggingface_hub
!pip install -q -e . --no-deps

# 4. Download models
!mkdir -p model
!wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt -O model/model.ckpt
!wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json -O model/config.json
```

### Alternative: Use Original Repository
If the Colab integration has issues, you can use the original AudioX:
```python
!git clone https://github.com/ZeyueT/AudioX.git
%cd AudioX
!pip install -e .
# Download models and run standard interface
```

## 📞 Getting Help

If you're still having issues:

1. **Check the error message carefully** - most issues have specific solutions above
2. **Try the diagnostic commands** to understand what's wrong
3. **Use the troubleshooting cell** in the notebook
4. **Report issues** with the full error message and diagnostic output

## 💡 Prevention Tips

1. **Always use GPU runtime** (Runtime → Change runtime type → GPU)
2. **Don't modify system packages** unless necessary
3. **Use the provided notebook cells** in order
4. **Save your work** - Colab sessions are temporary
5. **Monitor GPU usage** - free tier has limits

---

**Remember:** Dependency warnings are normal in Colab and don't prevent AudioX from working! 🎵
