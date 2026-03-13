"""
AudioX Colab Setup Utilities
Provides helper functions for setting up AudioX in Google Colab environment.
"""

import os
import sys
import torch
import platform
import subprocess
import urllib.request
from pathlib import Path
import json
import gc
from typing import Optional, Dict, Any

def check_colab_environment():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_colab_environment():
    """Setup optimal environment for AudioX in Colab"""
    print("🔧 Setting up Colab environment...")
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TMPDIR'] = './tmp'
    
    # Create necessary directories
    directories = ['./tmp', './demo_result', './model']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("✅ Environment setup complete!")

def install_system_dependencies():
    """Install system dependencies for Colab"""
    if not check_colab_environment():
        print("ℹ️  Not in Colab, skipping system dependency installation")
        return

    print("📦 Installing system dependencies...")

    try:
        # Update package list
        subprocess.run(['apt-get', 'update', '-qq'], check=True)

        # Install required packages
        packages = ['ffmpeg', 'libsndfile1', 'git-lfs']
        for package in packages:
            print(f"Installing {package}...")
            subprocess.run(['apt-get', 'install', '-y', package], check=True)

        print("✅ System dependencies installed!")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Some system dependencies may not have installed correctly: {e}")
        print("Continuing with setup...")

def check_gpu_setup():
    """Check and configure GPU setup"""
    print("🎮 Checking GPU setup...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU Available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        return device, gpu_memory
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, using CPU")
        return device, 0

def install_python_dependencies():
    """Install Python dependencies with Colab compatibility"""
    print("🐍 Installing Python dependencies...")

    try:
        # Install PyTorch with CUDA support (if not already installed)
        print("Installing PyTorch with CUDA support...")
        subprocess.run([
            'pip', 'install', '-q',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu118'
        ], check=True)

        # Install AudioX-specific dependencies from requirements
        print("Installing AudioX dependencies...")
        subprocess.run([
            'pip', 'install', '-q', '-r', 'requirements_colab.txt'
        ], check=True)

        # Install AudioX package in development mode
        print("Installing AudioX package...")
        subprocess.run(['pip', 'install', '-q', '-e', '.'], check=True)

        print("✅ Python dependencies installed!")

    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Some dependencies may have conflicts: {e}")
        print("Trying alternative installation method...")

        # Fallback: Install only essential packages
        essential_packages = [
            'gradio>=4.40.0',
            'aeiou',
            'einops',
            'safetensors',
            'transformers',
            'huggingface_hub'
        ]

        for package in essential_packages:
            try:
                subprocess.run(['pip', 'install', '-q', package], check=True)
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Failed to install {package}")

        # Try to install AudioX package
        try:
            subprocess.run(['pip', 'install', '-q', '-e', '.', '--no-deps'], check=True)
            print("✅ AudioX package installed (no deps)")
        except subprocess.CalledProcessError:
            print("⚠️  AudioX package installation failed")

def download_model_files(force_download: bool = False):
    """Download AudioX model files from Hugging Face"""
    print("📥 Downloading AudioX model files...")
    
    model_files = {
        'model.ckpt': 'https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt',
        'config.json': 'https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json'
    }
    
    def download_with_progress(url: str, filename: str):
        """Download file with progress bar"""
        filepath = f'model/{filename}'
        
        if os.path.exists(filepath) and not force_download:
            print(f"✅ {filename} already exists, skipping download")
            return
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                downloaded_mb = downloaded // 1024 // 1024
                total_mb = total_size // 1024 // 1024
                print(f"\r📥 {filename}: {percent:.1f}% ({downloaded_mb}MB/{total_mb}MB)", end="")
        
        print(f"📥 Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath, progress_hook)
            print(f"\n✅ {filename} downloaded successfully!")
        except Exception as e:
            print(f"\n❌ Error downloading {filename}: {e}")
            raise
    
    # Download all model files
    for filename, url in model_files.items():
        download_with_progress(url, filename)
    
    print("🎉 All model files downloaded!")

def verify_installation():
    """Verify that AudioX is properly installed and configured"""
    print("🔍 Verifying installation...")
    
    # Check if model files exist
    model_files = ['model/model.ckpt', 'model/config.json']
    for file_path in model_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"✅ {file_path} exists ({size_mb:.1f}MB)")
        else:
            print(f"❌ {file_path} missing!")
            return False
    
    # Check if AudioX can be imported
    try:
        from stable_audio_tools.interface.gradio import create_ui
        print("✅ AudioX modules can be imported")
    except ImportError as e:
        print(f"❌ Cannot import AudioX modules: {e}")
        return False
    
    # Check PyTorch and device
    device, gpu_memory = check_gpu_setup()
    
    print("✅ Installation verification complete!")
    return True

def optimize_for_colab():
    """Apply Colab-specific optimizations"""
    print("⚡ Applying Colab optimizations...")
    
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Set optimal number of workers for DataLoader
    os.environ['NUM_WORKERS'] = '2'  # Colab has limited CPU cores
    
    # Disable wandb offline mode to prevent issues
    os.environ['WANDB_MODE'] = 'disabled'
    
    print("✅ Optimizations applied!")

def full_colab_setup(force_download: bool = False):
    """Complete setup process for AudioX in Colab"""
    print("🚀 Starting full AudioX Colab setup...")

    if not check_colab_environment():
        print("⚠️  Not running in Colab. Some features may not work as expected.")

    try:
        # Step 1: Environment setup
        setup_colab_environment()

        # Step 2: System dependencies (Colab only)
        if check_colab_environment():
            install_system_dependencies()

        # Step 3: Python dependencies
        install_python_dependencies()

        # Step 4: Check GPU
        device, gpu_memory = check_gpu_setup()

        # Step 5: Download models
        download_model_files(force_download)

        # Step 6: Apply optimizations
        optimize_for_colab()

        # Step 7: Verify installation
        if verify_installation():
            print("🎉 AudioX Colab setup completed successfully!")
            print(f"🔧 Device: {device}")
            if gpu_memory > 0:
                print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
            print("🎛️ Ready to launch Gradio interface!")
            return True
        else:
            print("❌ Setup verification failed!")
            return False

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def clear_memory():
    """Clear GPU and system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared")
    gc.collect()
    print("🧹 System memory cleared")

def get_system_info():
    """Get system information for debugging"""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "in_colab": check_colab_environment()
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return info

if __name__ == "__main__":
    # Run full setup when executed directly
    success = full_colab_setup()
    if success:
        print("\n" + "="*50)
        print("🎉 Setup complete! You can now run the Gradio interface.")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ Setup failed. Please check the error messages above.")
        print("="*50)
