#!/usr/bin/env python3
"""
AudioX Demo Test Script
Quick test to verify AudioX installation and generate a sample audio
"""

import os
import sys
import torch
import json
from pathlib import Path

def test_installation():
    """Test if AudioX is properly installed"""
    print("🔍 Testing AudioX installation...")
    
    # Test imports
    try:
        from stable_audio_tools.interface.gradio import create_ui
        from stable_audio_tools.models.factory import create_model_from_config
        print("✅ AudioX modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AudioX modules: {e}")
        return False
    
    # Check model files
    model_files = ['model/model.ckpt', 'model/config.json']
    for file_path in model_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            print(f"✅ {file_path} exists ({size_mb:.1f}MB)")
        else:
            print(f"❌ {file_path} missing!")
            return False
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  No GPU available, will use CPU")
    
    return True

def generate_test_audio():
    """Generate a short test audio sample"""
    print("\n🎵 Generating test audio...")
    
    try:
        from stable_audio_tools.interface.gradio import generate_cond
        
        # Simple test parameters
        test_prompt = "Ocean waves crashing"
        
        print(f"📝 Prompt: {test_prompt}")
        print("⏱️  Generating (this may take a few minutes)...")
        
        # Generate audio with minimal settings for speed
        result = generate_cond(
            prompt=test_prompt,
            steps=50,  # Reduced for faster testing
            cfg_scale=6.0,
            seed=42,
            seconds_total=5  # Short duration for testing
        )
        
        if result:
            video_path, audio_path = result
            print(f"✅ Audio generated successfully!")
            print(f"🎵 Audio saved to: {audio_path}")
            return True
        else:
            print("❌ Audio generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return False

def quick_demo():
    """Run a quick demo to test everything"""
    print("🚀 AudioX Quick Demo")
    print("=" * 50)
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed!")
        return False
    
    print("\n✅ Installation test passed!")
    
    # Ask user if they want to test generation
    try:
        response = input("\n🎵 Test audio generation? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            if generate_test_audio():
                print("\n🎉 Demo completed successfully!")
                return True
            else:
                print("\n❌ Demo failed during audio generation")
                return False
        else:
            print("\n✅ Installation verified. Skipping audio generation test.")
            return True
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return False

def main():
    """Main function"""
    try:
        success = quick_demo()
        if success:
            print("\n" + "="*50)
            print("🎉 AudioX is ready to use!")
            print("🎛️ You can now launch the Gradio interface")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("❌ Demo failed. Please check the setup.")
            print("="*50)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
