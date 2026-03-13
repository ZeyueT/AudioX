#!/usr/bin/env python3
"""
AudioX Colab Launcher
Simple script to launch AudioX in Google Colab with optimal settings
"""

import sys
import os
from pathlib import Path

def main():
    """Main launcher function"""
    print("🚀 AudioX Colab Launcher")
    print("=" * 50)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        # Import setup utilities
        from colab_setup import full_colab_setup, check_colab_environment
        
        # Check environment
        if check_colab_environment():
            print("✅ Running in Google Colab")
        else:
            print("⚠️  Not running in Colab - some features may not work")
        
        # Run full setup
        print("\n🔧 Setting up AudioX...")
        setup_success = full_colab_setup()
        
        if not setup_success:
            print("❌ Setup failed! Please check the error messages above.")
            return False
        
        # Launch Gradio interface
        print("\n🎛️ Launching Gradio interface...")
        from colab_gradio_interface import launch_colab_demo
        
        # Launch with default settings
        interface = launch_colab_demo(
            model_config_path="./model/config.json",
            ckpt_path="./model/model.ckpt",
            share=True,
            debug=False
        )
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 AudioX is now running!")
    else:
        print("\n❌ Failed to launch AudioX")
        sys.exit(1)
