from stable_audio_tools import get_pretrained_model
from stable_audio_tools.interface.gradio import create_ui, get_system_memory_info
import json
import os

import torch

# Set PyTorch CUDA memory allocation configuration for better memory management
# This helps prevent OOM errors by limiting memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

def main(args):
    torch.manual_seed(42)
    
    # Print memory optimization status
    print("=" * 50)
    print("VRAM Optimization Settings:")
    print(f"  - FP16 (half precision): {'Enabled by default' if not args.no_half else 'Disabled'}")
    print(f"  - CPU offloading: {'Enabled by default' if not args.no_cpu_offload else 'Disabled'}")
    
    # Show detected system memory
    mem_info = get_system_memory_info()
    print(f"\nSystem Memory Detection:")
    print(f"  - VRAM: {mem_info['vram_total_gb']:.1f}GB total, {mem_info['vram_available_gb']:.1f}GB available")
    print(f"  - RAM: {mem_info['ram_total_gb']:.1f}GB total, {mem_info['ram_available_gb']:.1f}GB available")
    
    # Calculate and show planned allocation
    vram_target_gb = mem_info["vram_available_gb"] * 0.9
    ram_target_gb = mem_info["ram_available_gb"] * 0.8
    print(f"\nPlanned Memory Allocation:")
    print(f"  - Using: {vram_target_gb:.1f}GB VRAM (90% of available)")
    print(f"  - Using: {ram_target_gb:.1f}GB RAM (80% of available)")
    print("=" * 50)

    interface = create_ui(
        model_config_path = args.model_config,
        ckpt_path=args.ckpt_path,
        pretrained_name=args.pretrained_name,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        model_half=args.model_half
    )
    interface.queue()
    interface.launch(share=args.share, auth=(args.username, args.password) if args.username is not None else None)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run gradio interface')
    parser.add_argument('--pretrained-name', type=str, help='Name of pretrained model', required=False)
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint', required=False)
    parser.add_argument('--share', action='store_true', help='Create a publicly shareable link', required=False)
    parser.add_argument('--username', type=str, help='Gradio username', required=False)
    parser.add_argument('--password', type=str, help='Gradio password', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)
    
    # VRAM optimization arguments
    parser.add_argument('--no-half', action='store_true', help='Disable FP16 (half precision) - uses more VRAM', required=False)
    parser.add_argument('--no-cpu-offload', action='store_true', help='Disable CPU offloading - uses more VRAM', required=False)
    parser.add_argument('--max-vram', type=int, default=6, help='Maximum VRAM to use in GB (default: 6GB for 8GB GPUs)', required=False)
    
    args = parser.parse_args()
    main(args)