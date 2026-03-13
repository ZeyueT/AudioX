"""
Colab-optimized Gradio interface for AudioX
Includes memory management and Colab-specific optimizations
"""

import gc
import platform
import os
import gradio as gr
import json 
import torch
import torchaudio
import warnings
from typing import Optional, Tuple, Any

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import AudioX components
from stable_audio_tools.interface.gradio import (
    load_model, generate_cond, model_configurations,
    current_model, current_model_name, current_sample_rate, current_sample_size
)

def check_colab_environment():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_optimal_device():
    """Get the optimal device for the current environment"""
    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        has_mps = False

    if has_mps:
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def clear_memory():
    """Clear GPU and system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def colab_generate_wrapper(*args, **kwargs):
    """Wrapper for generate_cond with memory management"""
    try:
        # Clear memory before generation
        clear_memory()
        
        # Call the original generation function
        result = generate_cond(*args, **kwargs)
        
        # Clear memory after generation
        clear_memory()
        
        return result
    except Exception as e:
        # Clear memory on error
        clear_memory()
        raise e

def create_colab_ui(
    model_config_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    pretrained_name: Optional[str] = None,
    pretransform_ckpt_path: Optional[str] = None,
    model_half: bool = False
):
    """Create Colab-optimized Gradio interface"""
    
    global model_configurations
    device = get_optimal_device()
    
    print(f"🔧 Using device: {device}")
    
    # Set up model configurations
    model_configurations = {
        "default": {
            "model_config": model_config_path or "./model/config.json",
            "ckpt_path": ckpt_path or "./model/model.ckpt"
        }
    }
    
    # Custom CSS for better mobile experience
    custom_css = """
    .gradio-container { 
        max-width: 1120px; 
        margin: auto; 
    }
    .colab-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .memory-info {
        background: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-family: monospace;
    }
    """
    
    with gr.Blocks(css=custom_css, title="AudioX - Colab Demo") as interface:
        
        # Header
        with gr.Row():
            gr.HTML("""
            <div class="colab-header">
                <h1>🎧 AudioX: Diffusion Transformer for Anything-to-Audio Generation</h1>
                <p>Generate high-quality audio from text descriptions and videos</p>
                <p>
                    <a href="https://arxiv.org/abs/2503.10522" target="_blank">📄 Paper</a> | 
                    <a href="https://zeyuet.github.io/AudioX/" target="_blank">🌐 Project Page</a> | 
                    <a href="https://github.com/ZeyueT/AudioX" target="_blank">💻 GitHub</a>
                </p>
            </div>
            """)
        
        # System info
        with gr.Row():
            with gr.Column():
                system_info = f"""
                **System Information:**
                - Device: {device}
                - PyTorch: {torch.__version__}
                - Environment: {'Google Colab' if check_colab_environment() else 'Local'}
                """
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    system_info += f"\n- GPU: {gpu_name} ({gpu_memory:.1f}GB)"
                
                gr.Markdown(system_info)
        
        # Main interface
        with gr.Tab("🎵 Audio Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Input controls
                    prompt = gr.Textbox(
                        label="Text Prompt",
                        placeholder="Enter your audio description (e.g., 'Ocean waves crashing', 'Jazz piano music')",
                        lines=2
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (Optional)",
                        placeholder="Describe what you don't want in the audio",
                        visible=False
                    )
                    
                    with gr.Row():
                        video_file = gr.File(
                            label="Upload Video File (Optional)",
                            file_types=[".mp4", ".avi", ".mov", ".mkv"]
                        )
                        video_path = gr.Textbox(
                            label="Or Video Path",
                            placeholder="Enter video file path",
                            visible=False
                        )
                
                with gr.Column(scale=1):
                    # Generation settings
                    with gr.Accordion("⚙️ Generation Settings", open=True):
                        steps = gr.Slider(
                            minimum=10, maximum=250, step=10, value=100,
                            label="Steps (higher = better quality, slower)"
                        )
                        cfg_scale = gr.Slider(
                            minimum=1.0, maximum=15.0, step=0.5, value=7.0,
                            label="CFG Scale (prompt adherence)"
                        )
                        seed = gr.Number(
                            label="Seed (-1 for random)", value=-1, precision=0
                        )
                        
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        sampler_type = gr.Dropdown(
                            choices=["dpmpp-3m-sde", "dpmpp-2m-sde", "k-heun", "k-lms"],
                            value="dpmpp-3m-sde",
                            label="Sampler Type"
                        )
                        sigma_min = gr.Slider(
                            minimum=0.01, maximum=1.0, step=0.01, value=0.03,
                            label="Sigma Min"
                        )
                        sigma_max = gr.Slider(
                            minimum=100, maximum=1000, step=50, value=500,
                            label="Sigma Max"
                        )
            
            # Generation controls
            with gr.Row():
                generate_btn = gr.Button("🎵 Generate Audio", variant="primary", size="lg")
                clear_btn = gr.Button("🧹 Clear Memory", variant="secondary")
            
            # Outputs
            with gr.Row():
                with gr.Column():
                    video_output = gr.Video(label="Output Video (if input video provided)")
                    audio_output = gr.Audio(label="Generated Audio", type="filepath")
            
            # Memory management
            def clear_memory_and_update():
                clear_memory()
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    return f"🧹 Memory cleared! GPU: {memory_used:.1f}GB / {memory_total:.1f}GB used"
                return "🧹 Memory cleared!"
            
            clear_btn.click(
                fn=clear_memory_and_update,
                outputs=gr.Textbox(label="Memory Status", visible=False)
            )
        
        # Examples tab
        with gr.Tab("📚 Examples"):
            gr.Markdown("### 🎯 Try these example prompts:")
            
            examples = [
                ["Typing on a keyboard", 100, 7.0],
                ["Ocean waves crashing on the shore", 100, 7.0],
                ["Jazz piano music in a cozy cafe", 150, 8.0],
                ["Footsteps walking on gravel", 100, 6.0],
                ["Rain falling on a tin roof", 120, 7.5],
                ["Birds chirping in a forest", 100, 7.0],
                ["Electric guitar solo rock music", 150, 8.0],
                ["Classical orchestra playing symphony", 200, 8.5],
            ]
            
            gr.Examples(
                examples=examples,
                inputs=[prompt, steps, cfg_scale],
                label="Click to load example"
            )
        
        # Usage guide
        with gr.Tab("📖 Usage Guide"):
            gr.Markdown("""
            ## 🎯 How to Use AudioX
            
            ### 📝 Text-to-Audio Generation:
            1. Enter a descriptive text prompt
            2. Adjust generation settings if needed
            3. Click "Generate Audio"
            4. Wait for the audio to be generated (1-3 minutes)
            
            ### 🎬 Video-to-Audio Generation:
            1. Upload a video file or enter video path
            2. Enter a prompt like "Generate audio for this video"
            3. Adjust settings and generate
            
            ### ⚙️ Settings Guide:
            - **Steps**: More steps = higher quality but slower generation
            - **CFG Scale**: Higher values make the output follow the prompt more closely
            - **Seed**: Use the same seed for reproducible results
            
            ### 💡 Tips for Best Results:
            - Be specific and descriptive in your prompts
            - For music, mention genre, instruments, and mood
            - For sound effects, describe the source and environment
            - Start with default settings and adjust if needed
            
            ### 🚨 Colab Limitations:
            - Generation may take longer on free Colab instances
            - Large video files may cause memory issues
            - Session may timeout after inactivity
            """)
        
        # Set up the generation function
        generate_inputs = [
            prompt, negative_prompt, video_file, video_path,
            gr.State(None), gr.State(None),  # audio_prompt_file, audio_prompt_path
            gr.State(0), gr.State(10),  # seconds_start, seconds_total
            cfg_scale, steps, gr.State(0),  # preview_every
            seed, sampler_type, sigma_min, sigma_max,
            gr.State(0.0),  # cfg_rescale
            gr.State(False), gr.State(None), gr.State(1.0)  # init audio settings
        ]
        
        generate_btn.click(
            fn=colab_generate_wrapper,
            inputs=generate_inputs,
            outputs=[video_output, audio_output],
            show_progress=True
        )
    
    return interface

def launch_colab_demo(
    model_config_path: str = "./model/config.json",
    ckpt_path: str = "./model/model.ckpt",
    share: bool = True,
    debug: bool = False
):
    """Launch the Colab demo with optimal settings"""
    
    print("🎛️ Creating AudioX Colab interface...")
    
    # Create the interface
    interface = create_colab_ui(
        model_config_path=model_config_path,
        ckpt_path=ckpt_path,
        model_half=False  # Set to True if memory issues
    )
    
    # Configure for Colab
    interface.queue(max_size=5)  # Limit queue for Colab
    
    print("🚀 Launching AudioX Demo...")
    print("📱 Interface will be available at the public URL")
    print("⏱️  Please wait for the model to load...")
    
    # Launch with Colab-optimized settings
    return interface.launch(
        share=share,
        debug=debug,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False,
        enable_queue=True
    )

if __name__ == "__main__":
    launch_colab_demo()
