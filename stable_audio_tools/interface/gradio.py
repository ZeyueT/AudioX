import gc
import platform
import os
import subprocess as sp
import gradio as gr
import json 
import torch
import torchaudio

from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict
from ..data.utils import read_video, merge_video_audio


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device("cpu")

os.environ['TMPDIR'] = './tmp'

current_model_name = None
current_model = None



def load_model(model_name, model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model_configurations
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)
        print(f"Loading model checkpoint from {model_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")
    model.to(device).eval().requires_grad_(False)
    if model_half:
        model.to(torch.float16)
    print(f"Done loading model")
    return model, model_config, sample_rate, sample_size

def load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total):
    if audio_path is None:
        return torch.zeros((2, int(sample_rate * seconds_total)))
    audio_tensor, sr = torchaudio.load(audio_path)
    start_index = int(sample_rate * seconds_start)
    target_length = int(sample_rate * seconds_total)
    end_index = start_index + target_length
    audio_tensor = audio_tensor[:, start_index:end_index]
    if audio_tensor.shape[1] < target_length:
        pad_length = target_length - audio_tensor.shape[1]
        audio_tensor = F.pad(audio_tensor, (pad_length, 0))
    return audio_tensor

def generate_cond(
        prompt,
        negative_prompt=None,
        video_file=None,
        video_path=None,
        audio_prompt_file=None,
        audio_prompt_path=None,
        seconds_start=0,
        seconds_total=10,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        sample_size=485100,
        sample_rate=44100,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1
    ):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Prompt: {prompt}")
    preview_images = []
    if preview_every == 0:
        preview_every = None

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        has_mps = False
    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_name = 'default'
    cfg = model_configurations[model_name]
    model_config_path = cfg.get("model_config")
    ckpt_path = cfg.get("ckpt_path")
    pretrained_name = cfg.get("pretrained_name")
    pretransform_ckpt_path = cfg.get("pretransform_ckpt_path")
    model_type = cfg.get("model_type", "diffusion_cond")
    if model_config_path:
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None
    target_fps = model_config.get("video_fps", 5)
    global current_model_name, current_model
    if current_model is None or model_name != current_model_name:
        current_model, model_config, _, _ = load_model(
            model_name=model_name,
            model_config=model_config,
            model_ckpt_path=ckpt_path,
            pretrained_name=pretrained_name,
            pretransform_ckpt_path=pretransform_ckpt_path,
            device=device,
            model_half=False
        )
        current_model_name = model_name
        model = current_model
    else:
        model = current_model
    sample_rate = int(sample_rate)
    sample_size = int(sample_size)
    if video_file is not None:
        video_path = video_file.name
    elif video_path:
        video_path = video_path.strip()
    else:
        video_path = None
        
    if audio_prompt_file is not None:
        print(f'audio_prompt_file: {audio_prompt_file}')
        audio_path = audio_prompt_file.name
    elif audio_prompt_path:
        audio_path = audio_prompt_path.strip()
    else:
        audio_path = None
    
    Video_tensors = read_video(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=target_fps)        
    audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)
    
    audio_tensor = audio_tensor.to(device)
    seconds_input = sample_size / sample_rate
    print(f'video_path: {video_path}')
    
    if not prompt:
        prompt = ""
        
    conditioning = [{
        "video_prompt": [Video_tensors.unsqueeze(0)],        
        "text_prompt": prompt,
        "audio_prompt": audio_tensor.unsqueeze(0),
        "seconds_start": seconds_start,
        "seconds_total": seconds_input
    }] * batch_size
    if negative_prompt:
        negative_conditioning = [{
            "video_prompt": [Video_tensors.unsqueeze(0)],        
            "text_prompt": negative_prompt,
            "audio_prompt": audio_tensor.unsqueeze(0),
            "seconds_start": seconds_start,
            "seconds_total": seconds_total
        }] * batch_size
    else:
        negative_conditioning = None
    try:
        device = next(model.parameters()).device 
    except Exception as e:
        device = next(current_model.parameters()).device
    seed = int(seed)
    if not use_init:
        init_audio = None
    input_sample_size = sample_size
    if init_audio is not None:
        in_sr, init_audio = init_audio
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0)
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1)
        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)
        audio_length = init_audio.shape[-1]
        if audio_length > sample_size:
            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length
        init_audio = (sample_rate, init_audio)
    def progress_callback(callback_info):
        nonlocal preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]
        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))
    if mask_cropfrom is not None: 
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None 
    if model_type == "diffusion_cond":
        audio = generate_diffusion_cond(
            model, 
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            sample_size=input_sample_size,
            sample_rate=sample_rate,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            mask_args=mask_args,
            callback=progress_callback if preview_every is not None else None,
            scale_phi=cfg_rescale
        )
    elif model_type == "diffusion_uncond":
        audio = generate_diffusion_uncond(
            model, 
            steps=steps,
            batch_size=batch_size,
            sample_size=input_sample_size,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            callback=progress_callback if preview_every is not None else None
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    file_name = os.path.basename(video_path) if video_path else "output"
    output_dir = f"demo_result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_video_path = f"{output_dir}/{file_name}"
    torchaudio.save(f"{output_dir}/output.wav", audio, sample_rate)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if video_path:
        merge_video_audio(video_path, f"{output_dir}/output.wav", output_video_path, seconds_start, seconds_total)
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)
    del video_path
    torch.cuda.empty_cache()
    gc.collect()
    return (output_video_path, f"{output_dir}/output.wav")

def toggle_custom_model(selected_model):
    return gr.Row.update(visible=(selected_model == "Custom Model"))

def create_sampling_ui(model_config_map, inpainting=False):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🎧AudioX: Diffusion Transformer for Anything-to-Audio Generation  
            **[Project Page](https://zeyuet.github.io/AudioX/) · [Huggingface](https://huggingface.co/Zeyue7/AudioX) · [GitHub](https://github.com/ZeyueT/AudioX)**
            """
        )

        with gr.Tab("Generation"):

            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(show_label=False, placeholder="Enter your prompt")
                    negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt", visible=False)
                    video_path = gr.Textbox(label="Video Path", placeholder="Enter video file path")
                    video_file = gr.File(label="Upload Video File")
                    audio_prompt_file = gr.File(label="Upload Audio Prompt File", visible=False)
                    audio_prompt_path = gr.Textbox(label="Audio Prompt Path", placeholder="Enter audio file path", visible=False)
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Accordion("Video Params", open=False):                
                        seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Video Seconds Start")
                        seconds_total_slider = gr.Slider(minimum=0, maximum=10, step=1, value=10, label="Seconds Total", interactive=False)
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Accordion("Sampler Params", open=False):
                        steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
                        preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every")
                        cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG Scale")
                        sample_size_textbox = gr.Textbox(label="Sample Size (Sample Size / Sample Rate = Audio Duration)", value="485100")
                        sample_rate_textbox = gr.Textbox(label="Sample Rate (Sample Size / Sample Rate = Audio Duration)", value="44100")
                        seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")
                        sampler_type_dropdown = gr.Dropdown(
                            ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"],
                            label="Sampler Type",
                            value="dpmpp-3m-sde"
                        )
                        sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma Min")
                        sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma Max")
                        cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG Rescale Amount")
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Accordion("Init Audio", open=False, visible=False):
                        init_audio_checkbox = gr.Checkbox(label="Use Init Audio")
                        init_audio_input = gr.Audio(label="Init Audio")
                        init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.01, value=0.1, label="Init Noise Level")
            gr.Markdown("## Examples")
            with gr.Accordion("Click to show examples", open=False):
                with gr.Row():
                    gr.Markdown("**📝 Task: Text-to-Audio**")                
                    with gr.Column(scale=1.2):
                        gr.Markdown("Prompt: *Typing on a keyboard*")
                        ex1 = gr.Button("Load Example")
                    with gr.Column(scale=1.2):
                        gr.Markdown("Prompt: *Ocean waves crashing*")
                        ex2 = gr.Button("Load Example")
                    with gr.Column(scale=1.2):
                        gr.Markdown("Prompt: *Footsteps in snow*")
                        ex3 = gr.Button("Load Example")
                with gr.Row():
                    gr.Markdown("**🎶 Task: Text-to-Music**")                
                    with gr.Column(scale=1.2):
                        gr.Markdown("Prompt: *An orchestral music piece for a fantasy world.*")
                        ex4 = gr.Button("Load Example")
                    with gr.Column(scale=1.2):
                        gr.Markdown("Prompt: *Produce upbeat electronic music for a dance party*")
                        ex5 = gr.Button("Load Example")
                    with gr.Column(scale=1.2):
                        gr.Markdown("Prompt: *A dreamy lo-fi beat with vinyl crackle*")
                        ex6 = gr.Button("Load Example")
                with gr.Row():
                    gr.Markdown("**🎬 Task: Video-to-Audio**\nPrompt: *Generate general audio for the video*")
                    with gr.Column(scale=1.2):
                        gr.Video("example/V2A_sample-1.mp4")
                        ex7 = gr.Button("Load Example")
                    with gr.Column(scale=1.2):
                        gr.Video("example/V2A_sample-2.mp4")
                        ex8 = gr.Button("Load Example")
                    with gr.Column(scale=1.2):
                        gr.Video("example/V2A_sample-3.mp4")
                        ex9 = gr.Button("Load Example")
                with gr.Row():
                    gr.Markdown("**🎵 Task: Video-to-Music**\nPrompt: *Generate music for the video*")                
                    with gr.Column(scale=1.2):
                        gr.Video("example/V2M_sample-1.mp4")
                        ex10 = gr.Button("Load Example")
                    with gr.Column(scale=1.2):
                        gr.Video("example/V2M_sample-2.mp4")
                        ex11 = gr.Button("Load Example")
                    with gr.Column(scale=1.2):
                        gr.Video("example/V2M_sample-3.mp4")
                        ex12 = gr.Button("Load Example")
            with gr.Row():
                generate_button = gr.Button("Generate", variant='primary', scale=1)
            with gr.Row():
                with gr.Column(scale=6):
                    video_output = gr.Video(label="Output Video", interactive=False)
                    audio_output = gr.Audio(label="Output Audio", interactive=False)
                    send_to_init_button = gr.Button("Send to Init Audio", scale=1, visible=False)
            send_to_init_button.click(
                fn=lambda audio: audio,
                inputs=[audio_output],
                outputs=[init_audio_input]
            )
            inputs = [
                prompt, 
                negative_prompt,
                video_file,
                video_path,
                audio_prompt_file,
                audio_prompt_path,
                seconds_start_slider, 
                seconds_total_slider, 
                cfg_scale_slider, 
                steps_slider, 
                preview_every_slider,
                sample_size_textbox,
                sample_rate_textbox,
                seed_textbox, 
                sampler_type_dropdown, 
                sigma_min_slider, 
                sigma_max_slider,
                cfg_rescale_slider,
                init_audio_checkbox,
                init_audio_input,
                init_noise_level_slider
            ]
            generate_button.click(
                fn=generate_cond, 
                inputs=inputs,
                outputs=[
                    video_output,
                    audio_output
                ], 
                api_name="generate"
            ) 
            ex1.click(lambda: ["Typing on a keyboard", None, None, None, None, None, 0, 10, 7.0, 100, 0, "1225575558", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex2.click(lambda: ["Ocean waves crashing", None, None, None, None, None, 0, 10, 7.0, 100, 0, "3615819170", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex3.click(lambda: ["Footsteps in snow", None, None, None, None, None, 0, 10, 7.0, 100, 0, "1703896811", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex4.click(lambda: ["An orchestral music piece for a fantasy world.", None, None, None, None, None, 0, 10, 7.0, 100, 0, "1561898939", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex5.click(lambda: ["Produce upbeat electronic music for a dance party", None, None, None, None, None, 0, 10, 7.0, 100, 0, "406022999", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex6.click(lambda: ["A dreamy lo-fi beat with vinyl crackle", None, None, None, None, None, 0, 10, 7.0, 100, 0, "807934770", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex7.click(lambda: ["Generate general audio for the video", None, None, "example/V2A_sample-1.mp4", None, None, 0, 10, 7.0, 100, 0, "3737819478", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex8.click(lambda: ["Generate general audio for the video", None, None, "example/V2A_sample-2.mp4", None, None, 0, 10, 7.0, 100, 0, "1900718499", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex9.click(lambda: ["Generate general audio for the video", None, None, "example/V2A_sample-3.mp4", None, None, 0, 10, 7.0, 100, 0, "2289822202", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex10.click(lambda: ["Generate music for the video", None, None, "example/V2M_sample-1.mp4", None, None, 0, 10, 7.0, 100, 0, "3498087420", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex11.click(lambda: ["Generate music for the video", None, None, "example/V2M_sample-2.mp4", None, None, 0, 10, 7.0, 100, 0, "3753837734", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
            ex12.click(lambda: ["Generate music for the video", None, None, "example/V2M_sample-3.mp4", None, None, 0, 10, 7.0, 100, 0, "3510832996", "dpmpp-3m-sde", 0.03, 500, 0.0, False, None, 0.1], inputs=[], outputs=inputs)
        return demo

def create_txt2audio_ui(model_config_map):
    with gr.Blocks(css=".gradio-container { max-width: 1120px; margin: auto; }") as ui:
        with gr.Tab("Generation"):
            create_sampling_ui(model_config_map)
    return ui

def toggle_custom_model(selected_model):
    return gr.Row.update(visible=(selected_model == "Custom Model"))

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    global model_configurations
    global device

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    model_configurations = {
        "default": {
            "model_config": "./model/config.json",
            "ckpt_path": "./model/model.ckpt"
        }
    }
    ui = create_txt2audio_ui(model_configurations)
    return ui

if __name__ == "__main__":
    ui = create_ui(
        model_config_path='./model/config.json',
        share=True
    )
    ui.launch()
