import os
import torch
import torchaudio
from einops import rearrange

from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import read_video, merge_video_audio
from stable_audio_tools.interface.gradio import load_and_process_audio


class AudioXGenerator:
    """ComfyUI node for generating audio with AudioX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {"default": ""}),
                "video_path": ("STRING", {"default": ""}),
                "audio_path": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_output", "audio_output")
    FUNCTION = "generate"
    CATEGORY = "audiox"

    def __init__(self):
        self.model = None
        self.model_config = None
        self.sample_rate = None
        self.sample_size = None
        self.target_fps = None

    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        if self.model is None:
            self.model, self.model_config = get_pretrained_model("HKUSTAudio/AudioX")
            self.sample_rate = self.model_config.get("sample_rate", 48000)
            self.sample_size = self.model_config.get("sample_size", 48000 * 10)
            self.target_fps = self.model_config.get("video_fps", 5)
            self.model = self.model.to(self._device()).eval()

    def generate(self, text_prompt: str, video_path: str, audio_path: str, seed: int):
        self._load_model()
        device = self._device()

        seconds_total = self.sample_size / self.sample_rate

        video_tensor = read_video(video_path if video_path else None, seek_time=0, duration=seconds_total, target_fps=self.target_fps)
        audio_tensor = load_and_process_audio(audio_path if audio_path else None, self.sample_rate, 0, seconds_total)

        conditioning = [{
            "video_prompt": [video_tensor.unsqueeze(0)],
            "text_prompt": text_prompt or "",
            "audio_prompt": audio_tensor.unsqueeze(0),
            "seconds_start": 0,
            "seconds_total": seconds_total
        }]

        output = generate_diffusion_cond(
            self.model,
            steps=250,
            cfg_scale=6.0,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sample_rate=self.sample_rate,
            seed=seed,
            device=device,
            sampler_type="dpmpp-3m-sde",
            sigma_min=0.03,
            sigma_max=500.0,
        )

        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

        os.makedirs("audiox_output", exist_ok=True)
        audio_out = os.path.join("audiox_output", "output.wav")
        torchaudio.save(audio_out, output, self.sample_rate)

        if video_path and os.path.exists(video_path):
            video_out = os.path.join("audiox_output", "output.mp4")
            merge_video_audio(video_path, audio_out, video_out, 0, seconds_total)
        else:
            video_out = ""

        return (video_out, audio_out)


NODE_CLASS_MAPPINGS = {
    "AudioXGenerator": AudioXGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioXGenerator": "AudioX Generator",
}
