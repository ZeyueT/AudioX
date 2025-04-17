import math
import random
import torch

from torch import nn
from typing import Tuple
import os
import subprocess as sp
from PIL import Image
from torchvision import transforms
from decord import VideoReader, cpu
import torchaudio
from torch.nn import functional as F

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output


class PadCrop_Normalized_T(nn.Module):

    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int, torch.Tensor]:
        n_channels, n_samples = source.shape

        # Calculate the duration of the audio in seconds
        total_duration = n_samples // self.sample_rate
        
        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        
        if self.randomize and n_samples > self.n_samples:
            valid_offsets = [
                i * self.sample_rate for i in range(0, total_duration, 10) 
                if i * self.sample_rate + self.n_samples <= n_samples and 
                (total_duration <= 20 or total_duration - i >= 15)
            ]
            if valid_offsets:
                offset = random.choice(valid_offsets)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )


class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal


def adjust_video_duration(video_tensor, duration, target_fps):
    current_duration = video_tensor.shape[0]
    target_duration = duration * target_fps
    if current_duration > target_duration:
        video_tensor = video_tensor[:target_duration]
    elif current_duration < target_duration:
        last_frame = video_tensor[-1:]
        repeat_times = target_duration - current_duration
        video_tensor = torch.cat((video_tensor, last_frame.repeat(repeat_times, 1, 1, 1)), dim=0)
    return video_tensor

def read_video(filepath, seek_time=0., duration=-1, target_fps=2):
    if filepath is None:
        return torch.zeros((int(duration * target_fps), 3, 224, 224))
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        resize_transform = transforms.Resize((224, 224))
        image = Image.open(filepath).convert("RGB")
        frame = transforms.ToTensor()(image).unsqueeze(0)
        frame = resize_transform(frame)
        target_frames = int(duration * target_fps)
        frame = frame.repeat(int(math.ceil(target_frames / frame.shape[0])), 1, 1, 1)[:target_frames]
        assert frame.shape[0] == target_frames, f"The shape of frame is {frame.shape}"
        return frame

    vr = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    seek_frame = int(seek_time * fps)
    if duration > 0:
        total_frames_to_read = int(target_fps * duration)
        frame_interval = int(math.ceil(fps / target_fps))
        end_frame = min(seek_frame + total_frames_to_read * frame_interval, total_frames)
        frame_ids = list(range(seek_frame, end_frame, frame_interval))
    else:
        frame_interval = int(math.ceil(fps / target_fps))
        frame_ids = list(range(0, total_frames, frame_interval))

    frames = vr.get_batch(frame_ids).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

    if frames.shape[2] != 224 or frames.shape[3] != 224:
        resize_transform = transforms.Resize((224, 224))
        frames = resize_transform(frames)

    video_tensor = adjust_video_duration(frames, duration, target_fps)
    assert video_tensor.shape[0] == duration * target_fps, f"The shape of video_tensor is {video_tensor.shape}"
    return video_tensor

def merge_video_audio(video_path, audio_path, output_path, start_time, duration):
    command = [
        'ffmpeg',
        '-y',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-strict', 'experimental',
        output_path
    ]
    
    try:
        sp.run(command, check=True)
        print(f"Successfully merged audio and video into {output_path}")
        return output_path
    except sp.CalledProcessError as e:
        print(f"Error merging audio and video: {e}")
        return None
    
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
