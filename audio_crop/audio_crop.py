#from pywhispercpp.model import Model
from faster_whisper import WhisperModel, BatchedInferencePipeline
import torch
from ffmpeg import overwrite_output
from rapidfuzz import fuzz
import ffmpeg
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AudioCrop:
    def __init__(self, target_name, input_file, output_dir, model_size = 'small', batch_size=4):
        self.model_size = model_size
        self.target_name = target_name
        self.input_video = input_file
        self.output_dir = output_dir
        self.batch_size = batch_size

    # transcribe the video and get the crop durations
    def start_transcription(self,start_time_offset, crop_duration, similarity):
        print("Starting transcription...")
        model = WhisperModel(self.model_size, device= DEVICE, compute_type="auto")
        batched_model = BatchedInferencePipeline(model=model)
        target_lower = self.target_name.lower()
        segments, _ = batched_model.transcribe(self.input_video, vad_filter= True, word_timestamps= True, batch_size= self.batch_size)

        print("Starting timestamp find...")
        for seg in segments:
            for word in seg.words:
                if fuzz.ratio(word.word.lower(), target_lower) > similarity:
                    start_time = word.start + start_time_offset
                    self.start_cropping(start_time,float(crop_duration))


    def start_cropping(self, start, duration):
        new_folder_path = os.path.join(self.output_dir, self.target_name)
        os.makedirs(new_folder_path, exist_ok=True)
        output_path = os.path.join(new_folder_path, f"{start}.mp4")
        print(output_path)
        ffmpeg.input(self.input_video, ss=start).output(output_path, t=duration).run(overwrite_output= True)


