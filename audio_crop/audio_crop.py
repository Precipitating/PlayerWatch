from pywhispercpp.model import Model
import torch
from rapidfuzz import fuzz
import ffmpeg
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AudioCrop:
    def __init__(self, target_name, input_file, output_dir, model_size = 'small', processors=4):
        self.model_size = model_size
        self.target_name = target_name
        self.input_video = input_file
        self.output_dir = output_dir
        self.processors = processors

    # transcribe the video and get the crop durations
    def start_transcription(self,start_time_offset, crop_duration, similarity):
        print("Starting transcription...")
        model = Model(self.model_size)
        words = model.transcribe(self.input_video, token_timestamps= True, max_len=1, n_processors= self.processors)

        print("Starting timestamp find...")
        target_lower = self.target_name.lower()

        # Loop through segments and words

        for word in words:
            if fuzz.ratio(word.text.lower(), target_lower) > similarity:
                print(word.text)
                print(word.t0)
                start_time = float(word.t0 / 100) + start_time_offset
                end_time = start_time + abs(crop_duration)
                self.start_cropping(start_time,end_time)


    def start_cropping(self, start, end):
        new_folder_path = os.path.join(self.output_dir, self.target_name)
        os.makedirs(new_folder_path, exist_ok=True)
        output_path = os.path.join(new_folder_path, f"{start}.mp4")
        print(output_path)
        ffmpeg.input(self.input_video, ss=start, to=end).output(output_path).run()


