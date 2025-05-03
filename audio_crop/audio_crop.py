#from faster_whisper import WhisperModel
from pywhispercpp.model import Model
import torch
from rapidfuzz import fuzz
import ffmpeg
import os
import uuid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AudioCrop:
    def __init__(self, target_name, input_file, output_dir, model_size = 'small'):
        self.model_size = model_size
        self.target_name = target_name
        self.input_video = input_file
        self.output_dir = output_dir

        self.timestamps = []

    # transcribe the video and get the crop durations
    def start_transcription(self,start_time_offset, crop_duration):
        print("Starting transcription...")
        model = Model('medium')
        words = model.transcribe(self.input_video, token_timestamps= True, max_len=1, n_processors= 4)
       # model = WhisperModel(self.model_size, device=DEVICE)
        #segments, _ = model.transcribe(self.input_video, word_timestamps=True)

        print("Starting timestamp find...")
        target_lower = self.target_name.lower()

        # Loop through segments and words

        for word in words:
            if fuzz.ratio(word.text.lower(), target_lower) > 70:
                print(word.text)
                print(word.t0)
                start_time = float(word.t0 / 100) + start_time_offset
                end_time = start_time + abs(crop_duration)
                self.start_cropping(start_time,end_time)

        # for segment in segments:
        #     for word in segment.words:
        #         # Check if the word matches the target name
        #         if fuzz.ratio(word.word.lower(), target_lower) > 70:
        #             print(word.word)
        #             # Calculate start and end times and store in the timestamps list
        #             start_time = word.start + start_time_offset
        #             end_time = start_time + abs(crop_duration)
        #             self.start_cropping(start_time,end_time)



    def start_cropping(self, start, end):
        new_folder_path = os.path.join(self.output_dir, self.target_name)
        os.makedirs(new_folder_path, exist_ok=True)
        output_path = os.path.join(new_folder_path, f"{uuid.uuid4()}.mp4")
        print(output_path)
        ffmpeg.input(self.input_video, ss=start, to=end).output(output_path).run()


