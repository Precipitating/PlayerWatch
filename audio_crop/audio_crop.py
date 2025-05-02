from faster_whisper import WhisperModel
import torch
from rapidfuzz import fuzz
import ffmpeg
import os
import uuid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COMPUTE_TYPE = 'float16' if torch.cuda.is_available() else 'int8'

class AudioCrop:
    def __init__(self, target_name, input_file, output_dir, model_size = 'large-v3'):
        self.model_size = model_size
        self.model = WhisperModel(self.model_size, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.target_name = target_name
        self.input_video = input_file
        self.output_dir = output_dir


    # transcribe the video and get the crop durations
    def start_transcription(self, start_time_offset, crop_duration):
        print("Starting transcription...")

        segments, _ = self.model.transcribe(self.input_video, word_timestamps=True)

        # get word level timestamps and find the target name
        name_start_timestamps = []

        for segment in segments:
            for word in segment.words:
                if fuzz.ratio(word.word.lower(), self.target_name.lower()) > 70:
                    print(word.word)
                    # get starting times
                    start_time = word.start + start_time_offset
                    name_start_timestamps.append(start_time)
                    # set the ending time
                    end_time = start_time + crop_duration
                    name_start_timestamps.append(end_time)

        return name_start_timestamps



    def start_cropping(self, timestamps):
        if not timestamps:
            print("Timestamps empty cannot crop.")
            return

        if len(timestamps) % 2 != 0:
            print("Odd timestamp count.. Should be even")
            return

        for i in range(0, len(timestamps), 2):
            new_folder_path = os.path.join(self.output_dir, self.target_name)
            os.makedirs(new_folder_path, exist_ok=True)
            output_path = os.path.join(new_folder_path, f"{uuid.uuid4()}.mp4")
            print(output_path)
            ffmpeg.input(self.input_video, ss=timestamps[i], to=timestamps[i+1]).output(output_path).run()

        print("Audio cropping done")






