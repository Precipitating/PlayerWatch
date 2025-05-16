from faster_whisper import WhisperModel, BatchedInferencePipeline
import torch
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

    """
    Main function responsible for using faster-whisper to find the target_name via word timestamping,
    checking if its similar enough via rapidfuzz, and if so, crops the video.
    
    It goes through word by word, checking if its similar enough and uses that as the start point of the crop.
    It then crops at a fixed duration from the word start (+ offset if specified) and saves the video (using ffmpeg)

    Args:
        start_time_offset(int): An offset in seconds from when the target_name is detected
        crop_duration(int): How long the crop should be (starting from the start_time + start_time_offset)
        similarity(float): The percentage of how similar the word should be to be considered correct.

    """
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

    """
    Uses ffmpeg to save the video, with provided parameters from start_transcription
    Output path is created and the name of the file will be the start time.

    Args:
        start(int): The start time of the video (in seconds)
        duration(int): How long the crop should starting from start (seconds)

    """
    def start_cropping(self, start, duration):
        new_folder_path = os.path.join(self.output_dir, self.target_name)
        os.makedirs(new_folder_path, exist_ok=True)
        output_path = os.path.join(new_folder_path, f"{start}.mp4")
        print(output_path)
        ffmpeg.input(self.input_video, ss=start).output(output_path, t=duration).run(overwrite_output= True)


