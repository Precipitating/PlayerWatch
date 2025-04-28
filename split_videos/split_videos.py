from tqdm import tqdm
import os
from utils import save_video


class VideoSplitter():
    def __init__(self, tracker_array, frame_gen, source_path):
        self.tracker_array = tracker_array
        self.frame_gen = frame_gen
        self.source_path = source_path

    def is_possession_consistent(self, current_idx, steps_ahead=10, forgive_frames=3):
        if current_idx + steps_ahead > len(self.tracker_array):
            return False

        result = True
        current_tracker_id = self.tracker_array[current_idx][0]
        error_count = 0

        for i in range(current_idx, current_idx + steps_ahead):

            if error_count >= forgive_frames:
                result = False
                break

            if self.tracker_array[i] is None or self.tracker_array[i][0] != current_tracker_id:
                error_count += 1

        return result

    def crop_videos(self):
        frame_start = -1
        current_tracked_player = -1
        current_trim_frames = []
        for frame_idx, frame in enumerate(tqdm(self.frame_gen, desc="Cropping videos")):

            if frame_idx >= len(self.tracker_array):
                return

            if self.tracker_array[frame_idx] is None:
                if current_tracked_player != -1:
                    current_trim_frames.append(frame)
                continue

            if self.tracker_array[frame_idx][0] == current_tracked_player:
                current_trim_frames.append(frame)
                continue

            frame_valid = self.is_possession_consistent(current_idx=frame_idx, steps_ahead=10, forgive_frames=3)

            if frame_valid:
                if frame_start == -1:
                    frame_start = frame_idx
                    current_tracked_player = self.tracker_array[frame_idx][0]
                # found end point
                elif self.tracker_array[frame_idx][0] != current_tracked_player:
                    current_trim_frames.append(frame)
                    output_dir = f"output_videos/{current_tracked_player}"
                    output_video_dir = f"output_videos/{current_tracked_player}/{frame_idx}.mp4"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_video(source_path=self.source_path, target_path=output_video_dir, frames=current_trim_frames)
                    frame_start = -1
                    current_tracked_player = -1
                    current_trim_frames = []

            if current_tracked_player != -1:
                current_trim_frames.append(frame)
