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



    def handle_end_point(self, current_trim_frames, current_tracked_player, frame_idx):
        output_dir = f"output_videos/{current_tracked_player}"
        output_video_dir = f"output_videos/{current_tracked_player}/{frame_idx}.mp4"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_video(source_path=self.source_path, target_path=output_video_dir, frames=current_trim_frames)
        current_tracked_player = -1
        current_trim_frames = []

        # reset variables
        return -1, current_tracked_player, current_trim_frames

    def crop_videos(self):

       # clean_video_frames = list(self.frame_gen)

        frame_start = -1
        current_tracked_player = -1
        current_trim_frames = []
        current_grace_period = -1
        for frame_idx, frame in enumerate(tqdm(self.frame_gen, desc="Cropping videos")):

            # if we're in a grace period just keep adding frames
            if current_grace_period != -1:
                if current_grace_period > 0:
                    current_trim_frames.append(frame)
                    current_grace_period -= 1
                    continue
                else:
                    if current_grace_period == 0:
                        current_trim_frames.append(frame)
                        frame_start, current_tracked_player, current_trim_frames = self.handle_end_point(
                            current_trim_frames=current_trim_frames,
                            current_tracked_player=current_tracked_player,
                            frame_idx=frame_idx
                            )
                        current_grace_period = -1

            # if we're already tracking a player and tracker doesn't detect a possessor, just append frame and continue
            if self.tracker_array[frame_idx] is None:
                if current_tracked_player != -1:
                    current_trim_frames.append(frame)
                continue

            # if we keep hitting same tracker id on the player we're already tracking just append and continue
            if self.tracker_array[frame_idx][0] == current_tracked_player:
                current_trim_frames.append(frame)
                continue

            # check if we have a ball possessor that is atleast possessing for steps_ahead frames
            frame_valid = self.is_possession_consistent(current_idx=frame_idx, steps_ahead=20, forgive_frames=3)

            # if so, make it the current tracked player, else we found the clip end.
            if frame_valid:
                if frame_start == -1:
                    frame_start = frame_idx
                    current_tracked_player = self.tracker_array[frame_idx][0]
                # found end point, save it to its own folder.
                elif self.tracker_array[frame_idx][0] != current_tracked_player:
                    # add a grace period, we don't want to instantly trim as the new possessor needs time to control
                    if current_grace_period == -1:
                        current_grace_period = 30
                        continue



            if current_tracked_player != -1:
                current_trim_frames.append(frame)

        # we still have a hanging clip, end it if it ends without finding another possessor
        if current_tracked_player != -1:
            self.handle_end_point(current_trim_frames=current_trim_frames,
                                  current_tracked_player=current_tracked_player,
                                  frame_idx=len(self.tracker_array)
                                  )


