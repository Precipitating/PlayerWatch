from tqdm import tqdm
import os
from trackers.tracker import load_pickle_to_list
from utils import save_video
import gc

class VideoSplitter:
    def __init__(self, frame_gen, source_path, grace_period, frames_considered_possession, output_folder, ball_frame_forgiveness):
        self.tracker_array = []
        load_pickle_to_list(path= 'stubs/player_in_possession_buffer.pkl', container= self.tracker_array)
        self.frame_gen = frame_gen
        self.source_path = source_path
        self.possessor_grace_period = grace_period
        self.frames_considered_possession = frames_considered_possession
        self.ball_frame_forgiveness = ball_frame_forgiveness
        self.output_dir = output_folder



    """
    Function that determines if the player is considered possessing the ball
    It goes through the tracker_array from the current frame idx to steps_ahead times, and
    if error_count > forgive_frames, its not considered a consistent player possession.
    
    It is considered an error if the tracker id of the current frame isn't the same as the 
    tracker_id on the first index of the loop

    Args:
        current_idx (int): The current frame index
        steps_ahead (int): The amount of frames to check ahead step_ahead times
        forgive_frames (str, optional): The amount of errors allowed before not considered consistent anymore.

    Returns:
        False if error_count > forgive_frames
        True if it goes through tracker_array step_ahead times without error_count > forgive_frames
    """
    def is_possession_consistent(self, current_idx, steps_ahead=10, forgive_frames=3):
        # ensure we're not going over the video's length
        if current_idx > len(self.tracker_array):
            return False

        # ensure we're not comparing against no-one possessing the ball
        tracking = self.tracker_array[current_idx]
        if tracking is None:
            return False

        target_id = tracking[0]
        error_count = 0

        for i in range(current_idx, min(current_idx + steps_ahead, len(self.tracker_array))):
            tracking_i = self.tracker_array[i]
            if tracking_i is None or tracking_i[0] != target_id:
                error_count += 1
                if error_count > forgive_frames:
                    return False

        return True

    """
    Function handles saving the video after the video has been cropped correctly (trim_frames)
    
    Args:
        trim_frames (List(nd.array)): A list of frames that should start when the player is possessing, and end when they're not.
        player_id (int): Player id of the current player (ByteTrack assigned), could be same player but new ID due to them disappearing from the video.
        frame_idx (int): Current frame index, usually the end of a crop, used for naming the cropped video.

    Returns:
        Early return if no frames in trim_frames
        
    """
    def handle_end_point(self, trim_frames, player_id, frame_idx):
        print("end point reached")

        if not trim_frames:
            print("No frames to save for player", player_id)
            return

        output_dir = os.path.join(self.output_dir, str(player_id))
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{frame_idx}.mp4")
        print(f"Saving clip for player {player_id} to: {output_path}")

        save_video(
            source_path=self.source_path,
            target_path=output_path,
            frames=trim_frames
        )

    """
    Main function responsible for processing the cropped video
    Steps:
        1. Goes through a frame generator of the input video
        2. Gets the current player possessing the ball's tracker id via tracker_array (list from player_in_possession_buffer.pkl)
        3. If not tracking anyone yet, assign current_player to the tracked id and keep appending until a new player is consistently possessing the ball.
        4. If new possessor, keep appending frames to trim_frames until grace period is up
        5. Once grace period is up, save the cropped video by calling handle_end_point and pass in trim_frames.
        6. Reset all variables and repeat step 2.
        7. Once generator is exhausted and we're still tracking a player, just set end point to the end of the video and crop. 

    """
    def crop_videos(self):
        current_player = None
        trim_frames = []
        grace_counter = 0
        grace_active = False

        for frame_idx, frame in enumerate(tqdm(self.frame_gen, desc="Cropping videos")):
            tracker_info = self.tracker_array[frame_idx]
            tracked_id = tracker_info[0] if tracker_info else None

            # if we're in a grace period
            if grace_active:
                trim_frames.append(frame)
                grace_counter -= 1

                if grace_counter <= 0:
                    self.handle_end_point(trim_frames, current_player, frame_idx)
                    print("Cropped video saved")
                    current_player = None
                    trim_frames = []
                    grace_active = False
                continue

            # if we’re not tracking anyone yet
            if current_player is None:
                if tracked_id is not None and self.is_possession_consistent(
                        current_idx=frame_idx,
                        steps_ahead=self.frames_considered_possession,
                        forgive_frames=self.ball_frame_forgiveness
                ):
                    current_player = tracked_id
                    trim_frames.append(frame)
                continue

            # if tracking the same player
            if tracked_id == current_player:
                trim_frames.append(frame)
                continue

            # if another player is consistently possessing — begin grace period
            if tracked_id is not None and self.is_possession_consistent(
                    current_idx=frame_idx,
                    steps_ahead=self.frames_considered_possession,
                    forgive_frames=self.ball_frame_forgiveness
            ):
                grace_active = True
                grace_counter = self.possessor_grace_period
                trim_frames.append(frame)
                continue

            # if no possessor or inconsistent — still tracking current player
            if current_player is not None:
                trim_frames.append(frame)

        # handle hanging clip
        if current_player is not None and trim_frames:
            self.handle_end_point(trim_frames, current_player, len(self.tracker_array))




