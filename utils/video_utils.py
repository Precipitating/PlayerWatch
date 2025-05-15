import cv2
import supervision as sv

"""
Store a video's frames in a generator and return it.

Args:
    path(string): A path to the video specified by the user.
    stride(int): How many frames to skip per iteration
Returns:
    Generator of the video provided by path

"""
def read_video(path, stride = 1):

    frame_gen = sv.get_video_frames_generator(path, stride)
    return frame_gen

"""
Create a handle to a cv2.VideoWriter and return it
This is used to dynamically append frames to the handle and release when required.

Args:
    source_path(string): A path to the video specified by the user.
    target_path(int): Where the video should be saved when released
Returns:
    cv2.VideoWriter: Handle of the video

"""
def save_video_stream_frames(source_path, target_path):
    video_info = sv.VideoInfo.from_video_path(source_path)
    fps = video_info.fps
    width, height = video_info.resolution_wh

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    return out


"""
Save a video by providing a list of frames.
Args:
    source_path(string): A path to the video specified by the user.
    target_path(int): Where the video should be saved when released
    frames(List(nd.array)): A list of video frames
"""
def save_video(source_path, target_path, frames):
    video_info = sv.VideoInfo.from_video_path(source_path)
    fps = video_info.fps
    width, height = video_info.resolution_wh

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()





