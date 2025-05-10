import cv2
import supervision as sv

def read_video(path, stride = 1):

    frame_gen = sv.get_video_frames_generator(path, stride)
    return frame_gen


def save_video_stream_frames(source_path, target_path):
    video_info = sv.VideoInfo.from_video_path(source_path)
    fps = video_info.fps
    width, height = video_info.resolution_wh

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    return out

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





