import cv2

from trackers.tracker import BallTracker, BallHandler
from utils import read_video, save_video
from trackers import Tracker
from team_assign import TeamAssign
from sports.common.team import TeamClassifier
from split_videos import VideoSplitter
import supervision as sv
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    input_video_path = 'input_videos/vid.mp4'
    output_video_path = 'output_videos/output.avi'
    video_info = sv.VideoInfo.from_video_path(input_video_path)
    w, h = video_info.width, video_info.height
    tracker = Tracker('models/main/repobest.pt','models/ball/640/best.pt', w= w, h= h)


    # assign team colors
    frame_gen= read_video(input_video_path, 2)

    team_assigner = TeamAssign(frame_gen, tracker.model)
    crops = team_assigner.extract_crops(read_from_stub= True, stub_path='stubs/crop_stub.pk1')
    team_classifier = TeamClassifier(device=DEVICE)
    team_classifier.fit(crops)

    # read video
    frame_gen = read_video(input_video_path)

    # initialize model and annotate ball and player
    annotated_frames, ball_positions, player_positions = tracker.initialize_and_annotate(frame_gen= frame_gen,
                                                                                         team_classifier= team_classifier,
                                                                                         batch_size= 20,
                                                                                         read_from_stub= True,
                                                                                         stub_path= 'stubs/annotation_stub.pk1')



    ball_handler = BallHandler(incomplete_ball_positions= ball_positions,
                               annotated_frames= annotated_frames,
                               player_positions= player_positions)

    ball_annotated_frames, player_in_possession_buffer = ball_handler.handle_ball_tracking(read_from_stub= True, stub_path = 'stubs/ball_stub.pk1')




    print(player_in_possession_buffer)
    frame_gen = read_video(input_video_path)

    video_splitter = VideoSplitter(tracker_array= player_in_possession_buffer, frame_gen= frame_gen, source_path= input_video_path)
    video_splitter.crop_videos()


    #save_video(input_video_path, output_video_path, ball_annotated_frames)





if __name__ == "__main__":
    main()