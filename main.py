import cv2

from trackers.tracker import BallTracker
from utils import read_video, save_video
from trackers import Tracker
from team_assign import TeamAssign
from player_ball_assign import PlayerBallAssign
from camera_movement_estimator import CameraMovementEstimator
from sports.common.team import TeamClassifier
from utils.video_utils import save_video
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    input_video_path = 'input_videos/sample.mp4'
    output_video_path = 'output_videos/output.avi'
    tracker = Tracker('models/main/best.pt')

    # assign team colors
    frame_gen= read_video(input_video_path, stride= 30)

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


    ball_tracker = BallTracker(incomplete_ball_positions= ball_positions,
                               annotated_frames= annotated_frames,
                               player_positions= player_positions)

    ball_annotated_frames = ball_tracker.handle_ball_tracking()





    save_video(input_video_path, output_video_path, ball_annotated_frames)

    # tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pk1')




    #
    # # init tracker
    # tracker = Tracker('models/best.pt')
    # tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pk1')
    #
    # # get object positions
    # tracker.add_postiton_to_tracks(tracks)
    #
    #
    # # camera movement estimator
    # camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    # camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
    #                                                                           read_from_stub= True,
    #                                                                           stub_path= 'stubs/camera_movement_stub.pk1')
    #
    # camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    #
    # # interpolate ball pos
    # tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    #
    #
    # # assign player teams
    # team_assigner = TeamAssign()
    # team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    #
    # for frame_num, player_tracks in enumerate(tracks['players']):
    #     for player_id, track in player_tracks.items():
    #         team = team_assigner.get_player_team(video_frames[frame_num],
    #                                              track['bbox'],
    #                                              player_id)
    #
    #         tracks['players'][frame_num][player_id]['team'] = team
    #         tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    #
    #
    # # assign ball acquisition
    # player_assigner = PlayerBallAssign()
    # for frame_num, player_track in enumerate(tracks['players']):
    #     ball_bbox = tracks['ball'][frame_num][1]['bbox']
    #     assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
    #
    #     if assigned_player != -1:
    #         tracks['players'][frame_num][assigned_player]['has_ball'] = True
    #
    #
    # # draw output
    # ## draw object tracks
    # output_video_frames = tracker.draw_annotations(video_frames, tracks)
    #
    # # draw cam movement
    # output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    #
    # # save video
    # save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == "__main__":
    main()