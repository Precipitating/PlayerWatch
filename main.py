import cv2
from utils import read_video, save_video
from trackers import Tracker
from team_assign import TeamAssign
def main():
    # read video
    video_frames = read_video('input_videos/sample.mp4')

    ## init tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pk1')


    # assign player teams
    team_assigner = TeamAssign()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # draw output
    ## draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # save video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == "__main__":
    main()