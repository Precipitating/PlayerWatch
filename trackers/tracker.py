import copy

import cv2
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from filterpy.kalman import KalmanFilter
from ultralytics.models.sam.amg import build_all_layer_point_grids

import player_ball_assign
from utils import get_center_of_bbox, get_bbox_width, get_foot_pos

sys.path.append('../')

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

class BallTracker:
    def __init__(self, incomplete_ball_positions, annotated_frames, player_positions):
        self.incomplete_ball_positions = incomplete_ball_positions
        self.complete_ball_positions = []
        self.annotated_frames = annotated_frames
        self.player_assigner = player_ball_assign.PlayerBallAssign()
        self.player_positions = player_positions

        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )

        self.triangle_ball_possessor_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#880808'),
            base=25,
            height=21,
            outline_thickness=1
        )




    # interpolate and fill missing positions
    def fill_missing_positions(self):
        converted_positions = [x.xyxy[0] for x in self.incomplete_ball_positions]
        df_ball_positions = pd.DataFrame(converted_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.to_numpy()


        for i in range(len(self.incomplete_ball_positions)):
            if np.isnan(self.incomplete_ball_positions[i].xyxy[0]).any():
                prev_det = self.incomplete_ball_positions[i - 1]
                new_det = sv.Detections(xyxy=np.array([df_ball_positions[i]]),
                                        confidence= prev_det.confidence[:1],
                                        class_id=prev_det.class_id[:1],
                                        tracker_id=prev_det.tracker_id[:1] if prev_det.tracker_id is not None else None,
                                        data={k: v[:1] for k, v in prev_det.data.items()}
                                        )
                self.incomplete_ball_positions[i] = new_det






    def handle_ball_tracking(self):
        self.fill_missing_positions()

        final_frame_result = []
        for frame_num, frame in enumerate(tqdm(self.annotated_frames, desc="Processing Ball Tracking Frames")):
            # track player in possession of ball if within distance
            player_in_possession = None

            player_in_possession = self.player_assigner.assign_ball_to_player(ball_bbox=self.incomplete_ball_positions[frame_num].xyxy[0],
                                                                              players=self.player_positions[frame_num])
            if player_in_possession != -1:
                player_in_possession = self.player_positions[frame_num][self.player_positions[frame_num].tracker_id == player_in_possession]
                player_in_possession.xyxy = sv.pad_boxes(xyxy=player_in_possession.xyxy, px=10)

            # go thru annotated frames and annotate the ball
            frame = self.triangle_annotator.annotate(frame, self.incomplete_ball_positions[frame_num])

            # go thru annotated frames and annotate the player in possession of ball
            if isinstance(player_in_possession, sv.Detections):
                frame = self.triangle_ball_possessor_annotator.annotate(frame, player_in_possession)


            final_frame_result.append(frame)


        return final_frame_result







class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        print(self.model.names)
        self.tracker = sv.ByteTrack()


        # initialize labels
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2

        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )
        self.triangle_ball_possessor_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#880808'),
            base=25,
            height=21,
            outline_thickness=1
        )

        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            radius=8)

        self.vertex_annotator_2 = sv.VertexAnnotator(
            color=sv.Color.from_hex('#00BFFF'),
            radius=8)








    def add_postiton_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_pos(bbox)
                    tracks[object][frame_num][track_id]['position'] = position



    def resolve_goalkeepers_team_id(self, players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
        team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
        goalkeepers_team_id = []
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

        return np.array(goalkeepers_team_id)


    def get_player_in_possession(self, ball_bbox, players, assigner):

        assigned_player = assigner.assign_ball_to_player(players=players, ball_bbox= ball_bbox)
        if assigned_player != -1:
            return assigned_player
        else:
            return -1




    def initialize_and_annotate(self, frame_gen, batch_size, team_classifier, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                processed_frames = pickle.load(f)
                print("found stub")
                return processed_frames

        frame_batch = []
        annotated_result = []
        ball_positions = []
        player_positions = []

        player_assigner = player_ball_assign.PlayerBallAssign()

        for frame in tqdm(frame_gen, desc="Processing Frames"):
            frame_batch.append(frame)

            if len(frame_batch) >= batch_size:
                # if batch size reached process it
                detections_batch = self.model.predict(frame_batch, conf= 0.3)

                # process each frame in batch
                for frame_in_batch, detections in zip(frame_batch, detections_batch):
                    detections = sv.Detections.from_ultralytics(detections)

                    # ball
                    ball_detections = detections[detections.class_id == BALL_ID]
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

                    if len(ball_detections) == 0:
                        ball_positions.append(ball_positions[-1])
                        ball_positions[-1].xyxy[0] = np.nan
                    else:
                        ball_positions.append(ball_detections[0])



                    all_detections = detections[detections.class_id != BALL_ID]
                    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
                    all_detections = self.tracker.update_with_detections(all_detections)

                    players_detections = all_detections[all_detections.class_id == PLAYER_ID]
                    player_positions.append(copy.deepcopy(players_detections))



                    goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
                    referees_detections = all_detections[all_detections.class_id == REFEREE_ID]


                    players_crops = [sv.crop_image(frame_in_batch, xyxy) for xyxy in players_detections.xyxy]
                    players_detections.class_id = team_classifier.predict(players_crops)

                    goalkeepers_detections.class_id = self.resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)

                    referees_detections.class_id -= 1

                    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])



                    labels = [
                        f"#{tracker_id}"
                        for tracker_id
                        in all_detections.tracker_id
                    ]
                    all_detections.class_id = all_detections.class_id.astype(int)

                    # draw annotations
                    annotated_frame = frame_in_batch.copy()  # copy the frame to not modify the original one
                    annotated_frame = self.ellipse_annotator.annotate(annotated_frame, all_detections)
                    annotated_frame = self.label_annotator.annotate(annotated_frame, all_detections,labels)
                   # annotated_frame = self.triangle_annotator.annotate(annotated_frame, ball_detections)


                    # Append annotated frame to results
                    annotated_result.append(annotated_frame)

                # Reset the batch
                frame_batch = []






        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump((annotated_result, ball_positions, player_positions), f)


        return annotated_result, ball_positions, player_positions







