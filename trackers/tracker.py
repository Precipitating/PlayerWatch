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

import player_ball_assign
from utils import get_center_of_bbox, get_bbox_width, get_foot_pos
from sports.common.ball import BallTracker, BallAnnotator

sys.path.append('../')

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

class BallHandler:
    def __init__(self, incomplete_ball_positions, annotated_frames, player_positions, ball_dist):
        self.incomplete_ball_positions = incomplete_ball_positions
        self.complete_ball_positions = []
        self.annotated_frames = annotated_frames
        self.player_assigner = player_ball_assign.PlayerBallAssign(ball_dist)
        self.player_positions = player_positions
        self.ball_annotator = BallAnnotator(radius=7, buffer_size=10)

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
                det = self.incomplete_ball_positions[i]
                new_det = sv.Detections(xyxy=np.array([df_ball_positions[i]]),
                                        confidence= det.confidence[:1],
                                        class_id=det.class_id[:1],
                                        tracker_id=det.tracker_id[:1] if det.tracker_id is not None else None,
                                        data={k: v[:1] for k, v in det.data.items()}
                                        )
                self.incomplete_ball_positions[i] = new_det


    def handle_ball_tracking(self, read_from_stub = True, stub_path = None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                final_frame_result, player_in_possession_buffer = pickle.load(f)
                print("found ball tracking stub")
                return final_frame_result, player_in_possession_buffer

        self.fill_missing_positions()

        final_frame_result = []
        player_in_possession_buffer = []
        for frame_num, frame in enumerate(tqdm(self.annotated_frames, desc="Processing Ball Tracking Frames")):
            # track player in possession of ball if within distance
            player_in_possession = None

            player_in_possession = self.player_assigner.assign_ball_to_player(ball_bbox=self.incomplete_ball_positions[frame_num].xyxy[0],
                                                                              players=self.player_positions[frame_num])
            if player_in_possession != -1:
                player_in_possession = self.player_positions[frame_num][self.player_positions[frame_num].tracker_id == player_in_possession]
                player_in_possession_buffer.append(player_in_possession.tracker_id)
            else:
                player_in_possession_buffer.append(None)

            # go thru annotated frames and annotate the ball
            frame = self.ball_annotator.annotate(frame, self.incomplete_ball_positions[frame_num])

            # go thru annotated frames and annotate the player in possession of ball
            if isinstance(player_in_possession, sv.Detections):
                frame = self.triangle_ball_possessor_annotator.annotate(frame, player_in_possession)


            final_frame_result.append(frame)




        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump((final_frame_result, player_in_possession_buffer), f)


        return final_frame_result, player_in_possession_buffer







class Tracker:
    def __init__(self, model_path, ball_model_path, w,h):
        self.model = YOLO(model_path)
        self.ball_model = YOLO(ball_model_path)
        print(self.ball_model.names)
        self.tracker = sv.ByteTrack()

        self.ball_tracker = BallTracker(buffer_size=20)
        self.w, self.h = w,h


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


    def get_ball_detections(self, frame, ball_positions):
        # ball
        slicer = sv.InferenceSlicer(
            callback=self.callback,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            slice_wh=(self.w // 2 + 100, self.h // 2 + 100),
            overlap_ratio_wh=None,
            overlap_wh=(100, 100),
            iou_threshold=0.1
        )
        ball_detections = slicer(frame)

        if len(ball_detections) == 0:
            new_row = np.full((1, 4), np.nan)

            filler_detection = sv.Detections.empty()
            filler_detection.xyxy = np.vstack([filler_detection.xyxy, new_row])

            filler_detection.class_id = np.append(filler_detection.class_id, 0)
            filler_detection.confidence = np.append(filler_detection.confidence, 50.0)# filler value, doesn't matter as its getting predicted

            ball_positions.append(filler_detection)
        else:
            ball_detections = self.ball_tracker.update(ball_detections)
            ball_positions.append(ball_detections[0])



    def callback(self, patch: np.ndarray) -> sv.Detections:
        result = self.ball_model.predict(patch, conf=0.3)[0]
        return sv.Detections.from_ultralytics(result)




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

        for frame in tqdm(frame_gen, desc="Processing Frames"):
            frame_batch.append(frame)

            # since slicing can only handle one frame at a time, we'll do ball detection per frame
            self.get_ball_detections(frame, ball_positions)

            if len(frame_batch) >= batch_size:
                # if batch size reached process it
                detections_batch = self.model.predict(frame_batch, conf= 0.3)

                # process each frame in batch
                for frame_in_batch, detections in zip(frame_batch, detections_batch):
                    self.process_frame_batch(detections= detections,
                                             frame_in_batch= frame_in_batch,
                                             team_classifier= team_classifier,
                                             annotated_result= annotated_result,
                                             player_positions= player_positions
                                             )
                # Reset the batch
                frame_batch = []

            # handle rest of frames
        if len(frame_batch) > 0:
            print("processing rest of the frames")
            detections_batch = self.model.predict(frame_batch, conf=0.3)
            # process each frame in batch
            for frame_in_batch, detections in zip(frame_batch, detections_batch):
                self.process_frame_batch(detections=detections,
                                         frame_in_batch=frame_in_batch,
                                         team_classifier=team_classifier,
                                         annotated_result=annotated_result,
                                         player_positions=player_positions
                                         )

            frame_batch = []



        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump((annotated_result, ball_positions, player_positions), f)


        return annotated_result, ball_positions, player_positions




    def process_frame_batch(self, detections, frame_in_batch, team_classifier, annotated_result, player_positions):

        detections = sv.Detections.from_ultralytics(detections)

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
        annotated_frame = self.label_annotator.annotate(annotated_frame, all_detections, labels)

        # Append annotated frame to results
        annotated_result.append(annotated_frame)







