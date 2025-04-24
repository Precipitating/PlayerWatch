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

sys.path.append('../')

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

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





    def interpolate_ball_positions(self, ball_pos):
        # get ball's bounding box else return empty container
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_pos]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

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
        player_assigner = player_ball_assign.PlayerBallAssign()

        for frame in tqdm(frame_gen, desc="Processing Frames"):
            frame_batch.append(frame)

            if len(frame_batch) >= batch_size:
                # if batch size reached process it
                detections_batch = self.model.predict(frame_batch, conf= 0.1)

                # process each frame in batch
                for frame_in_batch, detections in zip(frame_batch, detections_batch):
                    detections = sv.Detections.from_ultralytics(detections)

                    # ball
                    ball_detections = detections[detections.class_id == BALL_ID]
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

                    all_detections = detections[detections.class_id != BALL_ID]
                    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
                    all_detections = self.tracker.update_with_detections(all_detections)

                    players_detections = all_detections[all_detections.class_id == PLAYER_ID]
                    goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
                    referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

                    # get player in possession of ball
                    player_in_possession = None
                    if len(ball_detections) != 0:
                        player_in_possession = self.get_player_in_possession(ball_bbox= ball_detections.xyxy[0], players= players_detections, assigner= player_assigner)

                        if player_in_possession != -1:
                            player_in_possession = players_detections[players_detections.tracker_id == player_in_possession]
                            player_in_possession.xyxy = sv.pad_boxes(xyxy=player_in_possession.xyxy, px=10)




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
                    annotated_frame = self.triangle_annotator.annotate(annotated_frame, ball_detections)

                    if isinstance(player_in_possession, sv.Detections):
                        annotated_frame = self.triangle_ball_possessor_annotator.annotate(annotated_frame, player_in_possession)


                    # Append annotated frame to results
                    annotated_result.append(annotated_frame)

                # Reset the batch
                frame_batch = []






        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(annotated_result, f)

        return annotated_result










    # def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
    #     # load data if read_from_stub is true
    #     if read_from_stub and stub_path is not None and os.path.exists(stub_path):
    #         with open(stub_path, 'rb') as f:
    #             tracks = pickle.load(f)
    #             print("found stub")
    #         return tracks
    #
    #
    #     detections = self.detect_frames(frames)
    #
    #     tracks = {
    #         "players":[], #
    #         "referees":[],
    #         "ball":[]
    #     }
    #
    #     for frame_num, detection in enumerate(detections):
    #         cls_names = detection.names
    #         # swap key/val around
    #         cls_names_inv = {v:k for k,v in cls_names.items()}
    #
    #         # convert to supervision detection format
    #         detection_supervision = sv.Detections.from_ultralytics(detection)
    #
    #         # convert GK to player
    #         for object_idx, class_id in enumerate(detection_supervision.class_id):
    #             if cls_names[class_id] == "goalkeeper":
    #                 detection_supervision.class_id[object_idx] = cls_names_inv["player"]
    #
    #
    #
    #         # track objects
    #         detections_with_tracks = self.tracker.update_with_detections(detection_supervision)
    #         tracks["players"].append({})
    #         tracks["referees"].append({})
    #         tracks["ball"].append({})
    #
    #         for frame_detection in detections_with_tracks:
    #             bbox = frame_detection[0].tolist()
    #             cls_id = frame_detection[3]
    #             track_id = frame_detection[4]
    #
    #             if cls_id == cls_names_inv['player']:
    #                 tracks["players"][frame_num][track_id] = {"bbox":bbox}
    #
    #             if cls_id == cls_names_inv['referee']:
    #                 tracks["referees"][frame_num][track_id] = {"bbox":bbox}
    #
    #
    #         for frame_detection in detection_supervision:
    #             bbox = frame_detection[0].tolist()
    #             cls_id = frame_detection[3]
    #
    #             if cls_id == cls_names_inv['ball']:
    #                 tracks["ball"][frame_num][1] = {"bbox": bbox}
    #
    #     if stub_path is not None:
    #         with open(stub_path, 'wb') as f:
    #             pickle.dump(tracks,f)
    #
    #     return tracks


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle = 0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )


        # draw player ids
        rectangle_width = 40
        rectangle_height = 20

        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2-rectangle_height//2) + 15
        y2_rect = (y2+rectangle_height//2) + 15


        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10


            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2

            )
        return frame


    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points],0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0, (0,0,0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ref_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            # draw ball

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))



            output_video_frames.append(frame)

        return output_video_frames



