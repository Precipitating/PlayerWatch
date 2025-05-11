import gzip
from ultralytics import YOLO, SAM
from ultralytics.models.sam import SAM2VideoPredictor
import supervision as sv
import pickle
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import player_ball_assign
from nicegui import ui
from sports.common.ball import BallTracker, BallAnnotator
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch

from utils import read_video
from utils.video_utils import save_video_stream_frames

sys.path.append('../')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


def store_as_pickle(path, data):
    with gzip.open(path, 'ab') as f:
        pickle.dump(data, f)
        f.flush()

def store_as_pickle_individually(path, data):
    with gzip.open(path, 'ab') as f:
        for d in data:
            pickle.dump(d, f)
        f.flush()



def load_pickle_to_list(path, container):
    with gzip.open(path, "rb") as f:
        while True:
            try:
                container.append(pickle.load(f))
            except EOFError:
                break

class BallHandler:
    def __init__(self, ball_dist, config):
        self.complete_ball_positions = []
        self.player_assigner = player_ball_assign.PlayerBallAssign(ball_dist)
        self.ball_annotator = BallAnnotator(radius=7, buffer_size=10)
        self.config = config
        if config['save_output_video']:
            self.final_annotated_video = save_video_stream_frames(config['input_video_path'], config['annotated_video_path'])

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
    def fill_missing_positions(self, batch_size = 50):
        # batch load for memory efficiency
        with gzip.open('stubs/ball_positions.pkl', "rb") as f:
            while True:
                batch = []
                # unpickle a batch into a list
                for _ in range(batch_size):
                    try:
                        item = pickle.load(f)
                        batch.append(item)
                    except EOFError:
                        print("Missing positions ball positions end of file reached")
                        break

                if not batch:
                    break

                # process batch if it isn't empty
                if batch:
                    converted_positions = [x.xyxy[0] for x in batch]
                    # attempt to fill Nones via interpolation and backfill
                    df_ball_positions = pd.DataFrame(converted_positions, columns=['x1', 'y1', 'x2', 'y2'])
                    df_ball_positions = df_ball_positions.interpolate()
                    df_ball_positions = df_ball_positions.bfill()
                    df_ball_positions = df_ball_positions.to_numpy()

                    # convert batch back to Detections object and store in a new pickle
                    for i in range(len(batch)):
                        det = batch[i]
                        if np.isnan(det.xyxy[0]).any():
                            det.xyxy[0] =np.array([df_ball_positions[i]])

                        store_as_pickle(path='stubs/complete_ball_positions.pkl', data=det)




    def process_ball_tracking(self, f1, f):
        if self.config['save_output_video']:
            for frame in read_video(self.config['annotated_players_path']):
                # load data per frame
                current_player_positions = pickle.load(f1)
                complete_ball_pos = pickle.load(f)

                player_in_possession = self.player_assigner.assign_ball_to_player(
                    ball_bbox=complete_ball_pos.xyxy[0],
                    players=current_player_positions)
                if player_in_possession != -1:
                    player_in_possession = current_player_positions[
                        current_player_positions.tracker_id == player_in_possession]
                    store_as_pickle(path='stubs/player_in_possession_buffer.pkl',
                                    data=player_in_possession.tracker_id)
                else:
                    store_as_pickle(path='stubs/player_in_possession_buffer.pkl', data=None)

                # annotate the ball
                frame = self.ball_annotator.annotate(frame, complete_ball_pos)

                # go thru annotated frames and annotate the player in possession of ball
                if isinstance(player_in_possession, sv.Detections):
                    frame = self.triangle_ball_possessor_annotator.annotate(frame, player_in_possession)

                self.final_annotated_video.write(frame)
        else:
            # load data per frame
            current_player_positions = pickle.load(f1)
            complete_ball_pos = pickle.load(f)

            player_in_possession = self.player_assigner.assign_ball_to_player(
                ball_bbox=complete_ball_pos.xyxy[0],
                players=current_player_positions)
            if player_in_possession != -1:
                player_in_possession = current_player_positions[
                    current_player_positions.tracker_id == player_in_possession]
                store_as_pickle(path='stubs/player_in_possession_buffer.pkl',
                                data=player_in_possession.tracker_id)
            else:
                store_as_pickle(path='stubs/player_in_possession_buffer.pkl', data=None)


    def handle_ball_tracking(self):
        self.fill_missing_positions()
        with gzip.open('stubs/complete_ball_positions.pkl', "rb") as f, gzip.open('stubs/player_positions.pkl', "rb") as f1:
            print("Processing ball tracking frames")
            while True:
                try:
                    self.process_ball_tracking(f1,f)
                except EOFError:
                    print("Frames finished")
                    if self.config['save_output_video']:
                        self.final_annotated_video.release()
                    break






class Tracker:
    def __init__(self, model_path, ball_model_path, w,h, config):
        if config['sam_2_mode']:
            self.sam_model = SAM("models/sam/sam2_b.pt")
            self.sam_prompt = 0
            self.sam_prompt_set = False
            self.florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base",
                                                                       torch_dtype=TORCH_DTYPE,
                                                                       trust_remote_code=True).to(DEVICE)
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)


        self.model = YOLO(model_path)
        self.config = config
        self.ball_model = YOLO(ball_model_path)
        self.tracker = sv.ByteTrack()
        if config['save_output_video']:
            self.annotated_video_handle = save_video_stream_frames(source_path= config['input_video_path'], target_path= config['annotated_players_path'])



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





    def get_player_in_possession(self, ball_bbox, players, assigner):
        assigned_player = assigner.assign_ball_to_player(players=players, ball_bbox= ball_bbox)
        if assigned_player != -1:
            return assigned_player
        else:
            return -1


    def get_ball_detections(self, frame):
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

        os.makedirs('stubs', exist_ok=True)

        if len(ball_detections) == 0:
            new_row = np.full((1, 4), np.nan)

            filler_detection = sv.Detections.empty()
            filler_detection.xyxy = np.vstack([filler_detection.xyxy, new_row])

            filler_detection.class_id = np.append(filler_detection.class_id, 0)
            filler_detection.confidence = np.append(filler_detection.confidence, 50.0)# filler value, doesn't matter as its getting predicted

            store_as_pickle(path= 'stubs/ball_positions.pkl', data= filler_detection)
        else:
            ball_detections = self.ball_tracker.update(ball_detections)
            store_as_pickle(path='stubs/ball_positions.pkl', data=ball_detections[0])

    def florence_inference(self, image, task_prompt, text_input=None):
        """
        Performs inference using the given image and task prompt.

        Args:
            image (PIL.Image or tensor): The input image for processing.
            task_prompt (str): The prompt specifying the task for the model.
            text_input (str, optional): Additional text input to refine the prompt.

        Returns:
            dict: The model's processed response after inference.
        """
        # Combine task prompt with additional text input if provided
        prompt = task_prompt if text_input is None else task_prompt + text_input

        # Generate input data for model processing from the given prompt and image
        inputs = self.processor(
            text=prompt,  # Text input for the model
            images=image,  # Image input for the model
            return_tensors="pt",  # Return PyTorch tensors
        ).to("cuda", torch.float16)  # Move inputs to GPU with float16 precision

        # Generate model predictions (token IDs)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"].cuda(),  # text input IDs to CUDA
            pixel_values=inputs["pixel_values"].cuda(),  # pixel values to CUDA
            max_new_tokens=1024,  # Set maximum number of tokens to generate
            early_stopping=False,  # Disable early stopping
            do_sample=False,  # Use deterministic inference
            num_beams=3,  # Set beam search width for better predictions
        )

        # Decode generated token IDs into text
        generated_text = self.processor.batch_decode(
            generated_ids,  # Generated token IDs
            skip_special_tokens=False,  # Retain special tokens in output
        )[0]  # Extract first result from batch

        # Post-process the generated text into a structured response
        parsed_answer = self.processor.post_process_generation(
            generated_text,  # Raw generated text
            task=task_prompt,  # Task type for post-processing
            image_size=(image.width, image.height),  # scaling output
        )

        return parsed_answer  # Return the final processed output


    def detect_ball_via_florence(self, frame):
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        results = self.florence_inference(frame,task_prompt, text_input='ball')["<OPEN_VOCABULARY_DETECTION>"]

        if results['bboxes']:
            return results['bboxes'][0]
        else:
            return None




    def get_sam_detections(self, input_path, frame):

        filler_detection = sv.Detections.empty()
        filler_detection.xyxy = np.array([np.nan, np.nan, np.nan, np.nan])

        filler_detection.class_id = np.append(filler_detection.class_id, 0)
        filler_detection.confidence = np.append(filler_detection.confidence,
                                                50.0)  # filler value, doesn't matter as its getting predicted

        # attempt to get the ball bbox via florence (first frame)
        ball_bbox = self.detect_ball_via_florence(Image.fromarray(frame))
        if ball_bbox:
            self.sam_prompt = ball_bbox
            overrides = dict(conf=0.3, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")

            # Initialize the predictor
            predictor = SAM2VideoPredictor(overrides=overrides)

            try:
                # Use the predictor to get results
                results = predictor(source=input_path, stream=True, bboxes=self.sam_prompt)
                os.makedirs('stubs', exist_ok=True)

                for result in results:
                    if len(result.boxes.xyxy) != 0:
                        print(result.boxes.xyxy.cpu().numpy())
                        filler_detection.xyxy = result.boxes.xyxy.cpu().numpy()
                        store_as_pickle(path='stubs/ball_positions.pkl', data=filler_detection)
                    else:
                        print("Fake frame needed for SAM")
                        filler_detection.xyxy = np.array([np.nan, np.nan, np.nan, np.nan])
                        store_as_pickle(path='stubs/ball_positions.pkl', data=filler_detection)
            except Exception as e:
                print(f"SAM Detection error {e}")



            self.sam_prompt_set = True




    def callback(self, patch: np.ndarray) -> sv.Detections:
        result = self.ball_model.predict(patch, conf=0.3)[0]
        return sv.Detections.from_ultralytics(result)






    def initialize_and_annotate(self, frame_gen):

        frame_batch = []
        for frame in tqdm(frame_gen, desc="Processing Frames"):
            frame_batch.append(frame)

            # since slicing can only handle one frame at a time, we'll do ball detection per frame
            if not self.config['sam_2_mode']:
                self.get_ball_detections(frame)
            elif not self.sam_prompt_set:
                self.get_sam_detections(self.config['input_video_path'], frame)
                if not self.sam_prompt_set:
                    ui.notify('First frame ball not found')
                    return False



            if len(frame_batch) >= self.config['batch_size']:
                # if batch size reached process it
                detections_batch = self.model.predict(frame_batch, conf= 0.3)

                # process each frame in batch
                for frame_in_batch, detections in zip(frame_batch, detections_batch):
                    self.process_frame_batch(detections= detections,
                                             frame_in_batch= frame_in_batch)
                # Reset the batch
                frame_batch = []

            # handle rest of frames
        if len(frame_batch) > 0:
            print("processing rest of the frames")
            detections_batch = self.model.predict(frame_batch, conf=0.3)
            # process each frame in batch
            for frame_in_batch, detections in zip(frame_batch, detections_batch):
                self.process_frame_batch(detections=detections,
                                         frame_in_batch=frame_in_batch)


        if self.config['save_output_video']:
            self.annotated_video_handle.release()
        return True






    def process_frame_batch(self, detections, frame_in_batch):

        detections = sv.Detections.from_ultralytics(detections)

        # all detection data including ball, ref and players
        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = self.tracker.update_with_detections(all_detections)

        players_detections = all_detections[all_detections.class_id == PLAYER_ID]

        # store to disk to reduce memory
        os.makedirs('stubs', exist_ok=True)
        store_as_pickle(path='stubs/player_positions.pkl', data=players_detections)

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

        # write frame to an open video handle
        if self.config['save_output_video']:
            self.annotated_video_handle.write(annotated_frame)



