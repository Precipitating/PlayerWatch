import gzip
import itertools
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
            self.final_annotated_video = save_video_stream_frames(config['input_video_path'],
                                                                  config['annotated_video_path'])

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

    """
    Loads a pickle of sv.Detections of the ball frame by frame, formats it to pandas Dataframe via the
    ball's bounding box dimensions and interpolates non detections (determined if the ball's xyxy is np.nan)

    Args:
        batch_size (int): Batch size to process data X at a time (to prevent loading all the positions into memory) 

    Saves:
        ball_positions.pkl but with the gap detections interpolated to complete_ball_positions.pkl
        
    """
    def fill_missing_positions(self, batch_size=100):
        # batch load for memory efficiency
        with gzip.open(filename='stubs/ball_positions.pkl', mode="rb") as f, gzip.open(
                filename='stubs/complete_ball_positions.pkl', mode="ab") as f1:
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
                            det.xyxy[0] = np.array([df_ball_positions[i]])

                        pickle.dump(det, f1)
            f1.flush()

    """
    This function does two things:
    1.  Determines which player is in possession of the ball per frame and saves the player's tracker id to
        player_in_possession_buffer.pkl (if no possession, then saves None)
        
    2. IF save_video_output, annotates the ball and player in possession to the already player annotated video
       in stubs/annotated_players_path.mp4

    Args:
        f1 (int): Ball positions pickle file (read mode)
        f (int): Player positions pickle file (read mode)
        f2 (int): Player_in_possession pickle file (append binary mode)

    Saves:
        player_in_possession.pkl with possession per frame
        IF save_output_video: (output video path)/output.mp4 frames appended via self.final_annotated_video

    """
    def process_ball_tracking(self, f1, f, f2):
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
                    pickle.dump(player_in_possession.tracker_id, f2)
                else:
                    pickle.dump(None, f2)

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
                pickle.dump(player_in_possession.tracker_id, f2)
            else:
                pickle.dump(None, f2)

    """:
    Runs fill_missing_positions and then process_ball_tracking
    Saves:
        self.final.annotated_video frames fully completed and released (saved)
    """
    def handle_ball_tracking(self):
        self.fill_missing_positions()
        with (gzip.open(filename='stubs/complete_ball_positions.pkl',mode= "rb") as f,
              gzip.open(filename='stubs/player_positions.pkl', mode= "rb") as f1,
              gzip.open(filename='stubs/player_in_possession_buffer.pkl',mode= "ab") as f2):

            print("Processing ball tracking frames")
            while True:
                try:
                    self.process_ball_tracking(f1, f, f2)
                except EOFError:
                    print("Frames finished")
                    if self.config['save_output_video']:
                        self.final_annotated_video.release()
                    break
            f2.flush()


class Tracker:
    def __init__(self, model_path, ball_model_path, w, h, config):
        self.config = config
        # SAM2 variables
        if config['sam_2_mode']:
            self.sam_model = SAM(config['sam_2_model_path'])
            self.sam_prompt = 0
            self.sam_prompt_set = False
            self.florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base",
                                                                       torch_dtype=TORCH_DTYPE,
                                                                       trust_remote_code=True).to(DEVICE)
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

        # Player/Ball tracking variables
        self.model = YOLO(model_path)
        self.ball_model = YOLO(ball_model_path)
        self.tracker = sv.ByteTrack()
        self.ball_tracker = BallTracker(buffer_size=20)

        # Save output variables
        if config['save_output_video']:
            self.annotated_video_handle = save_video_stream_frames(source_path=config['input_video_path'],
                                                                   target_path=config['annotated_players_path'])
            # Label shapes for save_output_video
            self.ellipse_annotator = sv.EllipseAnnotator(
                color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                thickness=2

            )
            self.label_annotator = sv.LabelAnnotator(
                color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                text_color=sv.Color.from_hex('#000000'),
                text_position=sv.Position.BOTTOM_CENTER
            )

            self.triangle_ball_possessor_annotator = sv.TriangleAnnotator(
                color=sv.Color.from_hex('#880808'),
                base=25,
                height=21,
                outline_thickness=1
            )


        # Video width/depth for inference slicer
        self.w, self.h = w, h


    """:
    Runs assign_ball_to_player and returns its result
    
    Returns:
        -1 if no possession
        player's tracker id if possession
    """
    def get_player_in_possession(self, ball_bbox, players, assigner):
        assigned_player = assigner.assign_ball_to_player(players=players, ball_bbox=ball_bbox)
        if assigned_player != -1:
            return assigned_player
        else:
            return -1

    """
    Handles using YOLO and InferenceSlicer to detect the ball in the provided input video

    Args:
        batch (List[nd.array]):             A list of video frames for processing (process x at a time to ensure memory is not used up if video is large)
        ball_positions_file (io.BufferedWriter):  A handle to ball_positions.pkl (empty) on append binary mode

    Saves:
        ball_positions.pkl: sv.Detections of ball data, if no detections will save an empty sv.Detections with np.nan populated in its bbox (fill_missing_positions will interpolate this)

    """
    def get_ball_detections(self, batch, ball_positions_file):
        # slice frame in small parts and attempt to detect ball
        slicer = sv.InferenceSlicer(
            callback=self.callback,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            slice_wh=(self.w // 2 + 100, self.h // 2 + 100),
            overlap_ratio_wh=None,
            overlap_wh=(100, 100),
            iou_threshold=0.1,
            thread_workers=self.config['slice_threads']
        )
        os.makedirs('stubs', exist_ok=True)

        for frame in batch:
            ball_detections = slicer(frame)
            # save data to ball_positions.pkl
            if len(ball_detections) == 0:
                new_row = np.full((1, 4), np.nan)

                filler_detection = sv.Detections.empty()
                filler_detection.xyxy = np.vstack([filler_detection.xyxy, new_row])

                filler_detection.class_id = np.append(filler_detection.class_id, 0)
                filler_detection.confidence = np.append(filler_detection.confidence,
                                                        50.0)  # filler value, doesn't matter as its getting predicted

                pickle.dump(filler_detection, ball_positions_file)
            else:
                ball_detections = self.ball_tracker.update(ball_detections)
                pickle.dump(ball_detections[0], ball_positions_file)

    """
    Performs inference using the given image and task prompt. Used only if SAM 2 option is checked

    Args:
        image (PIL.Image or tensor): The input image for processing.
        task_prompt (str): The prompt specifying the task for the model.
        text_input (str, optional): Additional text input to refine the prompt.

    Returns:
        dict: The model's processed response after inference.
    """
    def florence_inference(self, image, task_prompt, text_input=None):

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


    """
    Runs florence inference and returns its bbox data if detected.

    Args:
        frame (nd.array) The first frame of the video, which will attempt to detect the prompt (ball)
    Returns:
        dict: ball's bbox data or None (if no detection)
    """
    def detect_ball_via_florence(self, frame):
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        results = self.florence_inference(frame, task_prompt, text_input='ball')["<OPEN_VOCABULARY_DETECTION>"]

        if results['bboxes']:
            return results['bboxes'][0]
        else:
            return None
    """
    Performs SAM 2 ball tracking instead of YOLO ball tracking, with help from florence 2 to get the initial ball
    position on the first frame.

    Args:
        input_path (string): Video input path
        frame (nd.array): The first frame of the video for florence 2 ball detection
        ball_positions_file (io.BufferedWriter): A file that will be populated with sv.Detections (ball data)

    Saves:
        ball_positions_file: Empty sv.Detections if no detection or the ball data
    """
    def get_sam_detections(self, input_path, frame, ball_positions_file):

        # filler sv.Detections variable
        filler_detection = sv.Detections.empty()
        filler_detection.xyxy = np.array([[np.nan, np.nan, np.nan, np.nan]])

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
                results = predictor(source=input_path, stream=True, bboxes=self.sam_prompt)
                os.makedirs('stubs', exist_ok=True)

                for result in results:
                    if len(result.boxes.xyxy) != 0:
                        print(result.boxes.xyxy.cpu().numpy())
                        filler_detection.xyxy = result.boxes.xyxy.cpu().numpy()
                        pickle.dump(filler_detection, ball_positions_file)
                    else:
                        print("Fake frame needed for SAM")
                        filler_detection.xyxy = np.array([[np.nan, np.nan, np.nan, np.nan]])
                        pickle.dump(filler_detection, ball_positions_file)
            except Exception as e:
                print(f"SAM Detection error {e}")

            self.sam_prompt_set = True

    """
    Callback for inference slicer to get the sv.Detection result of the ball if found.
    This is only for one frame, as the library currently doesn't support batched processing
    as it is still pending as a PR.

    Returns:
        sv.Detections: Ball detection if found.
    """
    def callback(self, patch: np.ndarray) -> sv.Detections:
        result = self.ball_model.predict(patch, conf=0.3)[0]
        return sv.Detections.from_ultralytics(result)

    """
    Grabs a batch_size of frames from the frame generator
    Args:
        gen (generator): A frame generator
        batch_size: The amount of frames to batch 
        
    Yields:
       A generator composed of a list of frames.
       Each function call advances the frames by batch_size (unless exhausted, which will return an empty list)
    """
    def batch_generator(self, gen, batch_size):
        while True:
            batch = list(itertools.islice(gen, batch_size))
            if not batch:
                break
            yield batch

    """
    Main function for processing player (YOLO) and ball (YOLO/SAM2 & FLORENCE2) tracking methods
    For each batch of frames:
        Gets ball tracking data and store in ball_positions.pkl
        Gets player tracking data and store in player_positions.pkl
        If save_output_video, save player annotated frames mp4 file (process_frame_batch)
    Args:
        frame_gen (generator): A frame generator of a video
    Returns:
        True if function has successfully ran
        False if Florence 2 fails to detect the ball in the first frame 
    """
    def process_player_and_ball_tracking(self, frame_gen):
        os.makedirs('stubs', exist_ok=True)
        with (gzip.open(filename='stubs/player_positions.pkl', mode='ab') as f,
              gzip.open(filename='stubs/ball_positions.pkl', mode='ab') as f1):

            for batch in tqdm(self.batch_generator(frame_gen, self.config['batch_size']), desc= "Processing Frames"):
                # get ball detections
                if not self.config['sam_2_mode']:
                    self.get_ball_detections(batch, f1)
                elif not self.sam_prompt_set:
                    self.get_sam_detections(self.config['input_video_path'], batch[0], f1)
                    if not self.sam_prompt_set:
                        ui.notify('First frame ball not found')
                        return False

                # get player detections (and annotate them if save_output_video)
                detections_batch = self.model.predict(batch, conf=0.3)
                for frame_in_batch, detections in zip(batch, detections_batch):
                    self.process_player_frame_batch(frame_in_batch=frame_in_batch,
                                                    detections=detections,
                                                    player_positions_file= f
                                                    )
            f.flush()
            f1.flush()

        if self.config['save_output_video']:
            self.annotated_video_handle.release()
        return True

    """
    This function processes player detections per frame and handles its annotation
    Args:
        detections (sv.Detections): A singular detection from a batch (aligns with frame_in_batch)
        frame_in_batch (nd.array): A singular video frame from a batch (aligns with detections)
        player_positions_file (io.BufferedWriter): player_positions.pkl file to be written to (append binary)
    Saves:
        sv.Detections of each player's data in player_positions.pkl (appended frame by frame)
    """
    def process_player_frame_batch(self, detections, frame_in_batch, player_positions_file):
        # expects a single frame
        detections = sv.Detections.from_ultralytics(detections)

        # all detection data including ball, ref and players
        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = self.tracker.update_with_detections(all_detections)

        players_detections = all_detections[all_detections.class_id == PLAYER_ID]

        # store to disk to reduce memory
        os.makedirs('stubs', exist_ok=True)
        pickle.dump(players_detections, player_positions_file)

        # draw annotations
        # write frame to an open video handle
        if self.config['save_output_video']:
            labels = [
                f"#{tracker_id}"
                for tracker_id
                in all_detections.tracker_id
            ]
            all_detections.class_id = all_detections.class_id.astype(int)
            annotated_frame = frame_in_batch.copy()  # copy the frame to not modify the original one
            annotated_frame = self.ellipse_annotator.annotate(annotated_frame, all_detections)
            annotated_frame = self.label_annotator.annotate(annotated_frame, all_detections, labels)
            self.annotated_video_handle.write(annotated_frame)
