from sklearn.cluster import KMeans
from tqdm import tqdm
import supervision as sv
import os
import pickle


class TeamAssign:
    def __init__(self, input_frame_gen, input_model):
        self.frame_gen = input_frame_gen
        self.model = input_model
    def extract_crops(self, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                extracted_crop = pickle.load(f)
                print("found crop stub")
                return extracted_crop

        frame_batch = []

        crops = []
        for frame in tqdm(self.frame_gen, desc="Collecting Crops"):
            analyzed_frame = self.model.predict(frame, conf=0.3)[0]

            detections = sv.Detections.from_ultralytics(analyzed_frame)
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
            crops += players_crops


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(crops, f)


        return crops




