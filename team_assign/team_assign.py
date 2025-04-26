from sklearn.cluster import KMeans
from tqdm import tqdm
import supervision as sv
import os
import pickle

PLAYER_ID = 2

class TeamAssign:
    def __init__(self, input_frame_gen, input_model):
        self.frame_gen = input_frame_gen
        self.model = input_model



    def get_clustering_model(self, image):
        # reshape to 2D array
        image_2d = image.reshape(-1, 3)

        # perform k means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2),:]

        # get cluster model
        kmeans = self.get_clustering_model(top_half_image)

        # get cluster labels per pixel

        labels = kmeans.labels_

        # reshape back to image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # get player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans


        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1
        self.player_team_dict[player_id] = team_id

        return team_id



    def extract_crops(self, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                extracted_crop = pickle.load(f)
                print("found crop stub")
                return extracted_crop

        frame_batch = []

        crops = []
        for frame in tqdm(self.frame_gen, desc="Collecting Crops"):
            #frame_batch.append(frame)
            analyzed_frame = self.model.predict(frame, conf=0.3)[0]

            detections = sv.Detections.from_ultralytics(analyzed_frame)
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            player_detections = detections[detections.class_id == PLAYER_ID]
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
            crops += players_crops


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(crops, f)


        return crops




