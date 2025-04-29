import sys

from utils.bbox_utils import get_bottom_center_of_bbox

sys.path.append('../')
from utils import get_center_of_bbox, measure_dist


class PlayerBallAssign():
    def __init__(self):
        self.max_player_ball_dist = 50

    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos = get_center_of_bbox(ball_bbox)

        min_dist= float('inf')
        assigned_player= -1

        for i in range(len(players)):
            player_bbox = players.xyxy[i]
            player_pos = get_bottom_center_of_bbox(player_bbox)
            dist = measure_dist(player_pos, ball_pos)

            if dist < self.max_player_ball_dist and dist < min_dist:
                min_dist = dist
                assigned_player = players.tracker_id[i]


        return assigned_player


