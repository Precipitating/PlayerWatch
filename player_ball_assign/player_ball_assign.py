import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_dist


class PlayerBallAssign():
    def __init__(self):
        self.max_player_ball_dist = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos = get_center_of_bbox(ball_bbox)

        min_dist= 99999
        assigned_player= -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            dist_left = measure_dist((player_bbox[0], player_bbox[-1]),ball_pos)
            dist_right = measure_dist((player_bbox[2], player_bbox[-1]),ball_pos)

            dist = min(dist_left, dist_right)

            if dist < self.max_player_ball_dist:
                if dist < min_dist:
                    min_dist = dist
                    assigned_player = player_id


        return assigned_player


