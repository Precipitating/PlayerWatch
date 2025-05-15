import sys
from utils.bbox_utils import get_bottom_center_of_bbox
sys.path.append('../')
from utils import get_center_of_bbox, measure_dist


class PlayerBallAssign:
    def __init__(self, ball_dist):
        self.max_player_ball_dist = ball_dist

    """
    Gets the position between the ball and the player via the centre of the ball's bbox
    against the centre of the player's bbox
    
    Used to determine if the ball is close enough to to the player to determine it as possession.

    Args:
        players (sv.Detections): The current frame's player detection.
        ball_bbox (sv.Detections): The current frame's ball detection.

    Returns:
        -1 if ball and player is > min_dist
        tracker_id (unique player identifier) if < min_dist
    """
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


