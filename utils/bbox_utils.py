import math

"""
Gets the centre of a bbox
Args:
    bbox(tuple): A bbox tuple x1,y1,x2,y2
Returns:
    A tuple containing half of its width and height.
"""
def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)


"""
Gets the bottom centre of a bbox.
This is used to attempt to align it to the player's foot for ball to player distance tracking.
Args:
    bbox(tuple): A bbox tuple x1,y1,x2,y2
Returns:
    A bbox containing the center of its x and bottom of its y.
"""
def get_bottom_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    bottom_y = y2
    return int(center_x),int(bottom_y)


"""
Uses pythagoras to get the distance between two points
Args:
    p1: A bbox tuple x1,y1,x2,y2
    p2: A bbox tuple x1,y1,x2,y2
Returns:
    The distance between the two bboxes (pixels)
"""
def measure_dist(p1,p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

