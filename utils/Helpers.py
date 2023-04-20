import cv2 as cv
import numpy as np
import math

def draw_lane(img, points, thickness = 5, color = [255, 0, 255]):
    """
    Draw the lane directly on top of the input image given points
    
    Params:
        img (np.ndarray): input image
        points (np.ndarray): coordinates of the lines in nested array [[left-top, left-bot], [right-bot, right-top]]
        thickness (int): thickness of the drawing line
        color ([int, int, int]): color or the drawing line, in RGB color map
        
    Returns:
        lane_img (np.ndarray): image that has been drawn onto
    """
    # Deep copy the input img
    lane_img = img.copy()
  
    # Draw lines
    cv.line(lane_img, points[0], points[1], color, thickness)
    cv.line(lane_img, points[3], points[2], color, thickness)
    
    # Draw points
    for point in points:
        cv.circle(lane_img, point, radius = 1, color = [255, 0, 0], thickness = 1)
    
    return lane_img

def draw_line(img, lines, color = [255, 0, 255], thickness = 2):
    """
    Draw the lane directly on top of the input image given lines
    
    Params:
        img (np.ndarray): input image
        lines (np.ndarray): list of coordinates of the lines in array [left-top, left-bot, right-bot, right-top]
        thickness (int): thickness of the drawing line
        color ([int, int, int]): color or the drawing line, in RGB color map
        
    Returns:
        lane_img (np.ndarray): image that has been drawn onto
    """
    # Deep copy the input img
    lane_img = img.copy()
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(lane_img, (x1, y1), (x2, y2), color, thickness)
                
    return lane_img

def scale_abs(x, m = 255):
    """
    """
    x = np.absolute(x)
    x = np.uint8(m * x / np.max(x))
    return x 


def peak(frame, **kwargs) -> (int, int):
    """
    Find the x-coordinates of the first points (bottom points) 
        
    Params:
        frame (np.ndarray): the input 1-channel frame
            
    Returns:
        leftx (int): the first point's left coordinate
        rightx (int): the first point's right coordinate
    """
    # Decide the cut position
    starting = int(kwargs.get("starting", frame.shape[0] / 2))
    ending = int(kwargs.get("ending", frame.shape[0] + 1))
        
    hist = np.sum(frame[starting:ending, :], axis = 0)
    mid = int(hist.shape[0] / 2)
    leftx = np.argmax(hist[:mid])
    rightx = np.argmax(hist[mid:]) + mid
        
    return leftx, rightx


def px_to_mm(distance, img_shape, warp_ratio, **kwargs):
    """
    """
    SENSOR_SIZE = [3.58, 2.02]
    DIST_CAMERA_ROAD = 370 # (mm) MEASURE THIS
    # OFFSET_CAMERA = 17     # (mm) MEASURE THIS
    focal_len = kwargs.get("focal", 4.046)

    width_ratio = SENSOR_SIZE[0] / img_shape[0]
    height_ratio = SENSOR_SIZE[1] / img_shape[1]
    pixel_to_mm = (width_ratio + height_ratio)/2

    # Calculate the shifted distance 
    mid_dist_mm = pixel_to_mm * distance * warp_ratio
    # actual_distance = mid_dist_mm * DIST_CAMERA_ROAD / focal_len + OFFSET_CAMERA  
    actual_distance = mid_dist_mm * DIST_CAMERA_ROAD / focal_len

    return actual_distance
