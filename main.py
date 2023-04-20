# Import libraries
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess

# Import helper files
from utils.Birdview import BirdView
from utils.Calibration import Calibration
from utils.Helpers import draw_lane, draw_line, px_to_mm, calc_angle, calc_distance
from utils.Sourcepoints import return_src_pts, next_src_pts, return_windows
from utils.Polylines import PolyLines
from utils.Lanefilter import LaneFilter
from moviepy.editor import VideoFileClip

def projection(Cali, Lane, Poly, iframe, pre_windows = None, last_pt = None):
    # Apply color filters
    undist, _ = Cali.undistort(iframe)
    filtered = Lane.apply(undist)

    # Find starting and ending points of the lanes
    first_time = (pre_windows is None) or (last_pt is None)
    try:
        src_pts = return_src_pts(undist) if first_time else next_src_pts(filtered, last_pt, win_size = [40, 10])
    except ValueError:
        src_pts = np.array([(200, 0), (50, 360), (390, 360), (400, 0)], np.float32)
    dst_pts = np.array([(200, 0), (200, 360), (400, 360), (400, 0)], np.float32)
    windows = return_windows(filtered, src_pts, win = [40, 10])
    
    # Set up Bird View
    Bird = BirdView(src_pts, dst_pts)
    bird = Bird.bird_view(filtered)
    
    # Project the result to the undistorted frame
    result = Poly.return_polylines(bird)
    projected = Bird.project(undist, filtered, result['left_lane'], result['right_lane'], result['midpoints'])
    
    # Convert from px to cm
    warp_ratio = (src_pts[2][0] - src_pts[1][0]) / (dst_pts[2][0] - dst_pts[1][0])
    pos_cm = round(px_to_mm(result["position"], iframe.shape[:2][::-1], warp_ratio, focal = Cali.focal_length()) / 10, 2)
    width_cm = round(px_to_mm(result["lane"], iframe.shape[:2][::-1], warp_ratio, focal = Cali.focal_length()) / 10, 2)
    direction = "right " if pos_cm > 0 else "left "
    
    # Draw text on video
    text_pos = "Camera: " + direction + str(abs(pos_cm)) + " cm"
    text_lane = "Lane width: " + str(width_cm) + " cm"
    cv.putText(projected, text_pos, (15, 30), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
    cv.putText(projected, text_lane, (15, 60), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)

    return projected, windows, src_pts


def main():
    # Read reference image to set up the calibration
    img = cv.imread("./Lane/7.jpg")
    Cali = Calibration(iframe = img)

    # Set up Lane Filtering
    params = {'sat_thresh': 120, 'light_thresh': 20, 'light_thresh_agr': 180, 'grad_thresh': (0.7, 1.4), 'mag_thresh': 40, 'x_thresh': 20}
    Lane = LaneFilter(params)
    
    # Set up Curving
    Poly = PolyLines(num_windows = 9, margin = 35, min_px = 50)

    # Read and display video from camera (no write)
    cap = cv.VideoCapture(0)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv.CAP_PROP_FPS))
    print(f'Height {frame_h}, Width {frame_w}')
    print(f'FPS : {frame_fps:0.2f}, Frames: {frame_count}')
  
    if not cap.isOpened():
        print("error reading")
    window, pts = None, None
    while cap.isOpened():
        # Capture frame-by-frame
        ret, img = cap.read()
        if not ret:
            break
        # Resize the frame
        img = cv.resize(img, (640, 360))
        new_frame, window, pts = projection(Cali, Lane, Poly, img, window, pts)
        # Display the frame
        cv.imshow('Frame', new_frame)
        # Wait for 10ms before displaying the next frame
        # You can change this value to control the playback speed
        key = cv.waitKey(10)
        if key == ord('q'):
            break
    
    # Realese everything when done
    cap.release()
    vid_out.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()