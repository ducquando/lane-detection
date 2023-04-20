import cv2 as cv
import numpy as np
from utils.Helpers import draw_line, peak

def thresholding(frame) -> np.ndarray:
    """"""
    # Convert to HSB
    hls = cv.cvtColor(frame, cv.COLOR_RGB2HLS)
    lightness = hls[:, :, 1]

    # Use global threshold based on grayscale intensity.
    threshold = cv.inRange(lightness, 200, 255)
    
    return threshold


def edge_filter(frame) -> np.ndarray:
    """"""
    # Smooth with a Gaussian blur.
    kernel_size = 5
    blur_img = cv.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    # Perform Edge Detection.
    low_threshold = 60
    high_threshold = 80
    canny_img = cv.Canny(blur_img, low_threshold, high_threshold)
    
    return canny_img


def houghing(frame, **kwargs):
    """"""
    # Get optional arguments
    rho = kwargs.get("rho", 1)
    theta = kwargs.get("theta", np.pi / 180)
    threshold = kwargs.get("threshold", 50)
    min_len = kwargs.get("min_len", 10)
    max_gap = kwargs.get("max_gap", 50)

    lines = cv.HoughLinesP(frame, rho, theta, threshold, minLineLength = min_len, maxLineGap = max_gap)
    
    return lines


def extrapolate_lines(lines, upper_border, lower_border):
    """
    Extrapolate lines keeping in mind the lower and upper border intersections.
    """
    slopes = np.array([])
    consts = np.array([])
    
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            slope = (y1 - y2) / (x1 - x2) if (x1 - x2) != 0 else 0
            slopes = np.append(slopes, slope)
            c = y1 - slope * x1
            consts = np.append(consts, c)
            
    avg_slope = np.average(slopes)
    avg_consts = np.average(consts)
    
    # Calculate average intersection at lower_border.
    x_lane_lower_point = int((lower_border - avg_consts) / avg_slope)
    
    # Calculate average intersection at upper_border.
    x_lane_upper_point = int((upper_border - avg_consts) / avg_slope)
    
    return [x_lane_lower_point, lower_border, x_lane_upper_point, upper_border]


def extract_lanes(frame, lines):
    # Define bounds of the region of interest.
    roi_upper_border = 0
    roi_lower_border = frame.shape[0]

    # Use above defined function to identify lists of left-sided and right-sided lines.
    lines_left, lines_right = separate_left_right_lines(lines)

    # Extrapolate the lists of lines into recognized lane
    lane_left = extrapolate_lines(lines_left, roi_upper_border, roi_lower_border)
    lane_right = extrapolate_lines(lines_right, roi_upper_border, roi_lower_border)
    
    return lane_left, lane_right


def separate_left_right_lines(lines) -> (list, list):
    """
    Separate left and right lines depending on the slope.
    """
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > y2: # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])
                elif y1 < y2: # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])
    return left_lines, right_lines


def index_frame(frame, x, y, win) -> np.ndarray:
    """
    Crop the frame corresponding to the input middle x and y coordinates
    
    Params:
        frame (np.ndarray): the frame to be cropped
        x (int): middle x coordinate
        y (int): y coordinate
        win ([int, int]): the size of cropped region
        
    Returns:
        (np.ndarray): the cropped frame
    
    """
    new = frame.copy()
    reg_y = [y, y + int(win[1])] if y < (frame.shape[0] / 2) else [y - int(win[1]), y]
    reg_x = [x - int(win[0]/2), x + int(win[0]/2)]
    
    if x < int(win[0]/2):
        reg_x = [x, x + int(win[0])]
    elif x > frame.shape[1] - int(win[0]/2):
        reg_x = [x - int(win[0]), x]   
        
    return new[reg_y[0]:reg_y[1], reg_x[0]:reg_x[1]]


def sliding_window(frame, ptx, stride, win) -> (int, int, np.ndarray):
    """
    Slide a window across the top and the bottom of the frame
    
    Params:
        frame (np.ndarray): the input image frame
        stride (int): step size
        win ([int, int]): the window's size
    """
    point, y = int(ptx[0]), int(ptx[1])
    for x in range(point - 20, point + 20, stride):
        # yield the current window
        yield (x, y, index_frame(frame, x, y, win))

        
def return_windows(frame, points, win) -> np.ndarray:
    """
    Slide a window across the top and the bottom of the frame
    
    Params:
        frame (np.ndarray): the input image frame
        stride (int): step size
        win_size([int, int]): the window's size
        
    Returns:
        (np.ndarray): all cropped frames along the line
    """
    windows = []
    for (x, y) in points:
        x, y = int(x), int(y)
        windows.append(index_frame(frame, x, y, win))
        
    return windows


def return_windows_points(frame, points, win) -> list:
    """
    Slide a window across the top and the bottom of the frame
    
    Params:
        frame (np.ndarray): the input image frame
        stride (int): step size
        win_size([int, int]): the window's size
    """
    windows = []
    for (x, y) in points:
        x, y = int(x), int(y)
        windows.append(index_frame(frame, x, y, win))
    return list(zip(windows, points))


def calc_diff(this) -> float:  
    """
    Compare w' symetrical artificial lane windows
    """
    # Border
    BORDER = 6
    start = int((this.shape[0] - BORDER) / 2)
    
    # Straight lane
    straight = [0] * int(BORDER/2) + [1] * (this.shape[1] - BORDER) + [0] * int(BORDER/2)
    straight = np.full(this.shape, straight, dtype = int)
    s_mse = ((this - straight) ** 2).mean(axis = None)    
    
    # Left-shear
    left = np.zeros(this.shape, dtype = int)
    for i in range(start, start + BORDER):
        n = i - start + 1
        left[i] = [0] * n + [1] * (this.shape[1] - BORDER) + [0] * (BORDER - n)
    l_mse = ((this - left) ** 2).mean(axis = None)    
    
    # Right-shear
    right = np.zeros(this.shape, dtype = int)
    for i in range(start, start + BORDER):
        n = i - start + 1
        right[i] = [0] * (BORDER - n) + [1] * (this.shape[1] - BORDER) + [0] * n
    r_mse = ((this - right) ** 2).mean(axis = None)    
    
    # Determine the least mean squared error
    w_mse = s_mse if s_mse < l_mse and s_mse < r_mse else (l_mse if l_mse < r_mse else r_mse)
                
    # Return the weighted mean squared error
    return (s_mse + l_mse + r_mse + w_mse * 2) / 5


def is_next(new_mse, low_mse, new_x, low_x) -> bool:
    """
    Check criteria to be the next source point.
    To become the next one, its mse must lower than the currently lowest value and the distance between old and new point must not larger than 150px.
    
    Params:
        new_mse (float): the considering mse
        low_mse (float): the currently lowest mse
        new_x (int): new point's x-coordinate
        low_x (int): currently lowest mse's x-coordinate
    """
    smaller_mse = new_mse < low_mse
    max_space = abs(new_x - low_x) < 50
    
    return smaller_mse and max_space


def return_src_pts(frame) -> np.ndarray:
    """
    Return the source points for detected lane.
    The lanes are detected using Houghline and Canny edge detection algorithm.
    
    Params:
        frame(np.ndarray): frame to find source points
        
    Returns:
        (np.ndarray): this frame's source points (left-top, left-bottom, right-bottom, right-top)
    """
    # Threshold
    thresholded = thresholding(frame)
    
    # Filter
    filtered = edge_filter(thresholded)

    # Hough transform
    lines = houghing(filtered)

    # Draw all lines found onto a new image.
    hough_lines = draw_line(frame, lines)
    
    # Extract left and right lane
    lane_left, lane_right = extract_lanes(filtered, lines)

    # Left-top, left-bottom, right-bottom, right-top
    source_points = np.array([lane_left[2:], lane_left[0:2], lane_right[0:2], lane_right[2:]], np.float32)
    
    return source_points


def next_src_pts(curr_frame, points, win_size, stride = 4) -> np.ndarray:
    """
    Return this frame's source points accordingly to the previous point and a dark point
    
    Params:
        curr_frame (np.ndarray): frame to find source points
        points (np.ndarray): previous source points for lanes
        win_size ([int, int]): window's size
        stride (int): step size
    
    Returns:
        (np.ndarray): this frame's source points (left-top, left-bottom, right-bottom, right-top)
    """
    next_pts = []
    index = 0  # Keep track the source point
    
    for (curr_win, ptx) in return_windows_points(curr_frame, points, win_size): 
        # Initialize the best window to be that of the previous points
        best_win = [calc_diff(curr_win), ptx[0], ptx[1]]
        
        # Determine the peak window and compare
        cut = [0, 2] if index == 0 or index == 3 else [curr_frame.shape[0] - 1, curr_frame.shape[0] + 1]
        p_y = 0 if index == 0 or index == 3 else curr_frame.shape[0]
        p_left, p_right = peak(curr_frame, starting = cut[0], ending = cut[1])
        p_x = p_left if index == 0 or index == 1 else p_right
        peak_win = np.array(index_frame(curr_frame, int(p_x), int(p_y), win_size))
        mse = calc_diff(peak_win)
        best_win = [mse, p_x, p_y] if is_next(mse, best_win[0], p_x, best_win[1]) else best_win
        
        # Slide the windows and find other points having lower mse
        for (x, y, temp_win) in sliding_window(curr_frame, ptx, stride, win_size):
            temp_win, curr_win = np.array(temp_win), np.array(curr_win)
            if temp_win.shape == curr_win.shape:
                # Calculate the average mean square error and update if smaller
                mse = calc_diff(temp_win)
                best_win = [mse, x, y] if is_next(mse, best_win[0], x, best_win[1]) else best_win
        
        # Increase the index
        index += 1

        # Add the lowest-mse window positions to the final list
        next_pts.append(best_win[1:])  

    return np.array(next_pts, np.float32)