import cv2 as cv
import numpy as np
import math
from utils.Helpers import peak

class PolyLines:
    def __init__(self, num_windows, margin, min_px):
        """
        Create a new instance of the PolyLines class
        
        Params:
            num_windows (int): # of sliding windows needed to detect a line
            margin (int): type of input images
            min_px (int): # of rows of the printed chessboard
        """
        # Set up parameters from input arguments
        self.min_px = min_px
        self.margin = margin
        self.n = num_windows

        # Initiate other parameters
        self.frame, self.h, self.w, self.window_height = None, None, None, None
        self.all_x, self.all_y = None, None
        self.left_pixel, self.right_pixel = [], []
        self.left_x, self.left_y = None, None
        self.right_x, self.right_y = None, None
        self.out_img = None 
        self.left_curve, self.right_curve = None, None
        self.lane_width, self.rela_position = None, None
        self.midpoints = None
        self.result = {}
    
    
    def store_details(self, frame):
        """
        Update instance's parameters
        
        Params:
            frame (np.ndarray): the input 1-channel frame
            
        Side-effects:
            Update all "other" instance's parameters
        """
        self.out_img = np.dstack((frame, frame, frame)) * 255
        self.frame = frame
        self.h, self.w = frame.shape[0], frame.shape[1]
        self.mid = self.h / 2
        self.window_height = int(self.h / self.n)  
        self.all_x = np.array(frame.nonzero()[1])
        self.all_y = np.array(frame.nonzero()[0])
        
    
    def next_y(self, w) -> (int, int):
        """
        Find the next y-coordinates
        
        Params:
            w (int): width of the sliding window
        
        Returns:
            (int, int): the next point's bottom & top y-coordinates, respective
        """
        return self.h - (w + 1) * self.window_height, self.h - w * self.window_height 

    
    def next_x(self, current) -> (int, int):
        """
        Find the next x-coordinates
        
        Params:
            current (int): current x-coordinates
            
        Returns:
            (int, int): the next point's left & right x-coordinates, respective
        """
        return current - self.margin, current + self.margin


    def indices_within_boundary(self, y_lo, y_hi, x_left, x_right):
        """
        Returns the segment within all input points
        
        Params:
            y_lo (float): bottom y-coordinate
            y_hi (float): top y-coordinate
            x_left (float): left x-coordinate
            x_right (float): right x-coordinate
        """
        cond1 = (self.all_y >= y_lo)
        cond2 = (self.all_y < y_hi)
        cond3 = (self.all_x >= x_left)
        cond4 = (self.all_x < x_right)
        return (cond1 & cond2 & cond3 & cond4 ).nonzero()[0]
    
    
    def pixel_locations(self, indices) -> (float, float):
        """
        Get the indexed pixel from all pixels
        
        Params:
            indices (int): pixel's index
            
        Returns:
            (float, float): the indexed pixels (x, y)
        """
        return self.all_x[indices], self.all_y[indices]
    
    
    def draw_midline(self):
        """
        Draw the midline for the detected lane
        """
        # Calculate the midpoints
        midpoints = self.get_midpoints()
        
        # Draw the midline
        for y in range(self.h):
            cv.circle(self.out_img, (midpoints[y], y), 1, [255,0,0], 1)         
            
            
    def get_midpoints(self):
        """
        Return the midpoints for the detected lane
        """
        # Calculate the left and right lane based on found polynomial coeffs
        kl, kr = self.left_curve, self.right_curve
        midx = []
        
        for y in range(self.h):
            xl = kl[0] * (y ** 2) + kl[1] * y + kl[2]
            xr = kr[0] * (y ** 2) + kr[1] * y + kr[2]
            midx.append(int(xl + (xr - xl) / 2))
            
        self.midpoints = midx
        return midx    
    
    
    def update_lane(self):
        """
        Update self.rela_position and self.lane_width based on the current detected lane
        """
        # Calculate the left and right lane based on found polynomial coeffs
        y, kl, kr = self.h, self.left_curve, self.right_curve
        xl = kl[0] * (y ** 2) + kl[1] * y + kl[2]
        xr = kr[0] * (y ** 2) + kr[1] * y + kr[2]
        
        # Calculate and update position and lane width
        position = xl + (xr - xl) / 2
        self.rela_position = (position - self.w / 2)
        self.lane_width = abs(xr - xl)
    

    def plot(self, t = 2):
        """
        Fit the polynomial line with 2 degree of freedom into detected lane points; then draw it
        
        Params:
            t (int): line stroke's size
            
        Side-effects:
            Modify self.out_img to contain the images of curve line and sliding windows onto the lanes
        """
        # Fit the polynomial lines into left and right lanes
        kl = np.polyfit(self.left_y, self.left_x, 2)
        kr = np.polyfit(self.right_y, self.right_x, 2)
        self.left_curve, self.right_curve = kl, kr        
        
        # Calculate the left and right lane x-coordinates for for each y-coordinate
        ys = np.linspace(0, self.h - 1, self.h) 
        xls = kl[0] * (ys ** 2) + kl[1] * ys + kl[2] # Left xs
        xrs = kr[0] * (ys ** 2) + kr[1] * ys + kr[2] # Right xs
        
        # Convert them into int array for drawing
        xls, xrs, ys = xls.astype(np.uint32), xrs.astype(np.uint32), ys.astype(np.uint32)
        
        # Draw lines
        for xl, xr, y in zip(xls, xrs, ys):
            cv.line(self.out_img, (xl - t, y), (xl + t, y), (0, 255, 0), int(t / 2))
            cv.line(self.out_img, (xr - t, y), (xr + t, y), (0, 0, 255), int(t / 2))
  

    def return_polylines(self, frame):
        """
        Returns the fitted polynomial lines with 2 degree of freedom to detected lanes
        
        Params:
            frame (np.ndarray): 1-channel frame input (grayscale)
            
        Returns:
            result (dict): a dictionary containing all needed information
        """
        self.store_details(frame)
        mid_leftx, mid_rightx = peak(frame)

        left_pixel, right_pixel = [], []
        x, y = [None, None, None, None], [None, None]

        for w in range(self.n):
            y[0], y[1] = self.next_y(w)
            x[0], x[1] = self.next_x(mid_leftx) 
            x[2], x[3] = self.next_x(mid_rightx)

            left_pixel.append(self.indices_within_boundary(y[0], y[1], x[0], x[1]))
            right_pixel.append(self.indices_within_boundary(y[0], y[1], x[2], x[3]))
    
        self.left_pixel = np.concatenate(left_pixel)
        self.right_pixel = np.concatenate(right_pixel)

        self.left_x, self.left_y = self.pixel_locations(self.left_pixel)
        self.right_x, self.right_y = self.pixel_locations(self.right_pixel)
        
        if (self.left_x.size != 0) and (self.left_y.size != 0) and (self.right_x.size != 0) and (self.right_y.size != 0):
            self.plot()
            self.draw_midline()
            self.update_lane()

        self.result = {
            'image': self.out_img,
            'left_lane': self.left_curve,
            'right_lane': self.right_curve,
            'lane': self.lane_width,
            'midpoints': self.midpoints,
            'position': self.rela_position, 
        }

        return self.result