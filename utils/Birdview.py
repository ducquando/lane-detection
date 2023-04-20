import cv2 as cv
import numpy as np 
from utils.Calibration import Calibration
from utils.Sourcepoints import thresholding

class BirdView:   
    def __init__(self, src, dst):
        """
        Params:
            src (np.ndarray): source image's points
            dst (np.ndarray): destination image's points
        """
        # Specify the source and destination image
        self.src = src
        self.dst = dst
        
        # Warp matrix from src to dst, and vice versa
        self.warp = cv.getPerspectiveTransform(self.src, self.dst)
        self.inv_warp = cv.getPerspectiveTransform(self.dst, self.src)
        
    
    def bird_view(self, img) -> np.ndarray:
        """
        Get bird's view of the road from the input image with car's view
        
        Params:
            img (np.ndarray): image of car's view
            
        Returns:
            warp_img (np.ndarray): the image of road's view
            
        """
        dimension = img.shape[:2][::-1]
        warp_img = cv.warpPerspective(img, self.warp, dimension, flags = cv.INTER_LINEAR)
       
        return warp_img
    
    
    def calc_slope(self, img, midpoints):
#         # Create an empty image
#         blank = np.zeros_like(img)
#         dimension = blank.shape

#         # Draw midline
#         for y in range(int(dimension[0]*3/4), dimension[0]):
#             cv.circle(blank, (midpoints[y], y), 1, (255, 255, 255), 1)  

#         # Unwarp this
#         midline_fr = cv.warpPerspective(blank, self.inv_warp, dimension[:2][::-1])
#         midline_bnw = thresholding(midline_fr)
#         lines = cv.HoughLinesP(midline_bnw, 1, np.pi / 180, 50, minLineLength = 5)
        
#         # Detect slope
#         slopes = np.array([]) 
#         if lines is not None:
#             for line in lines:
#                 for x1, y1, x2, y2 in line:
#                     slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
#                     slopes = np.append(slopes, slope)
                    
        slopes = np.array([])
        dimension = img.shape
        
        # Calculate slopes of every 2 consecutive points
        for y in range(int(dimension[0]*3/4), dimension[0]):
            slope = 1 / (midpoints[y] - midpoints[y-1]) if (midpoints[y] - midpoints[y-1]) != 0 else 0
            slopes = np.append(slopes, slope)
            
        return np.average(slopes)
    

    def project(self, car_img, bird_img, kl, kr, midpoints, color = (0, 255, 0)):
        """
        Project the bird-view lane image into the car-view's lane image
        
        Params:
            car_img (np.ndarray): the car-view lane image
            bird_img (np.ndarray): the bird-view lane image
            kl (np.ndarray): left fit curve
            kr (np.ndarray): right fit curve
            color (int, int, int): the detected lane's color, in RGB color map
        
        Returns:
            result (np.ndarray): the projected image
        """
        # Create an empty image
        z = np.zeros_like(bird_img)
        bird_img = np.dstack((z, z, z))

        # Determine the same points between images
        h = bird_img.shape[0]
        ys = np.linspace(0, h - 1, h)
        lxs = kl[0] * (ys**2) + kl[1]* ys +  kl[2]
        rxs = kr[0] * (ys**2) + kr[1]* ys +  kr[2]
    
        pts_left = np.array([np.transpose(np.vstack([lxs, ys]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rxs, ys])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Fill in the empty image
        cv.fillPoly(bird_img, np.int_(pts), color)
        
        # Draw midline
        for y in range(bird_img.shape[0]):
            cv.circle(bird_img, (midpoints[y], y), 1, [255,255,255], 1)    
        
        # Warp bird view to car view
        dimension = bird_img.shape[:2][::-1]
        car_lane = cv.warpPerspective(bird_img, self.inv_warp, dimension)
        
        # Blend two images
        result = cv.addWeighted(car_img, 1, car_lane, 0.3, 0)
        return result
    
def main():
    """
    Testing function
    """
    # Read image
    img = cv.imread("./Lane/view.jpg")
    
    # Undistort image
    calib = Calibration(img)
    undistort = calib.undistort(img)

    # Left-top, left-bottom, right-bottom, right-top
    source_points = np.array([(640, 500), (20, 1700), (1950, 1700), (1580, 500)], np.float32)
    destination_points = np.array([(500, 0), (500, 1700), (1900, 1700), (1900, 0)], np.float32)

    # Convert to bird view img
    bird = BirdView(source_points, destination_points)
    img_bird = bird.bird_view(undistort)
    
    # Write image
    cv.imwrite("./Outputs/bird_view.jpg", img_bird)

if __name__ == "__main__":
    main()