# Import libraries
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import glob
import yaml

class Calibration:
    def __init__(self, iframe, chess_type = '.jpg', nx = 9, ny = 6): 
        """
        Create a new instance of the Calibration class accordingly to the input
        
        Params:
            iframe (np.ndarray): the input image frame
            img_type (str): the type of input images
            nx (int): # of rows of the printed chessboard
            ny (int): # of cols of the printed chessboard
        """
        # Locate the optional chessboard references
        optimal_reso = self.detect_opt_dimension(iframe)
        self.folder = "./Chessboard/%dx%d/" % (optimal_reso[0], optimal_reso[1])
        self.chess_type = chess_type
        
        # Specify the # of rows and cols of the printed chessboard
        self.nx, self.ny = nx, ny

        # Object points and image points from all the images
        self.objpoints, self.imgpoints, self.temp_img = self.read_imgs()

        # Image dimensions
        self.dimensions = self.temp_img.shape[::-1]
        
        # Camera's parameters
        self.mtx, self.dist, self.rvecs, self.tvecs = self.calibrate()
        
        
    def detect_opt_dimension(self, img) -> [int, int]:
        """
        Choose the optimal resolution accordingly to the input frame's dimension

        Params:
            img (np.ndarray): the input image

        Returns:
            optimal ([int, int]): the resolution's width and height, respectively
        """

        dim = img.shape[:2][::-1]
        p360, p720 = np.array([640, 360]), np.array([1280, 720])

        # Choose the one with lowest difference
        optimal = p360 if np.sum(np.abs(p360 - dim)) < np.sum(np.abs(p720 - dim)) else p720

        return optimal

        
    def read_imgs(self) -> (list, list, np.ndarray):
        """
        Iterate through all images and add object points to the images

        Returns:
            objpoints (list): 3D points in real world
            imgpoints (list): 2D points in image plane
            img (np.ndarray): the last image
        """
        # Specify the input and output
        images = glob.glob(self.folder + "*" + self.chess_type)
        objpoints = []
        imgpoints = []

        # Prepare object points
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)
        
        # Termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for fname in images:
            # Read image as grayscale
            img = cv.cvtColor(cv.imread(fname), cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(img, (self.nx, self.ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                cv.cornerSubPix(img, corners, (11,11), (-1,-1), criteria) # Refine corner
                imgpoints.append(corners)

        return objpoints, imgpoints, img

    
    def calibrate(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Calibrate the camera

        Returns:
            mtx (np.ndarray): camera's intrinsic matrix
            dist (np.ndarray): distortion coefficients
            rs (np.ndarray): rotation matrices
            ts (np.ndarray): translation matrices
        """
        _, mtx, dist, rs, ts = cv.calibrateCamera(self.objpoints, self.imgpoints, 
                                                  self.dimensions, None, None)

        return mtx, dist, rs, ts
    
    
    def undistort(self, iframe) -> np.ndarray:
        """
        Undistort a frame accordingly to its resolution

        Params:
            iframe (np.ndarray): the frame/image to be undistorted

        Returns:
            dst (np.adarray): the undistorted image
            [h, w] ([int, int]): frame's new resolution (height, width)
        """
        # Refine the camera matrix
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, self.dimensions, 1, self.dimensions)
        
        # Undistort the image
        dst = cv.undistort(iframe, self.mtx, self.dist, None, newcameramtx)
        
        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        return dst, [h, w]
    
    
    def save_undistort(self, iframe, name, **kwargs):
        """
        Save the undistorted version of an image

        Params:
            iframe (np.ndarray): the frame/image to be undistorted
            name (str): output image's name
            (Optional) out_folder (str): relative location of the output image
            (Optional) img_type (str): type of the input image
            (Optional) chess_type (str): type of chessboard images
            (Optional) export (bool): if True, export camera's parameters

        Side-effects:
            Save the undistort image with the same name to the pre-specified folder 
            Export the camera's parameters to the main folder
        """
        # Get optional arguments
        out_folder = kwargs.get("out_folder", './Outputs/')
        img_type = kwargs.get("img_type", ".jpg")
        chess_type = kwargs.get("chess_type", ".jpg")
        export = kwargs.get("export", False)

        # Undistort the input image and save it
        dst, _ = self.undistort(iframe)
        cv.imwrite(out_folder + "undistort_" + name + img_type, dst)

        # Check condition for exporting camera's parameters
        if export:
            self.export_params()
            
    def get_mtx_and_dist(self) -> (np.ndarray, np.ndarray):
        """
        Returns the camera's matrix and distortion coefficients
        """
        return self.mtx, self.dist
        
    
    def focal_length(self, sensor_width = 3.58) -> float:
        """
        Extract the camera's focal length (mm) knowing its sensor width
        
        Params:
            sensor_width (float): the sensor's width
        
        Returns:
            (float): the mean value of the calculated focal length
        """
        # Extract focal length from camera's matrix
        focal_len = np.array([self.mtx[0][0], self.mtx[1][1]])

        # Convert from pixels to mm
        focal_len_mm = focal_len * sensor_width / self.dimensions[0] 
        
        return np.mean(focal_len_mm)
        
        
    def total_error(self) -> float:
        """
        Calculate the total error of our calibration

        Returns:
            (float): the total error of the calibration
        """
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        return mean_error / len(self.objpoints)
    

    def export_params(self, filename = "cam"):
        """
        Save important parameters of the camera
        
        Params:
            filename (str): name of the output file

        Side-effects:
            Write a new file with camera's parameters to the default folder

        """
        # Specify the content to be written
        data = {'camera_matrix': np.asarray(self.mtx).tolist(),
                'dist_coeff': np.asarray(self.dist).tolist()}

        # Write all parameter values to the output file
        with open("./Outputs/" + filename + ".yaml", "w") as f:
            yaml.dump(data, f)
        
        
def main(name, img_type = '.jpg', folder = './', export_params = False):
    """
    Save the undistorted version of the input image and probably the corresponding undistorted chessboard images
    
    Params:
        name (str): the filename to be undistorted
        img_type (str): type of that file
        folder (str): relative location of that file corresponding to the main folder
        export_params (bool): if True, export camera's parameters
        
    Side-effects:
        Save the undistort image with the same name to the pre-specified folder 
    """    
    # Read images
    img = cv.imread(folder + name + img_type)
    cali = Calibration(img)
    
    # Undistort and save that image
    cali.save_undistort(img, name, img_type = img_type, export = export_params)
    
    
if __name__ == "__main__":
    main(name = "view", folder = "./Lane/", export_params = True)
    main(name = "cali3", folder = "./Chessboard/1280x720/")