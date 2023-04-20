import cv2 as cv
import numpy as np
import os

input_path = "./Chessboard/1280x720"
output_path = "./Chessboard/640x360"
new_size = (640, 360)

def resize():
    for file in os.listdir(input_path):
        img = cv.imread(os.path.join(input_path, file))
        new_img = cv.resize(img, new_size)
        cv.imwrite(os.path.join(output_path, file), new_img)
    return

if __name__ == "__main__":
    resize()
