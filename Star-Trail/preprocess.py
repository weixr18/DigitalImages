import cv2
import numpy as np
import matplotlib.pyplot as plt
import rawpy

ROWS = 3000
COLS = 1000
CHANNELS = 3


if __name__ == "__main__":
    """
    raw = rawpy.imread('raw/DSC08084.ARW')
    rgb = raw.postprocess()
    plt.imshow(rgb)
    plt.show()
    """

    img = cv2.imread("raw/DSC08090.JPG")
    plt.imshow(img)
    plt.show()

    pass
