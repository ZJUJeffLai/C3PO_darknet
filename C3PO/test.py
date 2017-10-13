import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

im = cv2.imread("data/scream.jpg")
# im = mpimg.imread("data/dog.jpg")
#new_img = draw_boxes(im, r[0][2])
plt.imshow(im)
plt.show()