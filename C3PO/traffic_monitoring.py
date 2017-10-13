from darknet import load_net, load_meta, detect
import cv2 # don't compile darknet with opencv, conflicts will cause seg fault
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

def pipeline(frame):
    return frame

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    print( "draw boxes" ) 
    # Make a copy of the image
    imcopy = np.copy(img)

    # Draw a rectangle given bbox coordinates
    left_up_corner = (int(bboxes[0] - bboxes[2]/2),int(bboxes[1] - bboxes[3]/2))
    bottom_right_corner = (int(bboxes[0] + bboxes[2]/2),int(bboxes[1] + bboxes[3]/2))
    cv2.rectangle(imcopy, left_up_corner, bottom_right_corner, color, thick)

    # Return the image copy with boxes drawn
    return imcopy

if __name__ == "__main__":
    # os.chdir("/home/sirius/Desktop/darknet/")
    net = load_net(b"cfg/yolo.cfg", b"weights/yolo.weights", 0)
    meta = load_meta(b"cfg/coco.data")
    r = detect(net, meta, b"data/dog.jpg")

    print (r[0][0]) # label
    print (r[0][1]) # confidence
    print (r[0][2]) # box coordinates
    print (r[0][2][0]) # box coordinates (x,y,w,h)

    print (r[1][0]) # label
    print (r[1][1]) # confidence
    print (r[1][2]) # box coordinates
    #im = cv2.imread("data/dog.jpg")
    im = mpimg.imread("data/dog.jpg")
    for detection in r:
        im = draw_boxes(im, detection[2])
    
    plt.imshow(im)
    plt.show()

    r = detect(net, meta, b"data/scream.jpg")
    print (r)

    r = detect(net, meta, b"data/eagle.jpg")
    print (r)
