from darknet import load_net, load_meta, detect
import cv2 # don't compile darknet with opencv, conflicts will cause seg fault
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

# Todo: display labels and confidence level

class Object:
    def __init__(self, detection_output, frame_seq = 0):
        self.label = self.extract_label(detection_output)
        self.box_top_left, self.box_bottom_right = self.extract_box_corners(detection_output)
        self.confidence = self.extract_confidence(detection_output)
        self.frame_seq = frame_seq

    def extract_label(self, detection_output):
        return detection_output[0]

    def extract_confidence(self,detection_output):
        return detection_output[1]

    def extract_box_corners(self, detection_output):
        x = detection_output[2][0]
        y = detection_output[2][1]
        w = detection_output[2][2]
        h = detection_output[2][3]
        return (int(x - w/2),int(y - h/2)), (int(x + w/2),int(y + h/2))

def pipeline(frame):
    return frame

def draw_boxes(img, object, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)

    # Draw a rectangle given bbox coordinates
    left_up_corner = object.box_top_left
    bottom_right_corner = object.box_bottom_right
    cv2.rectangle(imcopy, left_up_corner, bottom_right_corner, color, thick)

    # Return the image copy with boxes drawn
    return imcopy

if __name__ == "__main__":
    # os.chdir("/home/sirius/Desktop/darknet/")
    net = load_net(b"darknet/cfg/yolo.cfg", b"weights/yolo.weights", 0)
    meta = load_meta(b"darknet/cfg/coco.data")
    r = detect(net, meta, b"darknet/data/dog.jpg")

    print (r[0][0]) # label
    print (r[0][1]) # confidence
    print (r[0][2]) # box coordinates
    print (r[0][2][0]) # box coordinates (x,y,w,h)

    print (r[1][0]) # label
    print (r[1][1]) # confidence
    print (r[1][2]) # box coordinates
    #im = cv2.imread("data/dog.jpg")
    im = mpimg.imread("darknet/data/dog.jpg")
    for detection in r:
        ob = Object(detection)
        im = draw_boxes(im, ob)
    
    plt.imshow(im)
    plt.show()

    r = detect(net, meta, b"darknet/data/scream.jpg")
    print (r)

    r = detect(net, meta, b"darknet/data/eagle.jpg")
    print (r)
