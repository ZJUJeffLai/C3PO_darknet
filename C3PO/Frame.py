import cv2 # don't compile darknet with opencv, conflicts will cause seg fault
import numpy as np

def draw_box(img, object, color=(0, 255, 0), thick=4):
    # Make a copy of the image
    imcopy = np.copy(img)

    # Draw a rectangle given bbox coordinates
    left_up_corner = object.box_top_left
    bottom_right_corner = object.box_bottom_right
    cv2.rectangle(imcopy, left_up_corner, bottom_right_corner, color, thick)

    # Return the image copy with boxes drawn
    return imcopy

class Frame:
    def __init__(self, im, frame_seq = 0):
        self.img = im
        self.box_img = im
        self.objects = []
        self.frame_seq = frame_seq

    def add_object(self, object):
        # filter can be applied here based on label
        self.objects.append(object)


    def draw_boxes(self):
        for ob in self.objects:
            self.box_img = draw_box(self.box_img, ob)

    def add_labels(self):
        for ob in self.objects:
            self.box_img = cv2.putText(self.box_img, ob.label, ob.box_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    def draw_boxes_labels(self):
        for ob in self.objects:
            self.box_img = draw_box(self.box_img, ob)
            self.box_img = cv2.putText(self.box_img, ob.label, ob.box_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    def draw_trajectory(self):
        for ob in self.objects:
            for pt in ob.trajectory:
                self.box_img = cv2.circle(self.box_img, (pt[0], pt[1]), 3, ob.traj_color, -1)
            

    def detect_crossing(self, ct):

        display_text =  ct.stats()
        cv2.putText(self.box_img, display_text, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

        for ob in self.objects:
            if ob.origin[1] < 380:
                continue

            if ob.origin[1] > 420:
                continue

            if ob.label == "person":
                ct.add_pedestrian()
            elif ob.label == "truck":
                ct.add_vehicle()
            elif ob.label == "car":
                ct.add_vehicle()
            elif ob.label == "bicycle":
                ct.add_bike()
            elif ob.label == "motorcycle":
                ct.add_bike()
            elif ob.label == "bike":
                ct.add_bike()
