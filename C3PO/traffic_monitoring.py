from darknet import load_net, load_meta, detect
from Counter import Counter
from Object import Object
from Frame import Frame, draw_box
import cv2 # don't compile darknet with opencv, conflicts will cause seg fault
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque
import heapq as heapq
from heapq import *

from moviepy.editor import VideoFileClip
#from IPython.display import HTML


ct = Counter()
history = deque(maxlen = 1)
screen = [] # keep track of unique objects within the screen

p = 0
c = 0
b = 0

# Todo: display labels and confidence level

class Link:
    def __init__(self, src, end):
        self.src = src
        self.end = end
        self.distance = square_distance(src,end)


def detect_crossing_with_history(current_frame, last_frame):
    global history, p, c, b
    obs = [] 
    for ob in last_frame.objects:
        obs.append(ob)

    for i,ob in enumerate(current_frame.objects):
        if len(obs) > 0:
            heap = [(square_distance(ob.origin, x.origin), x) for x in obs]
            heapq.heapify(heap)
            cloest_neighbor = heappop(heap)[1]
            while(cloest_neighbor.label != ob.label and len(heap) > 0):
                cloest_neighbor = heappop(heap)[1]

            current_frame.objects[i].trajectory = cloest_neighbor.trajectory
            if cloest_neighbor.label == ob.label:
                current_frame.objects[i].trajectory.append(cloest_neighbor.origin)
                obs.remove(cloest_neighbor)
            else:
                current_frame.objects[i].trajectory.append(ob.origin) # no matching object from last frame, append itself
        else:
            # if the current frame has more objects than last frame, 
            # all the extra objects will append their own origins to the their own trajectories
            current_frame.objects[i].trajectory.append(ob.origin)           

    if len(obs) > 0: # last frame has more objects
        print("left over happens, current_frame # objs:", len(current_frame.objects))
        for ob in obs:
            ob.trajectory.append(ob.origin) # append its own origin to the trajectory
            current_frame.objects.append(ob)

    print("*********************current_frame # objs:", len(current_frame.objects))
    for i,ob in enumerate(current_frame.objects):
        print()
        print(i)
        print(ob.trajectory)
        print()
        if (len(ob.trajectory) - len(set(ob.trajectory)) > 6 ): # 6 repeating points
            # considered moving out of frame
            if ob.label == "person":
                p +=1
            elif ob.label == "truck" or ob.label == "car":
                c += 1
            elif ob.label == "bike" or ob.label == "motorcycle" or ob.label == "bicycle":
                b += 1
            current_frame.objects.pop(i) # destroy itself

    print("p:", p, "c:", c, "b:", b)


def square_distance(p1, p2):
    return (p1[0] - p2[0])^2 + (p1[1] - p2[1])^2

def pipeline(frame):
    global ct, history
    fm = Frame(frame)
    mpimg.imsave("current_frame.jpg", frame)
    results = detect(net, meta, b"current_frame.jpg")

    for detection in results:
        ob = Object(detection)
        fm.add_object(ob)

    fm.draw_boxes()
    fm.add_labels()

    cv2.line(fm.box_img, (0,420), (700,420), (255,0,0))
    cv2.line(fm.box_img, (0,380), (700,380), (255,0,0))

    fm.detect_crossing(ct)

    if len(history) > 0 :
        detect_crossing_with_history(fm, history.pop())

    fm.draw_trajectory()

    history.append(fm)

    return fm.box_img

if __name__ == "__main__":
    # os.chdir("/home/sirius/Desktop/darknet/")
    net = load_net(b"darknet/cfg/yolo.cfg", b"weights/yolo.weights", 0)
    meta = load_meta(b"darknet/cfg/coco.data")
    # r = detect(net, meta, b"darknet/data/dog.jpg")

    # print (r[0][0]) # label
    # print (r[0][1]) # confidence
    # print (r[0][2]) # box coordinates
    # print (r[0][2][0]) # box coordinates (x,y,w,h)

    # print (r[1][0]) # label
    # print (r[1][1]) # confidence
    # print (r[1][2]) # box coordinates
    # #im = cv2.imread("data/dog.jpg")
    # im = mpimg.imread("darknet/data/dog.jpg")
    # fm = Frame(im)
    # for detection in r:
    #     ob = Object(detection)
    #     fm.add_object(ob)

    # fm.draw_boxes()
    # fm.add_labels()
    
    # plt.imshow(fm.box_img)
    # plt.show()

    # r = detect(net, meta, b"darknet/data/scream.jpg")
    # print (r)

    # r = detect(net, meta, b"darknet/data/eagle.jpg")
    # print (r)

    output = 'project_output.mp4'
    clip_i = VideoFileClip("video.mp4")
    clip = clip_i.fl_image(pipeline)
    clip.write_videofile(output, audio=False)

    ct.stats_terminal_report()
