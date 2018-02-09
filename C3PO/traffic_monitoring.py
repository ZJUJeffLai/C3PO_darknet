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
import pickle

from moviepy.editor import VideoFileClip
#from IPython.display import HTML


ct = Counter()
history = deque(maxlen = 1)
screen = [] # keep track of unique objects within the screen

DEFINE_NEW_ROI = 0
NUM_OF_CORNERS = 5
CORNER_VALID = 0
MASK_VALID = 0
ROI_MASK = []
g_corners = []
g_polygon_num = 0
GRID_MATRIX_INIT = 0
grid_matrix = []
grid_step = 100

p = 0
c = 0
b = 0


'''
Brief
    This function will load cached user_defined_corners pickle file
Input
    None
Return
    user_defined_corners corners
'''
def load_user_defined_corners():
    user_defined_corners = pickle.load(open("cache/user_defined_corners.p", "rb" ))
    return user_defined_corners["corners"]

'''
Brief
    This function will handle user clicks on a opened cv2 image
Input
    event: event handler
    x: x coordinate
    y: y coordinate
    flags: flags (not in use)
    param: param (not in use)
Return
    None
'''
def clicks(event, x, y, flags, param):
    global g_corners, g_polygon_num
    if event == cv2.EVENT_LBUTTONDOWN:
        if g_polygon_num < NUM_OF_CORNERS:
            print([x,y])
            g_corners.append([x,y])
        g_polygon_num = g_polygon_num + 1

'''
Brief
    This function will let user to specify 4 corners to define a polygon area.
It's assumed that user will follow the order of top left, bottom left, bottom
right, and top right (counter-clockwise fashion).
Input
    img: input image
Return
    The user defined corners
'''
def get_corners(img):
    global g_corners, g_polygon_num
    # init
    g_corners = []
    g_polygon_num = 0

    # clone the image, and setup the mouse callback function
    clone = img.copy()
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", clicks)

    # keep looping until the 'c' key  is pressed or more than 4 corners are defined
    while True:
        # display the image and wait for a keypress
        if g_polygon_num < NUM_OF_CORNERS + 1:
            for i in range(0,g_polygon_num - 1):
                # Draw a red line with thickness of 5 px
                cv2.line(clone,(g_corners[i][0],g_corners[i][1]),(g_corners[i+1][0],g_corners[i+1][1]),(0,0,255),15)
        # fill the last line at fifth click
        if g_polygon_num >= NUM_OF_CORNERS + 1:
            cv2.line(clone,(g_corners[NUM_OF_CORNERS-1][0],g_corners[NUM_OF_CORNERS-1][1]),(g_corners[0][0],g_corners[0][1]),(0,0,255),5)

        cv2.imshow("img", clone)
        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord("c") or g_polygon_num > NUM_OF_CORNERS + 1:
            cv2.destroyAllWindows()
            break

    user_defined_corners = {}
    user_defined_corners["corners"] = g_corners
    pickle.dump(user_defined_corners, open( "cache/user_defined_corners.p", "wb" ))

    # minor correction for improved accuracy
    # leveling two points at the same height
    # g_corners[1][1] = g_corners[2][1]
    # g_corners[0][1] = g_corners[3][1]
    print("User defined corners: ", g_corners)

    return g_corners

# class Link:
#     def __init__(self, src, end):
#         self.src = src
#         self.end = end
#         self.distance = square_distance(src,end)


def detect_crossing_with_history(current_frame, last_frame):
    global history, p, c, b

    # obs = []
    # for ob in last_frame.objects:
    #     obs.append(ob)  
    obs = last_frame.objects 

    for i,ob in enumerate(current_frame.objects):
        if len(obs) > 0:
            heap = [(square_distance(ob.origin, x.origin), x) for x in obs]
            heapq.heapify(heap)
            before = len(obs)
            cloest_neighbor = heappop(heap)[1]
            after = len(obs)
            assert before == after
            while(cloest_neighbor.label != ob.label and len(heap) > 0):
                before = len(obs)
                cloest_neighbor = heappop(heap)[1]
                after = len(obs)
                assert before == after

            if cloest_neighbor.label == ob.label and square_distance(ob.origin, cloest_neighbor.origin) < 100:
                current_frame.objects[i].traj_color = cloest_neighbor.traj_color
                current_frame.objects[i].counted = cloest_neighbor.counted
                current_frame.objects[i].trajectory = cloest_neighbor.trajectory
                current_frame.objects[i].trajectory.append(cloest_neighbor.origin)
                #print(square_distance(ob.origin, cloest_neighbor.origin))
                cv2.line(current_frame.box_img,cloest_neighbor.origin,ob.origin,ob.traj_color, 5);
                before = len(obs)
                obs.remove(cloest_neighbor)
                after = len(obs)
                assert before == after + 1

            #    current_frame.objects[i].trajectory.append(ob.origin) # no matching object from last frame, append itself
            #    already handled by the Object constructor
        #else:
            # if the current frame has more objects than last frame, 
            # all the extra objects will append their own origins to the their own trajectories
            # current_frame.objects[i].trajectory.append(ob.origin)
            # do nothing, Object constructor already handle this. A new object will append its own origin into its trajectory       

    delta = 0
    before = len(current_frame.objects)
    for ob in obs:
        if len(ob.trajectory) > 0 :
            delta +=1
            ob.trajectory.popleft() # drop its oldest point
            ob.measurements.popleft() # drop its oldest point
            current_frame.objects.append(ob)
    after = len(current_frame.objects)
    assert after - before == delta
    del obs[:]           

    #print("*********************current_frame # objs:", len(current_frame.objects))
    for i,ob in enumerate(current_frame.objects):
        #print()
        #print(i)
        #print(ob.trajectory)
        #print()
        if (not ob.counted and len(ob.trajectory) > 5):
            # considered moving out of frame
            if ob.label == "person":
                ct.add_pedestrian()
            elif ob.label == "truck" or ob.label == "car":
                ct.add_vehicle()
            elif ob.label == "bike" or ob.label == "motorcycle" or ob.label == "bicycle":
                ct.add_bike()
            ob.counted = True

    #print("p:", p, "c:", c, "b:", b)
    assert len(last_frame.objects) == 0
    assert len(obs) == 0
    #print ("Last frame objects: " + str(len(last_frame.objects))+ "obs objects: "+ str(len(obs)) +"This frame objects: " + str(len(current_frame.objects)))


def square_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def pipeline(frame):
    global ct, history, DEFINE_NEW_ROI, CORNER_VALID, g_corners, ROI_MASK, MASK_VALID, GRID_MATRIX_INIT, grid_matrix, frame_seq
    if CORNER_VALID == 0 and DEFINE_NEW_ROI:
        get_corners(frame)
        CORNER_VALID = 1
    elif CORNER_VALID == 0:
        g_corners = load_user_defined_corners()
        CORNER_VALID = 1
    
    if MASK_VALID == 0:
        ROI_MASK = np.zeros_like(frame[:,:,0], dtype=np.uint8)
        np_corners = np.array(g_corners, dtype = np.int32)
        cv2.fillPoly(ROI_MASK, [np_corners], 255)
        MASK_VALID = 1

    if GRID_MATRIX_INIT == 0:
        grid_matrix = np.zeros((int(grid_step) + 1, int(grid_step) + 1))
        GRID_MATRIX_INIT = 1

    #print(ROI_MASK.shape)

    fm = Frame(frame, g_corners, ROI_MASK, grid_matrix, grid_step, frame_seq)

    mpimg.imsave("current_frame.jpg", frame)
    results = detect(net, meta, b"current_frame.jpg")

    #print(fm.img.shape)

    for detection in results:
        ob = Object(detection, frame_seq)
        fm.add_object(ob)

    # update frame number
    frame_seq += 1
    if frame_seq > FRAME_SQE_MAX:
        frame_seq = 0

    fm.draw_boxes()
    fm.add_labels()
    fm.draw_grid()
    fm.color_grid()
    fm.color_NROI()

    # cv2.line(fm.box_img, (0,420), (700,420), (255,0,0))
    # cv2.line(fm.box_img, (0,380), (700,380), (255,0,0))

    fm.detect_crossing(ct)

    if len(history) > 0 :
        detect_crossing_with_history(fm, history.pop())

    #print(len(fm.objects))

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
    # fm.draw_grid()
    # fm.color_grid()
    
    # plt.imshow(fm.box_img)
    # plt.show()

    # r = detect(net, meta, b"darknet/data/scream.jpg")
    # print (r)

    # r = detect(net, meta, b"darknet/data/eagle.jpg")
    # print (r)

    output = 'ch01.mp4'
    clip_i = VideoFileClip("cut_ch01.mp4")
    clip = clip_i.fl_image(pipeline)
    clip.write_videofile(output, audio=False)

    ct.stats_terminal_report()
