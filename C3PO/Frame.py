import cv2 # don't compile darknet with opencv, conflicts will cause seg fault
import numpy as np
import time, threading

def clear_grid_record(grid_matrix,r,c):
    grid_matrix[r][c] = 0

def draw_box(img, object, color=(0, 255, 0), thick=4):
    # Make a copy of the image
    imcopy = np.copy(img)

    # Draw a rectangle given bbox coordinates
    left_up_corner = object.box_top_left
    bottom_right_corner = object.box_bottom_right
    cv2.rectangle(imcopy, left_up_corner, bottom_right_corner, color, thick)

    # Return the image copy with boxes drawn
    return imcopy

def img_coord_to_grid_index(img_r,img_c, dim, grid_step):
    r_dist = int(dim[1] / grid_step);
    c_dist = int(dim[0] / grid_step);
    return (min(grid_step,int(img_r / r_dist)), min(grid_step,int(img_c / c_dist)))

def grid_index_img_coord(grid_r,grid_c,dim,grid_step):
    r_dist = int(dim[1] / grid_step);
    c_dist = int(dim[0] / grid_step);

    result = np.array([[grid_r*r_dist,grid_c*c_dist],
                    [(grid_r+1)*r_dist,grid_c*c_dist], 
                    [(grid_r+1)*r_dist,(grid_c+1)*c_dist],
                    [grid_r*r_dist,(grid_c+1)*c_dist]], dtype = np.int32)
    return [result]

class Frame:
    def __init__(self, im, g_corners, ROI_MASK, grid_matrix, grid_step, frame_seq = 0):
        self.img = im
        self.box_img = im
        self.objects = []
        self.frame_seq = frame_seq
        self.dim = im.shape
        self.grid_step = grid_step
        self.grid_matrix = grid_matrix
        self.np_corners = np.array(g_corners, dtype = np.int32)
        self.ROI_MASK = ROI_MASK
        self.NROI_MASK = ROI_MASK

    def add_object(self, object):
        ob = object
        # filter can be applied here based on label
        if self.ROI_MASK[object.origin[1]][object.origin[0]] == 255:
            # HIT ROI
            g_index = img_coord_to_grid_index(object.origin[0],object.origin[1],self.dim,self.grid_step)

            if ob.label == "person":
                self.grid_matrix[g_index[0]][g_index[1]] = 1
            elif ob.label == "truck":
                self.grid_matrix[g_index[0]][g_index[1]] = 1
            elif ob.label == "car":
                self.grid_matrix[g_index[0]][g_index[1]] = 1
            elif ob.label == "bicycle":
                self.grid_matrix[g_index[0]][g_index[1]] = 1
            elif ob.label == "motorcycle":
                self.grid_matrix[g_index[0]][g_index[1]] = 1
            elif ob.label == "bike":
                self.grid_matrix[g_index[0]][g_index[1]] = 1

            threading.Timer(10, clear_grid_record, [self.grid_matrix, g_index[0], g_index[1]]).start()
        else:
            # HIT NROI
            g_index = img_coord_to_grid_index(object.origin[0],object.origin[1],self.dim,self.grid_step)
            if ob.label == "person":
                self.grid_matrix[g_index[0]][g_index[1]] = 2
            elif ob.label == "truck":
                self.grid_matrix[g_index[0]][g_index[1]] = 2
            elif ob.label == "car":
                self.grid_matrix[g_index[0]][g_index[1]] = 2
            elif ob.label == "bicycle":
                self.grid_matrix[g_index[0]][g_index[1]] = 2
            elif ob.label == "motorcycle":
                self.grid_matrix[g_index[0]][g_index[1]] = 2
            elif ob.label == "bike":
                self.grid_matrix[g_index[0]][g_index[1]] = 2
            threading.Timer(10, clear_grid_record, [self.grid_matrix, g_index[0], g_index[1]]).start()


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


    def draw_grid(self):
        width = int(self.dim[1])
        height = int(self.dim[0])

        w_dist = int(width/self.grid_step)
        h_dist = int(height/self.grid_step)

        for h in range(0, height, h_dist):
            cv2.line(self.box_img,(0,h),(width,h),(0,0,0));

        for w in range(0, width, w_dist):
            cv2.line(self.box_img,(w,0),(w,height),(0,0,0));

    def color_grid(self):
        # self.grid_matrix[0][0] = 1
        # self.grid_matrix[1][1] = 1
        # self.grid_matrix[2][2] = 1
        # self.grid_matrix[3][3] = 1
        # self.grid_matrix[4][4] = 1
        # self.grid_matrix[5][5] = 1
        # self.grid_matrix[6][6] = 1
        # self.grid_matrix[10][10] = 1
        # self.grid_matrix[11][11] = 1

        mask = np.zeros_like(self.img, dtype=np.uint8)
        for r in range(0, self.grid_matrix.shape[0]):
            for c in range(0, self.grid_matrix.shape[1]):
                if self.grid_matrix[r][c] == 1:
                    cv2.fillPoly(mask, grid_index_img_coord(r,c,self.dim,self.grid_step), (0,255,0))
                elif self.grid_matrix[r][c] == 2:
                    cv2.fillPoly(mask, grid_index_img_coord(r,c,self.dim,self.grid_step), (255,0,0))
                elif self.grid_matrix[r][c] == 3:
                    cv2.fillPoly(mask, grid_index_img_coord(r,c,self.dim,self.grid_step), (0,0,255))

        self.box_img = cv2.addWeighted(self.box_img, 1, mask, 0.3, 0)

    def color_NROI(self):
        mask = np.ones_like(self.img, dtype=np.uint8)*255
        cv2.fillPoly(mask, [self.np_corners], (0,0,0))
        self.box_img = cv2.addWeighted(self.box_img, 1, mask, 0.3, 0)
        self.NROI_MASK = mask[:,:,0]



    def detect_crossing(self, ct):

        display_text =  ct.stats()
        cv2.putText(self.box_img, display_text, (40,15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

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
