from collections import deque
import numpy as np
import Kalman_filter as km
class Object:
    def __init__(self, detection_output, frame_seq):
        #print("object constructor")
        self.label = self.extract_label(detection_output)
        self.origin, self.box_top_left, self.box_bottom_right = self.extract_box_corners(detection_output)
        self.confidence = self.extract_confidence(detection_output)
        self.measurements = deque(maxlen = 30) 
        self.measurements.append(self.origin)
        self.trajectory = deque(maxlen = 30)
        self.trajectory.append(self.origin)
        self.counted = False
        self.colors = [(255,0,0), (0,0,255), (255,255,0), (0,255,255), (255,255,255), (0,0,0)] # (0,255,0), randint(0,100) % 7
        self.traj_color = self.colors[np.random.randint(0,100) % 6]
        self.Kalman_filter = km.Kalman_filter(self.origin[0], self.origin[1], frame_seq)

    def extract_label(self, detection_output):
        return bytes.decode(detection_output[0])

    def extract_confidence(self,detection_output):
        return detection_output[1]

    def extract_box_corners(self, detection_output):
        x = detection_output[2][0]
        y = detection_output[2][1]
        w = detection_output[2][2]
        h = detection_output[2][3]
        return (int(x), int(y)), (int(x - w/2),int(y - h/2)), (int(x + w/2),int(y + h/2))

    def process_measurement(self, origin,frame_seq):
        self.Kalman_filter.ProcessMeasurement(np.array(origin), frame_seq) # px, py
        self.trajectory.append(self.Kalman_filter.CurrentEstPos())
        assert len(self.trajectory) == len(self.measurements)

    def __lt__(self, other):
        if isinstance(other, Object):
            return 1 # don't care which one is larger

        return NotImplemented

    #def __del__(self):
        #print (self.label + " died")