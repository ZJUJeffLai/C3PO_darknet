from collections import deque
import numpy as np
class Object:
    def __init__(self, detection_output):
        self.label = self.extract_label(detection_output)
        self.origin, self.box_top_left, self.box_bottom_right = self.extract_box_corners(detection_output)
        self.confidence = self.extract_confidence(detection_output)
        self.trajectory = deque(maxlen = 8) 
        self.trajectory.append(self.origin)
        self.counted = False
        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,255,255), (0,0,0)]
        self.traj_color = self.colors[np.random.randint(0,100) % 7]

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

    def trajectory_tracking(self, point):
        self.trajectory.append(point)

    def __lt__(self, other):
        if isinstance(other, Object):
            return 1 # don't care which one is larger

        return NotImplemented