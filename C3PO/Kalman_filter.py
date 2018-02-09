import numpy as np
import time
#from traffic_monitoring import FRAME_SQE_MAX

FRAME_SQE_MAX = 1000

class Kalman_filter:
    def __init__(self, px, py, frame_seq):
        self.x_ = np.array([px,py,1,1])

        self.P_ = np.array([[1,0,0,0],
                            [0,1,0,0],
                            [0,0,1000,0],
                            [0,0,0,1000]])

        self.F_ = np.array([[1,0,1,0],
                            [0,1,0,1],
                            [0,0,1,0],
                            [0,0,0,1]])

        self.Q_ = np.array([[1,0,1,0],
                            [0,1,0,1],
                            [1,0,1,0],
                            [0,1,0,1]])

        self.H_ = np.array([[1,0,0,0],
                            [0,1,0,0]])

        self.R_ = np.array([[0.02, 0],
                            [0, 0.02]])

        # this time depends on the frame update rate, which is not the actual time elapsed in the real world
        self.t_ = frame_seq

    def Predict(self):
        self.x_ = self.F_.dot(self.x_)
        self.P_ = (self.F_.dot(self.P_)).dot(self.F_.T) + self.Q_;

    def Update(self, z):
        z_pred = self.H_.dot(self.x_)
        y = z - z_pred
        Ht = self.H_.T
        S = (self.H_.dot(self.P_)).dot(Ht) + self.R_
        Si = np.linalg.inv(S)
        K = (self.P_.dot(Ht)).dot(Si)

        # new estimate
        self.x_ = self.x_ + (K.dot(y))
        self.P_ -= (K.dot(self.H_)).dot(self.P_)

    def ProcessMeasurement(self, z, frame_seq):
        noise_ax = 9.0;
        noise_ay = 9.0;

        dt = frame_seq - self.t_

        # detect frame_seq reset 
        if (dt < 0):
            dt = frame_seq + (FRAME_SQE_MAX - self.t_)

        assert dt < FRAME_SQE_MAX
        assert dt > 0

        # Timer Update
        dt_2 = dt*dt
        dt_3 = dt_2*dt
        dt_4 = dt_3*dt
        self.time_ = frame_seq

        # set the process covariance matrix Q_
        self.Q_ = np.array([[dt_4/4*noise_ax,0,dt_3/2*noise_ax,0],
                                        [0,dt_4/4*noise_ay,0,dt_3/2*noise_ay],
                                        [dt_3/2*noise_ax,0,dt_2*noise_ax,0],
                                        [0,dt_3/2*noise_ay,0,dt_2*noise_ay]])
        self.Predict()
        self.Update(z)
        #print("X_ is", self.x_, "\nP_ is ", self.P_)

    def CurrentEstPos(self):
        return [self.x_[0], self.x_[1]]

    def SetTime(self, t):
        self.t_ = t