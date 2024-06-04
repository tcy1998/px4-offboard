import numpy as np
from numpy.random import multivariate_normal as sample_normal_vec


class Plant:
    def __init__(self, num_dim=6, x_init=np.array([0, 0, 10, 0, 0, 0])):
        self.i = 0

        # self.A = np.array([[1, 0, 0.1],
        #                    [0, 1, 0],
        #                    [0, 0, 1]])

        self.A = np.array([[0 ,0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]])

        # self.B = np.array([[0, 0],
        #                    [0, 0],
        #                    [0, 0]])

        self.B = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

        
        # self.C = np.array([[1, 0, 0],
        #                    [0, 1, 0],
        #                    [0, 0, 1]])

        self.C = np.eye(num_dim)

        self.Cov_w = 0.01 * np.eye(num_dim)
        self.Cov_v = 0.01 * np.eye(num_dim)

        self.x = x_init
        # self.f = np.array([0, 0, 0, 0, 0, 0])
        self.f = np.zeros(num_dim)

        '''data collection'''
        self.X1, self.X2, self.X3 = [self.x[0]], [self.x[1]], [self.x[2]]
        self.X1_dot, self.X2_dot, self.X3_dot = [self.x[3]], [self.x[4]], [self.x[5]]
        self.F1, self.F2, self.F3 = [self.f[0]], [self.f[1]], [self.f[2]]

    def update(self, u):

        d = np.sin(self.i)  # faulty model uncertainty
        self.f = np.array([d, 0, 0])

        '''Noise'''
        w = sample_normal_vec(np.zeros(6), self.Cov_w)

        '''State transition'''
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u) + 0.01 * d + w

        '''collect data'''
        self.X1.append(self.x[0])
        self.X2.append(self.x[1])
        self.X3.append(self.x[2])

        self.F1.append(self.f[0])

        self.i += 1

        return self.x

    def measurement(self, x):
        v = sample_normal_vec(np.zeros(6), self.Cov_v)
        y = np.dot(self.C, x) + 0.01 * v

        return y


