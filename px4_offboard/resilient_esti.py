import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import multi_dot


class ResilientEstimation:
    def __init__(self, plant_linear, x_init=np.array([0, 0, 10, 0, 0, 0])):
        self.A = plant_linear.A
        self.B = plant_linear.B
        self.C = plant_linear.C

        self.Q = plant_linear.Cov_w
        self.R = plant_linear.Cov_v

        ''' Initialization'''
        self.x_hat = x_init
        self.d_hat = np.zeros(3)
        self.Px = np.eye(6)

        '''data collection'''
        self.Xh1, self.Xh2, self.Xh3 = [self.x_hat[0]], [self.x_hat[1]], [self.x_hat[2]]
        self.Fh1, self.Fh2, self.Fh3 = [self.d_hat[0]], [self.d_hat[1]], [self.d_hat[2]]

    def update(self, u, y):
        """Prediction"""
        x_hat_p = np.dot(self.A, self.x_hat) + np.dot(self.B, u)
        Px_p = multi_dot([self.A, self.Px, self.A.T]) + self.Q

        """Fault estimation"""
        R_t = multi_dot([self.C, Px_p, self.C.T]) + self.R
        Pd = pinv(multi_dot([self.C.T, pinv(R_t), self.C]))
        M = multi_dot([Pd, self.C.T, inv(R_t)])

        self.d_hat = np.dot(M, (y - np.dot(self.C, x_hat_p)))
        Pxd = - multi_dot([self.Px, self.A.T, self.C.T, M.T])

        """Time update"""
        x_s = x_hat_p + self.d_hat
        Px_s = multi_dot([self.A, self.Px, self.A.T]) \
               + multi_dot([self.A, Pxd]) \
               + multi_dot([Pxd.T, self.A.T]) \
               + Pd - multi_dot([M, self.C, self.Q]) \
               - multi_dot([self.Q, self.C.T, M.T]) + self.Q
        R_ts = multi_dot([self.C, Px_s, self.C.T]) + self.R \
               - multi_dot([self.C, M, self.R]) - multi_dot([self.R, M.T, self.C.T])

        """State estimation"""
        L = np.dot((np.dot(Px_s, self.C.T) - multi_dot([M, self.R])), pinv(R_ts))
        self.x_hat = x_s + np.dot(L, (y - np.dot(self.C, x_s)))
        self.Px = multi_dot([(np.eye(6) - np.dot(L, self.C)), M, self.R, L.T]) \
                  + multi_dot([L, self.R, M.T, (np.eye(6) - np.dot(L, self.C)).T]) \
                  + multi_dot([(np.eye(6) - np.dot(L, self.C)), Px_s, (np.eye(6) - np.dot(L, self.C)).T]) \
                  + multi_dot([L, self.R, L.T])
        '''collect data'''
        self.Xh1.append(self.x_hat[0])
        self.Xh2.append(self.x_hat[1])
        self.Xh3.append(self.x_hat[2])

        self.Fh1.append(self.d_hat[0])


if __name__ == "__main__":
    from linear_plant import Plant
    from _plot_figs import PlotFigs
    time_step = 60

    sys = Plant()
    esti = ResilientEstimation(sys)
    u = np.zeros(3)

    for k in range(time_step):
        x = sys.update(u)
        y = sys.measurement(x)
        esti.update(u, y)

    plot = PlotFigs(time_step, sys, esti)

    plot.x1()
    plot.x2()
    plot.x3()

    plot.d()
