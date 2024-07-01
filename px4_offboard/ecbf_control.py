from px4_offboard.controller import *
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
from matplotlib.patches import Ellipse
import rclpy
import rclpy.logging

a = 0.5
b = 0.5
safety_dist = np.power(1, 4)

a = 0.25
b = 0.25
safety_dist = np.power(0.5, 4)
class SimpleDynamics():
    def __init__(self):
        ## State space
        r = np.array([np.array([1,-4])]).T # position
        rd = np.array([np.array([0, 0])]).T  # velocity
        self.state = {"r":r, "rd":rd}
        ## Params
        self.dt = 0.01

    def step(self, u):
        rd = self.state["rd"] + self.dt * u - self.state["rd"] * 0.02
        r = self.state["r"] + self.dt * self.state["rd"]

        self.state["rd"] = rd
        self.state["r"]  = r

class ECBF_control():
    def __init__(self, x_start, goal=np.array([[0], [10]])):
        self.state = x_start
        self.noise_state = x_start
        self.shape_dict = {} #TODO: a, b
        Kp = 6
        Kd = 8
        self.K = np.array([Kp, Kd])
        self.goal = goal
        self.use_safe = True
        self.noise_safe = False
        # pass

    def compute_plot_z(self, obs):
        plot_x = np.arange(0.0, 10, 0.4)
        plot_y = np.arange(0.0, 10, 0.4)
        xx, yy = np.meshgrid(plot_x, plot_y, sparse=True)
        z = np.zeros(xx.shape)
        for i in range(obs.shape[1]):
            ztemp = h_func(xx - obs[0][i], yy - obs[1][i], a, b, safety_dist) > 0
            z = z + ztemp
        # z = z / (obs.shape[1]-1)
        p = {"x":plot_x, "y":plot_y, "z":z}
        return p
        
        # plt.show()
    def plot_h(self, plot_x, plot_y, z):
        # h = plt.contourf(plot_x, plot_y, z, [-1, 0, 1],colors=['#808080', '#A0A0A0', '#C0C0C0'])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.pause(0.00000001)



    def compute_h(self, obs=np.array([[0], [0]]).T):
        if self.noise_safe == True:
            h = np.zeros((obs.shape[1], 1))
            for i in range(obs.shape[1]):
                rel_r = np.atleast_2d(self.noise_state["x"][:2]).T - obs[:, i].reshape(2,1)
                # TODO: a, safety_dist, obs, b
                hr = h_func(rel_r[0], rel_r[1], a, b, 2.0)
                h[i] = hr
            return h
        else:
            h = np.zeros((obs.shape[1], 1))
            for i in range(obs.shape[1]):
                rel_r = np.atleast_2d(self.state["x"][:2]).T - obs[:, i].reshape(2,1)
                # TODO: a, safety_dist, obs, b
                hr = h_func(rel_r[0], rel_r[1], a, b, 2.0)
                h[i] = hr
            return h

    def compute_hd(self, obs, obs_v):
        if self.noise_safe == True:
            hd = np.zeros((obs.shape[1], 1))
            for i in range(obs.shape[1]):
                rel_r = np.atleast_2d(self.noise_state["x"][:2]).T - obs[:, i].reshape(2,1)
                rd = np.atleast_2d(self.noise_state["xdot"][:2]).T - obs_v[:, i].reshape(2,1)
                term1 = (4 * np.power(rel_r[0],3) * rd[0])/(np.power(a,4))
                term2 = (4 * np.power(rel_r[1],3) * rd[1])/(np.power(b,4))
                hd[i] = term1 + term2
            return hd
        else:
            hd = np.zeros((obs.shape[1], 1))
            for i in range(obs.shape[1]):
                rel_r = np.atleast_2d(self.state["x"][:2]).T - obs[:, i].reshape(2,1)
                rd = np.atleast_2d(self.state["xdot"][:2]).T - obs_v[:, i].reshape(2,1)
                term1 = (4 * np.power(rel_r[0],3) * rd[0])/(np.power(a,4))
                term2 = (4 * np.power(rel_r[1],3) * rd[1])/(np.power(b,4))
                hd[i] = term1 + term2
            return hd

    def compute_A(self, obs):
        if self.noise_safe == True:
            A = np.empty((0,2))
            for i in range(obs.shape[1]):
                rel_r = np.atleast_2d(self.noise_state["x"][:2]).T - obs[:, i].reshape(2,1)
                A0 = (4 * np.power(rel_r[0], 3))/(np.power(a, 4))
                A1 = (4 * np.power(rel_r[1], 3))/(np.power(b, 4))
                Atemp = np.array([np.hstack((A0, A1))])
                
                A = np.array(np.vstack((A, Atemp)))
            # print(A)
            
            return A
        else:
            A = np.empty((0,2))
            for i in range(obs.shape[1]):
                rel_r = np.atleast_2d(self.state["x"][:2]).T - obs[:, i].reshape(2,1)
                A0 = (4 * np.power(rel_r[0], 3))/(np.power(a, 4))
                A1 = (4 * np.power(rel_r[1], 3))/(np.power(b, 4))
                Atemp = np.array([np.hstack((A0, A1))])
                
                A = np.array(np.vstack((A, Atemp)))
            # print(A)
            
            return A


    def compute_b(self, obs, obs_v):
        if self.noise_safe == True:
            """extra + K * [h hd]"""
            rel_r = np.atleast_2d(self.noise_state["x"][:2]).T - obs
            rd = np.atleast_2d(self.noise_state["xdot"][:2]).T - obs_v

            extra = -( (12 * np.square(rel_r[0]) * np.square(rd[0]))/np.power(a, 4) + (12 * np.square(rel_r[1]) * np.square(rd[1]))/np.power(b, 4) )
            extra = extra.reshape(obs.shape[1], 1)

            b_ineq =  extra - ( self.K[0] * self.compute_h(obs) + self.K[1] * self.compute_hd(obs, obs_v) )
            # print(b)
            return b_ineq
        else:
            """extra + K * [h hd]"""
            rel_r = np.atleast_2d(self.state["x"][:2]).T - obs
            rd = np.atleast_2d(self.state["xdot"][:2]).T - obs_v

            extra = -( (12 * np.square(rel_r[0]) * np.square(rd[0]))/np.power(a, 4) + (12 * np.square(rel_r[1]) * np.square(rd[1]))/np.power(b, 4) )
            extra = extra.reshape(obs.shape[1], 1)

            b_ineq =  extra - ( self.K[0] * self.compute_h(obs) + self.K[1] * self.compute_hd(obs, obs_v) )
            # print(b)
            return b_ineq



    def compute_safe_control(self,obs, obs_v, current_state):
        self.state = current_state
        if self.use_safe:
            A = self.compute_A(obs)

            b_ineq = self.compute_b(obs, obs_v)
            #Make CVXOPT quadratic programming problem
            P = matrix(np.eye(2), tc='d')
            q = -1 * matrix(self.compute_nom_control(current_state), tc='d')
            G = -1 * matrix(A.astype(np.double), tc='d')

            h = -1 * matrix(b_ineq.astype(np.double), tc='d')
            solvers.options['show_progress'] = False
            sol = solvers.qp(P,q,G, h, verbose=False) # get dictionary for solution

            optimized_u = sol['x']
            # print(optimized_u)

        else:
            optimized_u = self.compute_nom_control(current_state)


        return optimized_u

    def compute_nom_control(self, state, Kn=np.array([-0.8, -0.2])):
        #! mock
        # rclpy.logging.get_logger('px4_offboard').info("vel_state: %s" % state["xdot"][:3])
        # rclpy.logging.get_logger('px4_offboard').info("pos_state: %s" % state["x"][:3])
        vd = Kn[0]*(np.atleast_2d(state["x"][:2]).T - self.goal)
        u_nom = Kn[1]*(np.atleast_2d(state["xdot"][:2]).T - vd)

        if np.linalg.norm(u_nom) > 0.25:
            u_nom = (u_nom/np.linalg.norm(u_nom))* 0.25
        return u_nom.astype(np.double)





@np.vectorize
def h_func(r1, r2, a, b, safety_dist):
    hr = np.power(r1,4)/np.power(a, 4) + \
        np.power(r2, 4)/np.power(b, 4) - safety_dist
    return hr
