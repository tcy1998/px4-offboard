#!/usr/bin/env python

import rclpy
import numpy as np
import rclpy.logging
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition

from px4_offboard.ecbf_control import ECBF_control
from px4_offboard.controller import *
import numpy as np
import matplotlib.pyplot as plt
from px4_offboard.resilient_esti import ResilientEstimation
from px4_offboard.linear_plant import Plant
from px4_offboard._plot_figs import PlotFigs

class Offboard_acc_ctrl(Node):

    def __init__(self):

        self.safety_dist = 1.0
        self.Noise_dyn = False
        self.RE_estimator = True

        super().__init__('minimal_publisher')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
        
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_position_callback,
            qos_profile)
        
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        # Note: no parameter callbacks are used to prevent sudden inflight changes of radii and omega 
        # which would result in large discontinuities in setpoints

 
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def vehicle_position_callback(self, msg):
        self.vehicle_pos = self.NED_ENU(np.array([msg.x, msg.y, msg.z]))
        # self.vehicle_pos = np.array([msg.x, msg.y, msg.z])

    def ENU_NED(self, pos):
        return np.array([pos[1], pos[0], -pos[2]])
    
    def NED_ENU(self, pos):
        return np.array([pos[1], pos[0], -pos[2]])

    def robot_ecbf_step(self, state, state_hist, ecbf, new_obs):
        if self.Noise_dyn == True:
            noise_state = state.copy()
            sys = Plant()
            esti = ResilientEstimation(sys)
            u = np.zeros(3)
            x_noise = sys.update(u)
            y = sys.measurement(np.hstack((state['x'], state['xdot'])))
            esti.update(u, y)
            if self.RE_estimator == True:
                state['x']= [esti.Xh1[1], esti.Xh2[1], esti.Xh3[1]]
            else:
                state['x'] = x_noise[:3]
            noise_state['x'] = x_noise[:3]
            ecbf.noise_state = noise_state

        u_hat_acc = ecbf.compute_safe_control(obs=new_obs, obs_v=np.array([[0], [0]]))
        u_hat_acc = np.ndarray.flatten(np.array(np.vstack((u_hat_acc,np.zeros((1,1))))))  # acceleration
        u_hat_acc = self.ENU_NED(u_hat_acc)
        
        # u_motor, setpoints = go_to_acceleration(state, u_hat_acc, dyn.param_dict) # desired motor rate ^2
        # state = dyn.step_dynamics(state, u_motor)

        state["x"] = self.vehicle_pos
        state_hist.append(state["x"])

        return u_hat_acc, state

    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position=True
        offboard_msg.velocity=False
        offboard_msg.acceleration=False
        
        self.publisher_offboard_mode.publish(offboard_msg)

        # rclpy.logging.get_logger('px4_offboard').info("NAV_STATE: %d" % self.nav_state)
        state1 = {"x": np.array([0.0, 0.0, 2.0]),
                "xdot": np.zeros(3,),
                "theta": np.radians(np.array([0, 0, 0])),  
                "thetadot": np.radians(np.array([0, 0, 0]))  
                }
        goal1 = np.array([[2.0], [2.0]])
        ecbf1 = ECBF_control(state1, goal1)

        ## Save robot position history for plotting and analysis
        state1_hist = []
        state1_hist.append(state1["x"])

        ## Initialize obstacles
        new_obs1 = np.array([[1], [1]]) # center of obstacle
        tt = 0

        x_s, y_s, z_s = self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_pos[2]

        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
            while np.linalg.norm((x_s - state1["x"][0], y_s - state1["x"][1], z_s - state1["x"][2])) >= 0.1:
                start_pos = self.ENU_NED(np.array([state1["x"][0], state1["x"][1], state1["x"][2]]))
                trajectory_msg = TrajectorySetpoint()
                trajectory_msg.position[0] = start_pos[0]
                trajectory_msg.position[1] = start_pos[1]
                trajectory_msg.position[2] = start_pos[2]
                self.publisher_trajectory.publish(trajectory_msg)
                dis_to_target = np.linalg.norm((x_s - state1["x"][0], y_s - state1["x"][1], z_s - state1["x"][2]))
                rclpy.logging.get_logger('px4_offboard').info("Distance to target: %f" % dis_to_target)

            rclpy.logging.get_logger('px4_offboard').info("Reached initial position")
            offboard_msg.position = False
            offboard_msg.acceleration = True


            while np.linalg.norm((x_s - goal1[0][0], y_s - goal1[1][0])) >= 0.1 and tt < 1000:
                rclpy.logging.get_logger('px4_offboard').info("Time step: %d" % tt)
                u_hat_acc1, state = self.robot_ecbf_step(state1, state1_hist, ecbf1, new_obs1)
                rclpy.logging.get_logger('px4_offboard').info("Robot 1 state: %s" % state1["x"])
                rclpy.logging.get_logger('px4_offboard').info("Robot 1 acceleration: %s" % u_hat_acc1)
                x,y,z = state["x"][0], state["x"][1], state["x"][2]
                plt.cla()

                p1 = ecbf1.compute_plot_z(new_obs1)
                x_s, y_s = state['x'][0], state['x'][1]
                x += p1["x"]
                y += p1["y"]
                z += p1["z"]
                tt += 1


                trajectory_msg = TrajectorySetpoint()

                trajectory_msg.acceleration[0] = u_hat_acc1[0]
                trajectory_msg.acceleration[1] = u_hat_acc1[1]
                trajectory_msg.acceleration[2] = u_hat_acc1[2]
                print("Acceleration: ", u_hat_acc1)
                # trajectory_msg.position[0] = self.radius * np.cos(self.theta)
                # trajectory_msg.position[1] = self.radius * np.sin(self.theta)
                # trajectory_msg.position[2] = -self.altitude

                self.publisher_trajectory.publish(trajectory_msg)
            print("Reached goal")


def main(args=None):
    rclpy.init(args=args)

    offboard_control = Offboard_acc_ctrl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
