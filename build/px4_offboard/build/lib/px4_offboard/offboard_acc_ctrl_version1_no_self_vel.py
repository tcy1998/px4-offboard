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
        self.vehicle_pos = np.array([0.0, 0.0, 0.0])
        self.vehicle_vel = np.array([0.0, 0.0, 0.0])
        self.time_epi = 0
        self.takeoff = True
        self.control = False
        # Note: no parameter callbacks are used to prevent sudden inflight changes of radii and omega 
        # which would result in large discontinuities in setpoints

 
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def vehicle_position_callback(self, msg):
        self.vehicle_pos = self.NED_ENU(np.array([msg.x, msg.y, msg.z]))
        self.vehicle_vel = self.NED_ENU(np.array([msg.vx, msg.vy, msg.vz]))

    def ENU_NED(self, pos):
        return np.array([pos[1], pos[0], -pos[2]])
    
    def NED_ENU(self, pos):
        return np.array([pos[1], pos[0], -pos[2]])

    def robot_ecbf_step(self, state, ecbf, new_obs):
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

        # rclpy.logging.get_logger('px4_offboard').info("Robot 1 state: %s" % state["x"])
        u_hat_acc = ecbf.compute_safe_control(obs=new_obs, obs_v=np.array([[0], [0]]), current_state=state)
        u_hat_acc = np.ndarray.flatten(np.array(np.vstack((u_hat_acc,np.zeros((1,1))))))  # acceleration
        u_hat_acc = self.ENU_NED(u_hat_acc)
        

        return u_hat_acc

    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position=True
        offboard_msg.velocity=False
        offboard_msg.acceleration=True
        
        self.publisher_offboard_mode.publish(offboard_msg)

        ## Initialize robot start position in ENU
        state1 = {"x": np.array([-1.5, -2.5, 2.0]),
                "xdot": np.zeros(3,),
                "theta": np.radians(np.array([0, 0, 0])),  
                "thetadot": np.radians(np.array([0, 0, 0]))  
                }
        goal1 = np.array([[1.5], [2.5]])
        
        ecbf1 = ECBF_control(state1, goal1)

        ## Initialize obstacles
        new_obs1 = np.array([[0.0], [0.0]]) # center of obstacle

        x_s, y_s, z_s = self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_pos[2]

        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
            if self.takeoff:            
                if np.linalg.norm((x_s - state1["x"][0], y_s - state1["x"][1], z_s - state1["x"][2])) >= 0.3:
                    start_pos = self.ENU_NED(np.array([state1["x"][0], state1["x"][1], state1["x"][2]]))        # Send start pos in NED
                    trajectory_msg = TrajectorySetpoint()
                    trajectory_msg.position[0] = start_pos[0]
                    trajectory_msg.position[1] = start_pos[1]
                    trajectory_msg.position[2] = start_pos[2]
                    self.publisher_trajectory.publish(trajectory_msg)
                else:
                    self.takeoff = False
                    self.control = True
                    rclpy.logging.get_logger('px4_offboard').info("Reached initial position")
                    rclpy.logging.get_logger('px4_offboard').info("beginning control loop")

            if self.control:
                state1['x'] = self.vehicle_pos
                state1['xdot'] = self.vehicle_vel
                offboard_msg.position = False
                offboard_msg.velocity = False
                offboard_msg.acceleration = True
                self.publisher_offboard_mode.publish(offboard_msg)
                if np.linalg.norm((state1['x'][0] - goal1[0][0], state1['x'][1] - goal1[1][0] )) >= 0.3 and self.time_epi < 1200:
                    
                    # rclpy.logging.get_logger('px4_offboard').info("Time step: %d" % self.time_epi)
                    u_hat_acc1 = self.robot_ecbf_step(state1, ecbf1, new_obs1)
                    # rclpy.logging.get_logger('px4_offboard').info("Robot 1 state: %s" % state1["x"])
                    # rclpy.logging.get_logger('px4_offboard').info("Robot 1 acceleration: %s" % u_hat_acc1)
                    self.time_epi += 1


                    trajectory_msg = TrajectorySetpoint()

                    trajectory_msg.position[0] = "NaN"
                    trajectory_msg.position[1] = "NaN"
                    trajectory_msg.position[2] = -2.0

                    trajectory_msg.yaw = 0.0

                    trajectory_msg.velocity[0] = 0.1
                    trajectory_msg.velocity[1] = 0.0
                    trajectory_msg.velocity[2] = 0.0

                    trajectory_msg.acceleration[0] = u_hat_acc1[0]
                    trajectory_msg.acceleration[1] = u_hat_acc1[1]
                    trajectory_msg.acceleration[2] = 0.0
                    # trajectory_msg.position[2] = -2.0
                    self.publisher_trajectory.publish(trajectory_msg)

                    state1['x'] = self.vehicle_pos
                    state1['xdot'] = self.vehicle_vel
                else:
                    rclpy.logging.get_logger('px4_offboard').info("Time step: %d" % self.time_epi)
                    rclpy.logging.get_logger('px4_offboard').info("Robot 1 state: %s" % state1["x"])
                    rclpy.logging.get_logger('px4_offboard').info("Reach Target")
                    self.control = False
            
            if self.control == False and self.takeoff == False:
                offboard_msg.position = True
                offboard_msg.velocity = False
                offboard_msg.acceleration = False
                self.publisher_offboard_mode.publish(offboard_msg)
                trajectory_msg = TrajectorySetpoint()
                trajectory_msg.position[0] = 0.0
                trajectory_msg.position[1] = 0.0
                trajectory_msg.position[2] = -0.0
                self.publisher_trajectory.publish(trajectory_msg)
                # rclpy.logging.get_logger('px4_offboard').info("Landing")

def main(args=None):
    rclpy.init(args=args)

    offboard_control = Offboard_acc_ctrl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
