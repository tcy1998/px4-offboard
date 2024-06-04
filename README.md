# px4-offboard-accleration and CBF implementation
This a project based on Jaeyoung Lim project
```
git clone https://github.com/Jaeyoung-Lim/px4-offboard.git
```

The `px4_offboard` package contains the following nodes
- `offboard_control.py`: Example of offboard position control using position setpoints
- `visualizer.py`: Used for visualizing vehicle states in Rviz
- `offboard_acc_ctrl_version1.py`: New python code for accleration control with CBF

The source code is released under a BSD 3-Clause license.

- **Author**: Chuyuan Tao
- **Affiliation**: ACRL, University of Illinois, at Urbana and Champaign

## Setup
It is based on WSL or Ubuntu System.

If you are running this on a companion computer to PX4, you will need to build the package on the companion computer directly. 

More to Add.

## Running

### Software in the Loop
You will make use of 4 different terminals to run the offboard demo.

On the first terminal, run a SITL instance from the PX4 Autopilot firmware.
```
cd PX4-Autopilot
make px4_sitl gz_x500
```

On the second terminal terminal, run the micro-ros-agent which will perform the mapping between Micro XRCE-DDS and RTPS. So that ROS2 Nodes are able to communicate with the PX4 micrortps_client.
```
cd Micro-XRCE-DDS-Agent
MicroXRCEAgent udp4 -p 8888
```

Open the third terminal, open the QGround Control app.
```
./QGroundControl.AppImage
```

In order to run the offboard position control example, open a forth terminal and run the the node.
This runs two ros nodes, which publishes offboard position control setpoints and the visualizer.
```
cd px4-offboard
source install/setup.bash
ros2 launch px4_offboard offboard_accleration_control.launch.py
```
Make changes on the python code, need to build before running the launch.py
```
colcon build
source install/setup.bash
ros2 launch px4_offboard offboard_accleration_control.launch.py
```

