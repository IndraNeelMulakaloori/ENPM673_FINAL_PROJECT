# Turtlebot Perception Challenge

This repository contains the code and resources for the ENPM673 Turtlebot Perception Challenge, a project focused on perception, detection, and navigation using a simulated Turtlebot in a Gazebo environment. The project leverages ROS2 for robot control, sensor integration, and perception tasks such as ArUco marker detection and object recognition using YOLOv8.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation & Build Instructions](#installation--build-instructions)
- [Running the Project](#running-the-project)
  - [Python Version](#python-version)
  - [C++ Version](#c-version)
- [Gazebo Simulation](#gazebo-simulation)
- [Camera Image Visualization](#camera-image-visualization)
- [Manual Turtlebot Control](#manual-turtlebot-control)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Structure

```

├── enpm673_final_proj/           # Main ROS2 package
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── enpm673_module/
│   │   ├── __init__.py
│   │   ├── aruco_detector.py
│   │   └── enpm673_final_proj.py
│   ├── launch/
│   │   └── enpm673_world.launch.py
│   ├── models/                   # Gazebo models (YOLO, ArUco, etc.)
│   ├── scripts/
│   │   └── enpm673_final_proj_main.py
│   ├── src/
│   │   └── enpm673_final_proj.cpp
│   └── worlds/
│       └── enpm673.world
├── perception_pkg/               # Additional perception package
├── screenshots/                  # Images and videos for documentation
├── pose.txt
├── README.md
└── LICENSE
```

---

## Installation & Build Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/TommyChangUMD/ENPM673_turtlebot_perception_challenge.git
   cd ENPM673_turtlebot_perception_challenge/
   ```

2. **Source ROS2 and build the package:**
   ```sh
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install --packages-select enpm673_final_proj
   ```

---

## Running the Project

### Python Version

1. **Source the workspace:**
   ```sh
   source install/setup.bash
   ```

2. **Run the main Python node:**
   ```sh
   ros2 run enpm673_final_proj enpm673_final_proj_main.py
   ```

   - Main source files:
     - enpm673_final_proj.py
     - enpm673_final_proj_main.py

### C++ Version

1. **Source the workspace:**
   ```sh
   source install/setup.bash
   ```

2. **Run the C++ node:**
   ```sh
   ros2 run enpm673_final_proj cpp_enpm673_final_proj
   ```

   - Main source file:
     - enpm673_final_proj.cpp

---

## Gazebo Simulation

To launch the Gazebo simulation with the custom world:

```sh
source install/setup.bash
source /usr/share/gazebo/setup.bash   # May not be needed on all systems
ros2 launch enpm673_final_proj enpm673_world.launch.py "verbose:=true"
```

- The Gazebo world file is located at: enpm673.world

![Gazebo Simulation](screenshots/gazebo.png)

To stop the simulation, press `Ctrl+C` in the terminal.

---

## Camera Image Visualization

To view the camera feed using `rqt_image_view`:

```sh
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

!rqt_image_view

---

## Manual Turtlebot Control

You can manually drive the Turtlebot using teleoperation packages such as `teleop_twist_keyboard`:

```sh
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

## Troubleshooting

- Ensure all dependencies are installed and sourced correctly.
- If Gazebo models do not appear, verify the `models` directory is installed to the correct location.
- For ROS2 or Gazebo issues, consult the official documentation or check environment variables.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

---
