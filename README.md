# CPG
Implementation of Central Pattern Generators (CPG) on Unitree Go1

This repository contains a ROS 2 workspace (`ws_go1`) with:
- **quad_cpg** — a Python package that generates foot trajectories using a Hopf oscillator network and applies PD control to the Unitree GO1 joints.
- **unitree_ros2_sim** — description, Gazebo launch/worlds, and controller configs for the GO1.

The CPG is based on a Hopf network with gait-specific coupling and simple sensory feedback (contacts). It publishes joint torques/commands to Gazebo through standard ROS 2 topics.
