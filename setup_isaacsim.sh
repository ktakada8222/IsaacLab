#!/bin/bash

# Isaac Sim / Isaac Lab Vulkan setting
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Isaac Sim ROS2 Bridge setting
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Isaac Sim internal ROS2 Bridge libraries
ISAACSIM_ROS2_BRIDGE_LIB="/home/tron/miniconda3/envs/takada_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"
