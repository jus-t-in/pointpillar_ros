cmake_minimum_required(VERSION 2.8.3)
project(detection_ros)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  jsk_recognition_msgs
  message_generation
)

add_message_files(
  FILES
  ObstacleDetection.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

