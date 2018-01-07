cmake_minimum_required(VERSION 2.8.3)
project(find_similar_objects)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

add_message_files(
  FILES
)

add_service_files(
  FILES
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
)
