cmake_minimum_required(VERSION 3.8)
project(rail_guard_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
  "msg/GaugeStatus.msg"
  "msg/TrackCorridor.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_package()
