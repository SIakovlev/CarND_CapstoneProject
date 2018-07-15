# Install script for directory: /home/sergey/Documents/sdc/CarND_CapstoneProject/ros/src/styx_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/styx_msgs/msg" TYPE FILE FILES
    "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/src/styx_msgs/msg/TrafficLight.msg"
    "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/src/styx_msgs/msg/TrafficLightArray.msg"
    "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/src/styx_msgs/msg/Waypoint.msg"
    "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/src/styx_msgs/msg/Lane.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/styx_msgs/cmake" TYPE FILE FILES "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/build/styx_msgs/catkin_generated/installspace/styx_msgs-msg-paths.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/devel/include/styx_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/devel/share/roseus/ros/styx_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/devel/share/common-lisp/ros/styx_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/devel/share/gennodejs/ros/styx_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  execute_process(COMMAND "/usr/bin/python" -m compileall "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/devel/lib/python2.7/dist-packages/styx_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/devel/lib/python2.7/dist-packages/styx_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/build/styx_msgs/catkin_generated/installspace/styx_msgs.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/styx_msgs/cmake" TYPE FILE FILES "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/build/styx_msgs/catkin_generated/installspace/styx_msgs-msg-extras.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/styx_msgs/cmake" TYPE FILE FILES
    "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/build/styx_msgs/catkin_generated/installspace/styx_msgsConfig.cmake"
    "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/build/styx_msgs/catkin_generated/installspace/styx_msgsConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/styx_msgs" TYPE FILE FILES "/home/sergey/Documents/sdc/CarND_CapstoneProject/ros/src/styx_msgs/package.xml")
endif()

