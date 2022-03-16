## Robotics - Simple mover tutorial

Goal: Learn how to write ROS nodes in C++.

Info on how to run the simulation can be find on the [tutorial github](https://github.com/udacity/RoboND-simple_arm/blob/simple_mover/README.md).

#### `simple_mover`: node publishing joint angle commands to `simple_arm`.

The goal of the `simple_mover` node is to command each joint in the simple arm and make it swing between -pi/2 to pi/2 over time.

![alt text](https://github.com/JeanJulesBigeard/Practical-Works-Machine-Learning/blob/master/ROS%20tutorials/Simple-mover-tutorial-ROS-Gazebo-master/Images/Screenshot%20from%202019-11-25%2019-54-08.png)

-> Topics:

To do so, it must publish joint angle command messages to the following topics:

*Topic Name*:	/simple_arm/joint_1_position_controller/command

*Message Type*:	std_msgs/Float64

*Description Commands*: joint 1 to move counter-clockwise, units in radians

*Topic Name*:	/simple_arm/joint_2_position_controller/command

*Message Type*:	std_msgs/Float64

*Description	Commands*: joint 2 to move counter-clockwise, units in radians

-> CMakeLists.txt:

In order for catkin to generate the C++ libraries, you must first modify simple_arm’s CMakeLists.txt.
CMake is the build tool underlying catkin, and CMakeLists.txt is a CMake script used by catkin. 

As the names might imply, the `std_msgs` package contains all of the basic message types, and message_generation is required to generate message libraries for all the supported languages (cpp, lisp, python, javascript). The `contoller_manager` is another package responsible for controlling the arm.

```xml
find_package(catkin REQUIRED COMPONENTS
        std_msgs
        message_generation
        controller_manager
)
```
These instructions ask the compiler to include the directories, executable file, link libraries, and dependencies for your C++ code

```xml
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(simple_mover src/simple_mover.cpp)
target_link_libraries(simple_mover ${catkin_LIBRARIES})
add_dependencies(simple_mover simple_arm_generate_messages_cpp)
```

Build and run:

Build `simple_mover`

```shell
$ cd /home/workspace/catkin_ws/
$ catkin_make
```
Run `simple_mover`

1-/ Launch simple_arm as follows:
```shell
$ cd /home/workspace/catkin_ws/
$ source devel/setup.bash
$ roslaunch simple_arm robot_spawn.launch
```
2-/ Once the ROS Master, Gazebo, and all of our relevant nodes are up and running, we can finally launch simple_mover. To do so, open a new terminal and type the following commands:
```shell
$ cd /home/workspace/catkin_ws/
$ source devel/setup.bash
$ rosrun simple_arm simple_mover
```

#### `arm_mover`: node provides the service `safe_move`, which allow for other nodes in the sstem to send `movement_commands`.

In addition to allowing movements via a service interface, `arm_mover` also allows for configurable minimum and maximum joint angle, by using parameters.

-> Do to so, we've created a new service for `simple_arm` : `GoToPosition.srv`.

NB: An interaction with a service consists of two messages. A node passes a request message to the service, and the service returns a response message to the node. The definitions of the request and response message types are contained within .srv files living in the srv directory under the package’s root.

![alt text](https://github.com/JeanJulesBigeard/Practical-Works-Machine-Learning/blob/master/ROS%20tutorials/Simple-mover-tutorial-ROS-Gazebo-master/Images/screen-shot-2018-10-30-at-11.33.36-am.png)

Service definitions always contain two sections, separated by a ‘---’ line. The first section is the definition of the request message. Here, a request consists of two float64 fields, one for each of `simple_arm`’s joints. The second section contains the service response. The response contains only a single field, `msg_feedback`. The msg_feedback field is of type string, and is responsible for indicating that the arm has moved to a new position.

-> `CMakeLists.txt`:

`add_service_files()`: Tells catkin to add the newly created service file.

```cpp
 add_service_files(
   FILES
   GoToPosition.srv
)
```
`generate_messages()`: Responsible for generating the code.

```cpp
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
)
```

-> `package.xml` is responsible for defining many of the package’s properties, such as the name of the package, version numbers, authors, maintainers, and dependencies.

-> Checking service: 

```shell
$ cd /home/workspace/catkin_ws/
$ source devel/setup.bash
$ rossrv show GoToPosition
```
-> Arm Mover: Build, Launch and Interact:

Build:

```shell
$ cd /home/workspace/catkin_ws/
$ catkin_make
```

Launch with the new service:

To get the `arm_mover` node, and accompanying `safe_move` service, to launch along with all of the other nodes, modify `robot_spawn.launch`.

Launch files, when they exist, are located within the `launch` directory in the root of a catkin package. Inside a launch file, you can instruct ROS Master which nodes to run. Also you can specify certain parameters and arguments for each of your nodes. Thus, a launch file is necessary inside a ROS package containing more than one node or a node with multiple parameters and arguments. This launch file can run all the nodes within a single command: `roslaunch package_name launch_file.launch`. `simple_arm`’s launch file is located in `/home/workspace/catkin_ws/src/simple_arm/launch`

Inside the launch file, the node tag specifies the name, type, package name and output channel. The ROS parameters specify the min and max joint angles.

In order to get the arm_mover node to launch:

```xml
  <!-- The arm mover node -->
  <node name="arm_mover" type="arm_mover" pkg="simple_arm" output="screen">
    <rosparam>
      min_joint_1_angle: 0
      max_joint_1_angle: 1.57
      min_joint_2_angle: 0
      max_joint_2_angle: 1.0
    </rosparam>
  </node>
```

Testing the new service:

Launch the `simple_arm`, verify that the `arm_mover` node is running and that the `safe_move` service is listed:

```shell
$ cd /home/workspace/catkin_ws/
$ source devel/setup.bash
$ roslaunch simple_arm robot_spawn.launch
```
Verify that the node and service have indeed launched:

```shell
$ rosnode list
$ rosservice list
```
Check that both the service (`/arm_mover/safe_move`) and the node (`/arm_mover`) show up as expected. If they do not appear, check the logs in the roscore console. You can now interact with the service using rosservice.

-> Camera stream:
```shell
$ rqt_image_view /rgb_camera/image_raw
```
Adjusting the view:
The camera is displaying a gray image. This is to be expected, given that it is pointing straight up, towards the gray sky of our Gazebo world. To point the camera towards the numbered blocks on the countertop, we need to rotate both joint 1 and joint 2 by approximately pi/2 radians.

```shell 
$ cd /home/workspace/catkin_ws/
$ source devel/setup.bash
$ rosservice call /arm_mover/safe_move "joint_1: 1.57
joint_2: 1.57"
```
Looking at the `roscore` console, we can see the problem. The requested angle for joint 2 was out of the safe bounds, so it was clamped. We requested 1.57 radians, but the maximum joint angle was set to 1.0 radians.

By setting the `max_joint_2_angle` on the parameter server, we should be able to increase joint 2’s maximum angle and bring the blocks into view the next time we request a service. To update that parameter, use the command `rosparam`.

```shell
$ rosparam set /arm_mover/max_joint_2_angle 1.57
```
![alt text](https://github.com/JeanJulesBigeard/Practical-Works-Machine-Learning/blob/master/ROS%20tutorials/Simple-mover-tutorial-ROS-Gazebo-master/Images/Screenshot%20from%202019-11-28%2020-15-08.png)


### `look_away`: this node will subscribe to the /rgb_camera/image_raw topic, which has image data from the camera mounted on the end of the robotic arm.

Whenever the camera is pointed towards an uninteresting image - in this case, an image with uniform color - the callback function will request a `safe_move` service to safely move the arm to something more interesting. 

Building the package:

```shell
$ cd /home/workspace/catkin_ws/
$ catkin_make
```

Launching the nodes:

```shell
$ cd /home/workspace/catkin_ws/
$ source devel/setup.bash
$ roslaunch simple_arm robot_spawn.launch
```

Interacting with the arm:

```shell
$ rqt_image_view /rgb_camera/image_raw
```

To check that everything is working as expected, open a new terminal and send a service call to point the arm directly up towards the sky.

```shell
$ cd /home/workspace/catkin_ws/
$ source devel/setup.bash
$ rosservice call /arm_mover/safe_move "joint_1: 0
joint_2: 0"
```
