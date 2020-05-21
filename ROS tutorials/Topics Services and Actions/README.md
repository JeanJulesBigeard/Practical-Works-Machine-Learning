#### Publish velocity

Create your own publisher and make the robot move:

1 - The /cmd_vel topic is the topic used to move the robot. Do a rostopic info /cmd_vel in order to get information about this topic, and identify the message it uses. You have to modify the code to use that message.

2 - In order to fill the Twist message, you need to create an instance of the message. In Python, this is done like this: var = Twist()

3 - In order to know the structure of the Twist messages, you need to use the rosmsg show command, with the type of the message used by the topic /cmd_vel.

4 - In this case, the robot uses a differential drive plugin to move. That is, the robot can only move linearly in the x** axis, or rotationaly in the angular **z axis. This means that the only values that you need to fill in the Twist message are the linear x** and the angular **z.

5 - The magnitudes of the Twist message are in m/s, so it is recommended to use values between 0 and 1. For example, 0'5 m/s.

#### Read odometry

Create a package in order to print the odometry of the robot:

1 - The odometry of the robot is published by the robot into the /odom topic.

2 - You will need to figure out what message uses the /odom topic, and how the structure of this message is.

#### Laser scan command

1 - Create a Publisher that writes into the /cmd_vel topic in order to move the robot.

2 - Create a Subscriber that reads from the /kobuki/laser/scan topic. This is the topic where the laser publishes its data.

3 - Depending on the readings you receive from the laser's topic, you'll have to change the data you're sending to the /cmd_vel topic in order to avoid the wall. This means, use the values of the laser to decide.

The logic that your program has to follow is the next one:

If the laser reading in front of the robot is higher than 1 meter (there is no obstacle closer than 1 meter in front of the robot), the robot will move forward.

If the laser reading in front of the robot is lower than 1 meter (there is an obstacle closer than 1 meter in front of the robot), the robot will turn left.

If the laser reading at the right side of the robot is lower than 1 meter (there is an obstacle closer than 1 meter at the right side of the robot), the robot will turn left.

If the laser reading at the left side of the robot is lower than 1 meter (there is an obstacle closer than 1 meter at the left side of the robot), the robot will turn right.

#### unit_4_services

Create a node calling the /execute_trajectory service to move the arm on a trajectory in a file.

First of all, create a package to place all the future code. For better future reference, you can call it unit_4_services, with dependencies rospy and iri_wam_reproduce_trajectory.

Create a launch called my_robot_arm_demo.launch, that starts the /execute_trajectory service. As explained in the Example 4.3, this service is launched by the launch file start_service.launch, which is in the package iri_wam_reproduce_trajectory.

Get information of what type of service message does this /execute_trajectory service uses, as explained in Example 4.6.

Make the robotic arm move following a trajectory, which is specified in a file.

Modify the previous code of Example 4.5, which called the /trajectory_by_name service, to call now the /execute_trajectory service instead. The new Python file could be called exercise_4_1.py, for future reference.

Here you have the code necessary to get the path to the trajectory files based on the package where they are. Here, the trajectory file get_food.txt is selected, but you can use any of the available in the config folder of the iri_wam_reproduce_trajectory package.

Modify the main launch file my_robot_arm_demo.launch, so that now it also launches the Python code you have just created in exercise_4_1.py.

Finally, execute the my_robot_arm_demo.launch file and see how the robot performs the trajectory.

#### Services quiz:

The name of the package where you'll place all the code related to the quiz will be services_quiz.

The name of the launch file that will start your Service Server will be start_bb8_move_custom_service_server.launch.

The name of the service will be /move_bb8_in_square_custom.

The name of your Service message file will be BB8CustomServiceMessage.srv.

The Service message file will be placed at the services_quiz package, as indicated in the 1st point.

The name of the launch file that will start your Service Client will be call_bb8_move_in_square_custom_service_server.launch (This launch file doesn't have to start the Service Server, only the Service Client).

The small square has to be of, at least, 1 sqm. The big square has to be of, at least, 2 sqm.

#### Action client to move a drone:

1) You can send Twist commands to the quadcopter in order to move it. These commands have to be published in **/cmd_vel** topic. Remember the **TopicsUnit**.

2) You must send movement commands while waiting until the result is received, creating a loop that sends commands at the same time that check for completion. In order to be able to send commands while the action is in progress, you need to use the SimpleActionClient function **get_state()**.

3) Also, take into account that in some cases, the 1st message published into a topic may fail (because the topic connections are not ready yet). It's important to bear this in mind specially for **taking off/landing** the drone, since it's based in a single publication into the corresponding topics.

####  Package with Action Server that moves the AR.Drone in the air, making a square:

a) Create a package with an action server that makes the drone move in a square when called.

b) Call the action server through the topics and observe the result and feedback.

c) Your code should move the drone while taking pictures.

The size of the side of the square should be specified in the goal message as an integer.
The feedback should publish the current side (as a number) the robot is at while doing the square.
The result should publish the total number of seconds it took the drone to do the square
Use the Test.action message for that action server. Use the shell command find /opt/ros/kinetic/ -name Test.action to find where that message is defined. Then, analyze the different fields of the msg in order to learn how to use it in your action server. As you can see its in the package actionlib

#### Actions quizz:

Create a Package with an action server with custom action message to move ardone.

The new action server will receive two words as a goal: TAKEOFF or LAND.

When the action server receives the TAKEOFF word, the drone will take off.

When the action server receives the LAND word, the drone will land.

As a feedback, it publishes once a second what action is taking place (taking off or landing).

When the action finishes, the result will return nothing.
