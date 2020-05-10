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
