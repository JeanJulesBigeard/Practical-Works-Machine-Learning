#include "ros/ros.h"
#include "std_msgs/Float64.h"

// ros is the official client library for ROS. It provides most of the fundamental 
// functionality required for interfacing with ROS via C++. It has tools for creating 
// Nodes and interfacing with Topics, Services, and Parameters.


// From the std_msgs package, the Float64 header file is imported. The std_msgs 
// package also contains the primitive message types in ROS. Later, you will be 
// publish Float64 messages to the position command topics for each joint.

int main(int argc, char** argv)
{

// A ROS node is initialized with the init() function and 
// registered with the ROS Master. Here arm_mover is the name of 
// the node. Notice that the main function takes both argc and argv arguments 
// and passes them to the init() function.

    // Initialize the arm_mover node
    ros::init(argc, argv, "arm_mover");

//A node handle object n is instantiated from the NodeHandle class. This node 
// handle object will fully initialize the node and permits it to communicate with 
// the ROS Master.

    // Create a handle to the arm_mover node
    ros::NodeHandle n;

// Two publishers are declared, one for joint 1 commands, and one for joint 2 commands.
//  The node handle will tell the ROS master that a Float64 message will be published 
//  on the joint topic. The node handle also sets the queue size to 10 in the second 
//  argument of the advertise function.

    // Create a publisher that can publish a std_msgs::Float64 message on the /simple_arm/joint_1_position_controller/command topic
    ros::Publisher joint1_pub = n.advertise<std_msgs::Float64>("/simple_arm/joint_1_position_controller/command", 10);
    // Create a publisher that can publish a std_msgs::Float64 message on the /simple_arm/joint_2_position_controller/command topic
    ros::Publisher joint2_pub = n.advertise<std_msgs::Float64>("/simple_arm/joint_2_position_controller/command", 10);

// A frequency of 10HZ is set using the loop_rate object. Rates are used in ROS to limit 
// the frequency at which certain loops cycle. Choosing a rate that is too high may result 
// in unnecessary CPU usage, while choosing a value too low could result in high latency. 
// Choosing sensible values for all of the nodes in a ROS system is a bit of a fine art.

    // Set loop frequency of 10Hz
    ros::Rate loop_rate(10);

// We set start_time to the current time. In a moment we will use this to determine 
// how much time has elapsed. When using ROS with simulated time (as we are doing here), 
// ros-Time-now will initially return 0, until the first message has been received on the 
// /clock topic. This is why start_time is set and polled continuously until a nonzero value 
// is returned.

    int start_time, elapsed;

    // Get ROS start time
    while (not start_time) {
        start_time = ros::Time::now().toSec();
    }

    while (ros::ok()) {

// In the main loop, the elapsed time is evaluated by measuring the current time and 
// subtracting the start time.

        // Get ROS elapsed time
        elapsed = ros::Time::now().toSec() - start_time;

// The joint angles are sampled from a sine wave with a period of 10 seconds,
// and in magnitude from [-pi/2, +pi/2].

        // Set the arm joint angles
        std_msgs::Float64 joint1_angle, joint2_angle;
        joint1_angle.data = sin(2 * M_PI * 0.1 * elapsed) * (M_PI / 2);
        joint2_angle.data = sin(2 * M_PI * 0.1 * elapsed) * (M_PI / 2);

// Each trip through the body of the loop will result in two joint command messages 
// being published.

        // Publish the arm joint angles
        joint1_pub.publish(joint1_angle);
        joint2_pub.publish(joint2_angle);

// Due to the call to loop_rate.sleep(), the loop is traversed at approximately 10 Hertz. 
// When the node receives the signal to shut down (either from the ROS Master, or via a 
// signal from a console window), the loop will exit.

        // Sleep for the time remaining until 10 Hz is reached
        loop_rate.sleep();
    }

    return 0;
}