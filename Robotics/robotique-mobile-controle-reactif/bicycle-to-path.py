#!/bin/env python3


"""

TP Reactive control for mobile robots

author: Alexandre Chapoutot and David Filliat

"""
import math

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
import matplotlib.patches as patches

# Parameters
show_animation = True

### Util functions

def angle_wrap (a):
    while a > np.pi:
        a = a - 2.0 * np.pi
    while a < -np.pi:
        a = a + 2.0 * np.pi
    return a


#print (angle_wrap (3.18))

def dist (xTrue, xGoal):
    error = xGoal - xTrue
    if error.size == 3:
        error[2] = angle_wrap(error[2])
    return error


def plot_arrow(x, y, yaw, length=0.1, width=0.05, fc="r", ec="k"):  # pragma: no cover
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


# Simulate vehicle

def simulate_bicycle (xTrue, u, v, phi):
    # Display trajectory

    dt = 0.001 # integration time
    dVMax = 0.01 # limit linear acceleration
    dPhiMax = 0.008 # limit angular acceleration

    # limit acceleration
    delta_v = min(u[0] - v ,dVMax)
    delta_v = max(delta_v, -dVMax)
    delta_phi = min(u[1] - phi , dPhiMax)
    delta_phi = max(delta_phi, -dPhiMax)
    v = v + delta_v
    phi = phi + delta_phi

    # Limit control
    v = min(1.0, v)
    v = max(-1.0, v)
    phi = min(1.2, phi)
    phi = max(-1.2, phi)
    u = np.array([v,phi])

    # update state
    state = np.array([xTrue[0]+v*dt*np.cos(xTrue[2]), xTrue[1]+v*dt*np.sin(xTrue[2]), xTrue[2]+v/0.5*dt*np.tan(phi)]);
    state[2] = angle_wrap(state[2]);
    return (state, u, v, phi)

# Compute control
def bicycle_to_path_control (xTrue, waypoints, waypointId, xGoal):
    # xTrue is the robot current pose : [ x y theta ]'
    # waypoints is set of points defining the path : [ [x1, x2], [y1, y2], ...]
    # u is the control : [v phi]
    # indexWaypoint, xGoal

    Kp= 3
    Ka = 2
    K_proj = 0.3
    alpha_max = math.pi


    u = np.array([0.0, 0.0])

    if waypointId == 0:
        waypointId += 1
    
    goal = waypoints[waypointId]
    previous_goal = waypoints[waypointId-1]

    distance = np.linalg.norm(goal[0:2]-xTrue[0:2])



    if distance < 0.05:
        waypointId = min(waypointId+1, len(waypoints)-1)
        
    # Project our current direction on the path direction to get as close as possible to it with a dynamic goal
    direction = xTrue[0:2] - previous_goal[0:2]
    path_direction = goal[0:2] - previous_goal[0:2]
    norm = np.sqrt(sum(path_direction**2))
    proj_of_pos_on_path = (np.dot(direction, path_direction)/norm**2)*path_direction  
    xGoal = previous_goal[0:2] + proj_of_pos_on_path[0:2] + K_proj * path_direction[0:2]/norm

    # Back to bicycle, with a dynamic goal
    p = np.sqrt((xGoal[0] - xTrue[0])**2 + (xGoal[1] - xTrue[1])**2)
    alpha = np.arctan2((xGoal[1] - xTrue[1]),(xGoal[0] - xTrue[0])) - xTrue[2]
    alpha = angle_wrap(alpha)

    if abs(alpha) > alpha_max:
        u = np.array([0, Ka * alpha])

    u = np.array([Kp * p, Ka * alpha])

    return (u, waypointId, xGoal)





#### Main function
def main():
    print("reactive control of BiCycle Pose Start")

    # Goals and initial starting position
    waypoints = np.array([ [0.1, 0, 0],
                           [4, 0, 0],
                           [4, 4, 0],
                           [3.5 , 1 , 0],
                           [0, 4, 0],
                           [0, 1, -1.57] ])

    xTrue = np.array([0.0, 0.0, 0.0])
    waypointId = 0
    xGoal = waypoints[waypointId]

    # Initial speed and angle
    v = 0
    phi = 0

    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        # display initial position
        plot_arrow (xTrue[0], xTrue[1], xTrue[2], fc='r')
        # display Path
        plt.plot(waypoints[:,0:1], waypoints[:,1:2], 'go--',
                 linewidth=2, markersize=1)
        #plt.show()

    k = 0
    print ("Initial position ", xTrue[0:2])
    print ("final waypoint position ", waypoints[-1,0:2])
    while np.amax(np.absolute(dist(xTrue[0:2],waypoints[-1,0:2]))) > 0.05 and k < 200000:
        # Compute Control
        [u, waypointId, xGoal]  = bicycle_to_path_control (xTrue, waypoints, waypointId, xGoal)


        # Simultion of the vehicle motion
        [xTrue, u, v, phi] = simulate_bicycle(xTrue, u, v, phi)

        if show_animation:
            if k % 100 == 1:
                plot_arrow (xTrue[0], xTrue[1], xTrue[2], fc='g');
                plt.plot(xTrue[0], xTrue[1], ".g")
                plt.pause(0.000001)

        k = k + 1

    print ("k = ", k)
    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
