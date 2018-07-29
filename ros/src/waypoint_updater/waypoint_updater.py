#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math
import sys

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

#STEEPNESS = 5
#XCENTER = 5

MPH_TO_MPS = 0.44704
MAX_SPEED = 5 * MPH_TO_MPS

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.decel_limit = rospy.get_param('~decel_limit', -5)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.pose = None
        self.stopline_wp_idx = -1

        #rospy.spin()
        self.loop()

    def loop(self):
    	rate = rospy.Rate(50)
    	while not rospy.is_shutdown():
    		if self.pose and self.base_lane:
    			self.publish_waypoints()
    		rate.sleep()

    def publish_waypoints(self):
        lane = self.generate_lane()
    	self.final_waypoints_pub.publish(lane)

    def generate_lane(self):
        lane = Lane()

        # lane.header = self.base_waypoints.header
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        # set max speed for each waypoint
        for wp in base_waypoints:
            wp.twist.twist.linear.x = MAX_SPEED

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        #C = waypoints[0].twist.twist.linear.x
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0) # to make sure that car front stops before the stopline
            dist = self.distance(waypoints, i, stop_idx)

            vel = np.sqrt(2 * abs(self.decel_limit) * dist)
            #vel = C / (1 + np.exp(-STEEPNESS/XCENTER*(dist-XCENTER))) + 1
            if vel < 1:
                vel = 0.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def get_closest_waypoint_idx(self):
    
        if self.pose is None or self.waypoints_tree is None:
            # No data to work from yet (in simulator can get here before we
            # have waypoint tree object)
            return 0
            
    	x = self.pose.pose.position.x
    	y = self.pose.pose.position.y
    	closest_idx = self.waypoints_tree.query([x, y], 1)[1]

    	closest_coord = np.array(self.waypoints_2d[closest_idx])
    	prev_coord = np.array(self.waypoints_2d[closest_idx-1])
    	car_coord = np.array([x, y])

    	# if the closest waypoing is behind the car
    	if np.dot(closest_coord - prev_coord, car_coord - closest_coord) > 0:
    		closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
    	
    	return closest_idx

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_lane = waypoints

        if not self.waypoints_2d:
        	self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        	self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
