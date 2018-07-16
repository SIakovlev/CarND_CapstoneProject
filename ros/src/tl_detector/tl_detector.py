#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import cv2
import yaml
import sys # for debug output

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.camera_image = None
        self.lights = []
        
        self.training_image_idx = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        # In simulator, get ground truth traffic light states in here too:
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        
        # Walkthrough video ~2 mins:
        # /image_color is camera data, but might alternatively want to use image_raw instead
        # This version is somehow independent of colour scheme (how?) which may make
        # the classifier easier
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        # sys.stderr.write("Debug tl_detector got an image\n") CW this does happen
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            # CW: e.g. if light changed from yellow to red
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            # CW: in case classifier a bit noisy/unstable, debounce state
            self.last_state = self.state
            # CW: only really interested in red lights, at which we must stop
            # (but could be more cautious and act if yellow)
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
            # Note: changed to x,y coords as suggested by walkthrough video ~3min.

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        # CW: from walkthrough video ~4min suggests KDTree again as we used for
        # waypoint updater
        # Returning index of closest desired path waypoint
        return self.waypoints_tree.query([x, y], 1)[1]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # CW walkthrough video: for debug only get state from classifier
        # so can return actual state
        
	grabbing_training_images = True # make a ROS param?
	
	if not grabbing_training_images:
	    # To debug rest of system, just return known light state provided
	    # by simulator
	    return light.state
	
	# We need an image to do any image processing...
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

	# We now have an image, and a known state for it, so save
	# it as a training item with ground truth
	filename = "img" + str(self.training_image_idx) + "_" + str(light.state) + ".jpg"
	self.training_image_idx += 1
	cv2.imwrite(filename, cv_image)
	sys.stderr.write("Debug tl_detector: saved training image " + filename + "\n")
	
        #Get classification
	# TODO once we have a classifier!
        #return self.light_classifier.get_classification(cv_image)

        return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # CW: started with example code from walkthrough video at ~3min
        closest_light = None
        light_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        sys.stderr.write("Debug: tl_detector process_traffic_lights() pose ok=%s\n" % repr(self.pose is not None))
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            sys.stderr.write("Debug: tl_detector process_traffic_lights() car_wp_idx=%d\n" % car_wp_idx)
            #TODO find the closest visible traffic light (if one exists)
            
            # CW: starting with walkthrough code suggestion; gets closest in terms of
            #  waypoint index rather than actual distance, but that's fine assuming
            #  waypoints are sorted in list following the route (which they are of course)
            # List of traffic lights not too long so OK to search all
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # get stop line waypoint index
                line = stop_line_positions[i]
                # Go from coords of light stop line to nearest waypoint in our list
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                
                # find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    light_wp_idx = temp_wp_idx


        if closest_light:
            state = self.get_light_state(closest_light)
            sys.stderr.write("Debug: tl_detector process_traffic_lights() returning light_wp_idx=%d state=%d\n" % (light_wp_idx, state))
            return light_wp_idx, state
        else:
            # self.waypoints = None
            sys.stderr.write("Debug: tl_detector process_traffic_lights() returning -1 as no closest_light\n")
            return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
