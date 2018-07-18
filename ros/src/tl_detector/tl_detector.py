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
import math
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
        
        # Some variables for development use
        # Could make these ROS params to avoid editing code for different experiments
        self.stub_return_ground_truth = True   # If set, cheat by just returning known state
        self.grab_training_images = True       # If set, saving image files for classifier training
        self.using_real_images = False         # True for ROS bag of real images, false for simulator
        self.training_image_idx = 0            # For training image unique filenames
        self.sim_image_grab_max_range = 50     # Only grab image when close to traffic light
        self.sim_image_grab_min_range = 3      # But not too close
        self.sim_image_grab_min_spacing = 1    # Distance gap between images
        self.image_grab_last_light = None      # Identify which light we were approaching last time
        self.image_grab_last_distance = 0      # Distance from light last time
        self.real_image_grab_decimator = 20    # Only grab fraction of simulator images
        

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
        # so can return actual state while developing other components
	
        light_state_known = light.state             # ground truth state when in simulator
        light_state_inferred = TrafficLight.UNKNOWN # default: no classifier result
	
        if not self.has_image:
            # We cannot infer state using classifier, nor grab training images
            self.prev_light_loc = None
        elif self.stub_return_ground_truth and not self.grab_training_images:
            # We have no use for the image if not faking it as we're not grabbing images right now
            pass
        else:
            # Either for training data grab or real classification or both, we need an image
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            if self.grab_training_images:
                # We now have an image, and a known state for it (if in simulator), so save
                # it as a training item with ground truth
                filename = "real" if self.using_real_images else "sim"
                filename += "_" + str(self.training_image_idx) + "_" + str(light.state) + ".jpg"
                filename = "../../../data/training_images/" + filename
                if (self.using_real_images and 
                      self.training_image_idx % self.real_image_grab_decimator != 0):
                    # Skip this image so we don't get too many when using real data
                    pass
                elif (not self.using_real_images and not self.good_position_to_save_sim_image(light)):
                    # In the simulator, only want to grab images close to traffic lights
                    # and not too close to each other, skip if not suitable right now
                    pass
                else:
                    # This is a good time to save a training image
                    cv2.imwrite(filename, cv_image)
                    sys.stderr.write("Debug tl_detector: saved training image " + filename + "\n")
                # A bit arbitrary but incrementing the image index whether or not we used
                # the image, so that if we change settings, we will in principle get the same
                # index at the same point in the simulation or .bag file replay:
                self.training_image_idx += 1

            if not self.stub_return_ground_truth:
                # Get classification, this is not a drill!
                # TODO will get UNKNOWN until we have a classifier!
                light_state_inferred = self.light_classifier.get_classification(cv_image)

        # Return either ground truth for debug or real classifier result for production
        return light_state_known if self.stub_return_ground_truth else light_state_inferred

    def good_position_to_save_sim_image(self, closest_light):
        """Considers whether we are within the range limits before a traffic light
           to save a training image, and also whether we have moved far enough since
           the last one to save a new image, so that we only end up saving a reasonable
           number of training images from the simulation and only ones that have
           traffic lights in.
           
        Returns:
           bool: True if now is a good time to save a training image"""
          
        # TODO implement
        
        # Figure out 2D Euclidean distance between us and this closest light
        delta_x = self.pose.pose.position.x - closest_light.pose.pose.position.x
        delta_y = self.pose.pose.position.y - closest_light.pose.pose.position.y
        dist_sqd = delta_x * delta_x + delta_y * delta_y
        distance = math.sqrt(dist_sqd)
        

        if (self.sim_image_grab_min_range <= distance <= self.sim_image_grab_max_range):
            # We're within a suitable range of the light we're approaching
            if (self.image_grab_last_light is None or
                self.image_grab_last_light.state != closest_light.state):
                # Definitely grab image if first light we've found, or it has changed colour
                do_grab_image = True
            elif closest_light.pose.pose.position.x != self.image_grab_last_light.pose.pose.position.x:
                # First time we've been in range for this particular light so
                # we definitely want to grab it (bit lazy to use exact equality of
                # coordinate but works OK; header.seq always zero so no use)
                #sys.stderr.write("dist=%f first time this light True\n" % distance)
                do_grab_image = True
            elif distance <= self.image_grab_last_distance - self.sim_image_grab_min_spacing:
                # We have approached the light more closely than the last time we
                # grabbed an image by enough distance for it to be worth capturing a new image
                #sys.stderr.write("dist=%f got closer so True\n" % distance)
                do_grab_image = True
            else:
                # We have not moved enough since last time, so skip this time as the
                # image will be more or less the same as the last time
                #sys.stderr.write("dist=%f not much closer so False\n" % distance)
                do_grab_image = False
        else:
            # We're not in the right distance bracket but make sure we get first
            # image when we do get within range
            self.image_grab_last_light_x = 0
            #sys.stderr.write("dist=%f outside limits so False\n" % distance)
            do_grab_image = False

        if do_grab_image:
            self.image_grab_last_light = closest_light
            self.image_grab_last_distance = distance
            
        return do_grab_image
        
            
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
        #sys.stderr.write("Debug: tl_detector process_traffic_lights() pose ok=%s\n" % repr(self.pose is not None))
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            #sys.stderr.write("Debug: tl_detector process_traffic_lights() car_wp_idx=%d\n" % car_wp_idx)
            
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
     
        if not closest_light and self.grab_training_images:
            # When playing from the bag file, we don't have pose information but nevertheless
            # we want to grab the training image so we still want to call get_light_state(),
            # so invent one with unknown state
            closest_light = TrafficLight()
            closest_light.state = TrafficLight.UNKNOWN
            light_wp_idx = -1

        if closest_light:
            state = self.get_light_state(closest_light)
            #sys.stderr.write("Debug: tl_detector process_traffic_lights() returning light_wp_idx=%d state=%d\n" % (light_wp_idx, state))
            return light_wp_idx, state
        else:
            # self.waypoints = None
            #sys.stderr.write("Debug: tl_detector process_traffic_lights() returning -1 as no closest_light\n")
            return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
