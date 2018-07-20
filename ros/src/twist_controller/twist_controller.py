
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
    	wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

    	self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        # values are taken from the project walkthrough
    	kp = 0.3
    	ki = 0.1
    	kd = 0.0
    	mn = 0.0 # min throttle value
    	mx = 0.2 # max throttle value
    	self.throttle_controller = PID(kp, ki, kd, mn, mx)

    	tau = 0.5
    	ts = 0.02
    	self.vel_filter = LowPassFilter(tau, ts)

    	self.vehicle_mass = vehicle_mass
    	self.fuel_capacity = fuel_capacity
    	self.brake_deadband = brake_deadband
    	self.decel_limit = decel_limit
    	self.accel_limit = accel_limit
    	self.wheel_radius = wheel_radius

    	self.last_t = rospy.get_time()

    def control(self, ref_v, ref_yaw, cur_v, dbw_enabled):
        if not dbw_enabled:
        	self.throttle_controller.reset()
        	return 0.0, 0.0, 0.0

        cur_v_filtered = self.vel_filter.filt(cur_v)

        # 1) Steering calculation
        steering = self.yaw_controller.get_steering(ref_v, ref_yaw, cur_v_filtered)

        # 2) Throttle calculation
        error_v = ref_v - cur_v_filtered
        self.last_v = cur_v_filtered

        # sample time calculation
        current_t = rospy.get_time()
        sample_time = current_t - self.last_t
        self.last_t = current_t

        throttle = self.throttle_controller.step(error_v, sample_time)

        # 3) Brake calculation
        brake = 0.0
        if ref_v == 0.0 and cur_v_filtered < 0.1:
        	# car is about to brake
        	throttle = 0
        	brake = 400 
        elif throttle < 0.01 and error_v < 0.0:
        	throttle = 0
        	decel = max(error_v, self.decel_limit)
        	brake = abs(decel) * self.vehicle_mass * self.wheel_radius;

        return throttle, brake, steering


