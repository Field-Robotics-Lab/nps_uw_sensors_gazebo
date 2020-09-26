#!/usr/bin/env python
#import sys
from argparse import ArgumentParser
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

# references:
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
# http://docs.ros.org/melodic/api/cv_bridge/html/python/index.html

# Notes:
# Arguments passed here from the launch file must match arguments in SDF.

# subscriber name constants, should be replaced with sdf inputs
IMAGE_DEPTH_TOPIC = "depth_camera_sonar_sensor_camera/image_depth"
IMAGE_NORMALS_TOPIC = "depth_camera_sonar_sensor_camera/image_normals"

# publisher name constants, should be replaced with sdf inputs
RAY_IMAGE_TOPIC = "sonar_ray_image"
RAY_POINT_CLOUD_TOPIC = "sonar_ray_point_cloud"
BEAM_IMAGE_TOPIC = "sonar_beam_image"
BEAM_POINT_CLOUD_TOPIC = "sonar_beam_point_cloud"

# get angle beam width, vert_count, horiz_count for one Sonar beam
def depth_camera_args():
    # args
    parser = ArgumentParser(description="Start depth camera sonar ROS node")
    parser.add_argument("beam_width", type=float, help="Beam width angle")
    parser.add_argument("horiz_count", type=int, help="number of rays horiz")
    parser.add_argument("vert_count", type=int, help="number of rays vert")
    parser.add_argument("lobe_k1", type=float, help="Sonar lobe constant k1")
    parser.add_argument("lobe_k2", type=float, help="Sonar lobe constant k2")
    parser.add_argument("__name", type=str, help="launch adds name, unused")
    parser.add_argument("__log", type=str, help="launch adds log, unused")
    args = parser.parse_args()
    return args.beam_width, args.horiz_count, args.vert_count, \
           args.lobe_k1, args.lobe_k2

# Given a horizontal beam width and the horizontal and vertical ray count,
# calculates the vertical beam width then iteratively sweeps through ray points
# from top to bottom to return the following index and angular information
# about each ray: horiz_index, vert_index, horiz_angle, vert_angle.
def ray_matrix_iterator(beam_width, horiz_count, vert_count):
    beam_height = beam_width * vert_count / horiz_count
    print("beam_width: %f, beam_height: %f, horiz_count %d, vert_count %d"%(
                          beam_width, beam_height, horiz_count, vert_count))

    # set up horizontal and vertical starting angles and angle steps
    if horiz_count < 2:
        horiz_angle_0 = 0
        horiz_angle = 0
    else:
        horiz_angle_0 = -beam_width/2.0
        horiz_angle = beam_width / (horiz_count - 1)

    if vert_count < 2:
        height_0 = 0
        vert_angle = 0
    else:
        height_0 = -beam_height/2.0
        vert_angle = beam_height / (vert_count - 1)

    # perform sweep
    vert_angle = height_0
    for vert_index in range(vert_count):
        horiz_angle = horiz_angle_0
        for horiz_index in range(horiz_count):
            yield horiz_index, vert_index, horiz_angle, vert_angle  # yield
            horiz_angle += horiz_angle
        vert_angle += vert_angle

# calculate lobe power matrix
def lobe_power_matrix_constant(beam_width, horiz_count, vert_count, k1, k2):

    lobe_power_matrix = np.zeros((vert_count, horiz_count), np.float32)

    # put lobe power constant in each each element
    for column, row, horizontal_angle, vertical_angle in ray_matrix_iterator(
                                     beam_width, horiz_count, vert_count):
        ray_power = 1.0 # should calculate using angles and constants
        lobe_power_matrix[row,column] = ray_power

    return lobe_power_matrix

# currently retro power is hardcoded to 1
def retro_power_matrix_constant(vert_count, horiz_count):
    retro_matrix = np.ones((vert_count, horiz_count), np.float32)
    return retro_matrix

# calculate depth power matrix from depth values
def calculate_depth_power_matrix(depth_matrix,
                                 beam_width, horiz_count, vert_count):
    depth_power_matrix = np.zeros((vert_count, horiz_count), np.float32)

    # put depth power in each each element
    for column, row, horizontal_angle, vertical_angle in ray_matrix_iterator(
                                     beam_width, horiz_count, vert_count):
        depth = depth_matrix[row,column]
        depth_power = depth # use equation instead
        depth_power_matrix[row, column] = depth_power
    return depth_power_matrix

def calculate_normals_power_matrix(normals_f4,
                                 beam_width, horiz_count, vert_count):
    normals_power_matrix = np.zeros((vert_count, horiz_count), np.float32)

    # put normals angle tuple in each each element
    for column, row, horizontal_angle, vertical_angle in ray_matrix_iterator(
                                     beam_width, horiz_count, vert_count):

        # calculate horizontal and vertical angles from x,y,z,1
        x,y,z,_unused = normals_f4[row, column]
        horizontal_normal = 0.0 # replace with calculation that uses xyz
        vertical_normal = 0.0
        horizontal_normal += horizontal_angle
        vertical_normal += vertical_angle

        # apply math on vertical and horizontal normals to get normals power
        normals_power_matrix[row, column] = 1.0

    return normals_power_matrix

# calculate the ray power matrix from the power matrices
def powers_to_rays(depth_power_matrix, normals_power_matrix,
                   lobe_power_matrix, retro_power_matrix):

    # rays
    ray_power_matrix = cv2.multiply(depth_power_matrix, normals_power_matrix)
    ray_power_matrix = cv2.multiply(ray_power_matrix, lobe_power_matrix)
    ray_power_matrix = cv2.multiply(ray_power_matrix, retro_power_matrix)
    return ray_power_matrix

class SonarNode:
    def __init__(self):

        # user inputs from the launch file.  They should match sdf.
        self.beam_width, self.horiz_count, self.vert_count, \
                      self.lobe_k1, self.lobe_k2 = depth_camera_args()

        # depth and normals power matrices from depth and normals from GPU
        self.depth_matrix = None
        self.depth_power_matrix = None
        self.normals_power_matrix = None

        # lobe power matrix
        self.lobe_power_matrix = lobe_power_matrix_constant(
                    self.beam_width, self.horiz_count, self.vert_count,
                    self.lobe_k1, self.lobe_k2)

        # retro power matrix should be calculated but retro power is hardcoded
        self.retro_power_matrix = np.ones(
                               (self.vert_count, self.horiz_count), np.float32)

        # ROS subscribers
        self.depth_sub = rospy.Subscriber(IMAGE_DEPTH_TOPIC,
                                            Image, self.on_depth_image)
        self.normals_sub = rospy.Subscriber(IMAGE_NORMALS_TOPIC,
                                            Image, self.on_normals_image)

        # ROS publishers
        self.ray_pub = rospy.Publisher(RAY_IMAGE_TOPIC, Image, queue_size=10)

        # ROS-CV bridge
        self.bridge = CvBridge()

    # calculate depth power matrix from Gazebo depth_image and cache it
    def on_depth_image(self, depth_image):
        rospy.loginfo(rospy.get_caller_id() + " received depth_image")
        self.depth_matrix = self.bridge.imgmsg_to_cv2(depth_image)
        print("depth image shape: ", self.depth_matrix.shape)
        if self.depth_matrix.shape != (self.vert_count, self.horiz_count):
            # bad
            print("Invalid depth image shape", self.depth_matrix.shape)
            return

        self.depth_power_matrix = calculate_depth_power_matrix(
                    self.depth_matrix,
                    self.beam_width, self.horiz_count, self.vert_count)
    
    # calculate normals power matrix from Gazebo normals_image
    # then calculate ray power and beam power
    def on_normals_image(self, normals_image):
        rospy.loginfo(rospy.get_caller_id() + " received normals_image")
        normals_matrix_f4 = self.bridge.imgmsg_to_cv2(normals_image)
        print("normals image shape: ", normals_matrix_f4.shape)
        if normals_matrix_f4.shape != (self.vert_count, self.horiz_count, 4):
            # bad
            print("Invalid normals image shape", normals_matrix_f4.shape)
            return

        self.normals_power_matrix = calculate_normals_power_matrix(
                    normals_matrix_f4,
                    self.beam_width, self.horiz_count, self.vert_count)

        # generate all the outputs
        self.generate_outputs()

    # generate all the outputs
    def generate_outputs(self):

        # use depth and normals matrices to make rays
        ray_power_matrix = powers_to_rays(self.depth_power_matrix,
                                    self.normals_power_matrix,
                                    self.lobe_power_matrix,
                                    self.retro_power_matrix)

        # advertise ray_power_matrix to ROS, inheriting the 32FC1 format
        self.ray_pub.publish(self.bridge.cv2_to_imgmsg(ray_power_matrix,
                                                       "passthrough"))
#        print("Ray power matrix", ray_power_matrix)

#        # create and publish the ray point cloud
#        ray_cloud = ray_point_cloud(ray_power_matrix,
#                    self.beam_width, self.horiz_count, self.vert_count)
#        self.ray_cloud_pub.publish(ray_cloud)
#
#        # create and publish the beam point cloud
#        beam_cloud = beam_point_cloud(ray_power_matrix,
#                    self.beam_width, self.horiz_count, self.vert_count)
#        self.beam_cloud_pub.publish(ray_cloud)

if __name__ == '__main__':
    rospy.init_node('depth_and_normals_to_sonar', anonymous=True)
    ros_sonar_node = SonarNode()

    # spin until Ctrl-C
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

