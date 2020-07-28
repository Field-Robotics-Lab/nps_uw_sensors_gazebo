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

# ROS-CV bridge
bridge = CvBridge()

# get angle width, height, num_rows, num_cols for one Sonar beam
def depth_camera_args():
    # args
    parser = ArgumentParser(description="Start depth camera sonar ROS node")
    parser.add_argument("width", type=float, help="Width angle")
    parser.add_argument("height", type=float, help="Height angle")
    parser.add_argument("num_cols", type=int, help="number of beams horiz")
    parser.add_argument("num_rows", type=int, help="number of beams vert")
    parser.add_argument("lobe_k1", type=float, help="Sonar lobe constant k1")
    parser.add_argument("lobe_k2", type=float, help="Sonar lobe constant k2")
    args = parser.parse_args()
    return args.width, args.height, args.num_rows, args.num_cols, \
           args.lobe_k1, args.lobe_k2

# matrix iterator given horizontal and vertical angle and count
def matrix_iterator(width, height, num_cols, num_rows):
    print("num_cols %d,num_rows %d, width: %f, height: %f"%(width, height, num_cols, num_rows))
    print(7/0) # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

    # set up horizontal and vertical starting angles and angle steps
    # returns column, row, horizontal angle, vertical angle
    if num_cols < 2:
        width_0 = 0
        d_width = 0
    else:
        width_0 = -width/2.0
        d_width = width / (num_cols - 1)

    if num_rows < 2:
        height_0 = 0
        d_height = 0
    else:
        height_0 = -height/2.0
        d_height = height / (num_rows - 1)

# perform sweep
    height_i = height_0
    for i in range(num_rows):
        width_j = width_0
        for j in range(num_cols):
            yield j, i, width_j, height_i  # yield
            width_j += d_width
        height_i += d_height

# calculate lobe power matrix
def lobe_power_matrix_constant(width, height, num_cols, num_rows, k1, k2):

    lobe_power_matrix = np.zeros((num_rows, num_cols), np.float32)

    # put lobe power constant in each each element
    for column, row, horizontal_angle, vertical_angle in matrix_iterator(
                                     width, height, num_cols, num_rows):
        ray_power = 1.0 # should calculate using angles and constants
        lobe_power_matrix[row,column] = ray_power

    return lobe_power_matrix

# currently retro power is hardcoded to 1
def retro_power_matrix_constant(num_rows, num_cols):
    retro_matrix = np.ones((num_rows, num_cols), np.float32)
    return retro_matrix

# calculate depth power matrix from depth values
def calculate_depth_power_matrix(depth_matrix,
                                 width, height, num_cols, num_rows):
    depth_power_matrix = np.zeros((num_rows, num_cols), np.float32)

    # put depth power in each each element
    for column, row, horizontal_angle, vertical_angle in matrix_iterator(
                                     width, height, num_cols, num_rows):
        depth = depth_matrix[row,column]
        depth_power = 1 # use equation
        depth_power_matrix[row, column] = depth_power
    return depth_power_matrix

def calculate_normals_power_matrix(normals_f4,
                                 width, height, num_cols, num_rows):
    normals_power_matrix = np.zeros((num_rows, num_cols), np.float32)

    # put normals angle tuple in each each element
    for column, row, horizontal_angle, vertical_angle in matrix_iterator(
                                     width, height, num_cols, num_rows):

        # calculate horizontal and vertical angles from x,y,z,1
        print("row: %d, col: %d"%(row, column))
        x,y,z,_unused = normals_f4[row, column]
        horizontal_normal = 0.0 # replace with calculation that uses xyz
        vertical_normal = 0.0
        horizontal_normal += horizontal_angle
        vertical_normal += vertical_angle

        # apply math on vertical and horizontal normals to get normals power
        normals_power_matrix[row, column] = 1.0

    return normals_power_matrix

class SonarCalculator:
    def __init__(self):

        # user inputs from the launch file.  They should match sdf.
        self.width, self.height, self.num_rows, self.num_cols, \
                      self.lobe_k1, self.lobe_k2 = depth_camera_args()

        # depth and normals power matrices from depth and normals from GPU
        self.depth_power_matrix = None
        self.normals_power_matrix = None

        # lobe power matrix
        self.lobe_power_matrix = lobe_power_matrix_constant(
                       self.width, self.height, self.num_rows, self.num_cols,
                       self.lobe_k1, self.lobe_k2)

        # retro power matrix should be calculated but retro power is hardcoded
        self.retro_power_matrix = np.ones(
                               (self.num_rows, self.num_cols), np.float32)

        # ROS subscribers
        self.depth_sub = rospy.Subscriber(
               "depth_camera_sonar_single_beam_sensor_camera/image_depth",
               Image, self.on_depth_image)
        self.normals_sub = rospy.Subscriber(
               "depth_camera_sonar_single_beam_sensor_camera/image_normals",
               Image, self.on_normals_image)

        # ROS publishers
        self.ray_pub = rospy.Publisher("image_sonar_rays_topic", Image)

    # calculate depth power matrix from Gazebo depth_image and cache it
    def on_depth_image(self, depth_image):
        rospy.loginfo(rospy.get_caller_id() + "received depth_image")
        depth_matrix = bridge.imgmsg_to_cv2(depth_image)
        print("depth image shape: ", depth_matrix.shape)
        if depth_matrix.shape != (self.num_rows, self.num_cols):
            # bad
            print("Invalid depth image shape", self.depth_matrix.shape)
            return

        self.depth_power_matrix = calculate_depth_power_matrix(depth_matrix,
                       self.width, self.height, self.num_rows, self.num_cols)
    
    # calculate normals power matrix from Gazebo normals_image
    # then calculate ray power and beam power
    def on_normals_image(self, normals_image):
        rospy.loginfo(rospy.get_caller_id() + "received normals_image")
        normals_matrix_f4 = bridge.imgmsg_to_cv2(normals_image)
        print("normals image shape: ", normals_matrix_f4.shape)
        if normals_matrix_f4.shape != (self.num_rows, self.num_cols, 4):
            # bad
            print("Invalid normals image shape", normals_matrix_f4.shape)
            return

        self.normals_power_matrix = calculate_normals_power_matrix(
                       normals_matrix_f4,
                       self.width, self.height, self.num_rows, self.num_cols)

        # use depth and normals matrices to make rays
        ray_matrix = self.images_to_rays()

        # advertise ray_matrix to ROS, keep 32FC1 format
        self.ray_pub.publish(bridge.cv2_to_imgmsg(ray_matrix, "passthrough"))
        print("Ray power matrix", ray_matrix)

        # use rays to make one beam
        # make beam, TBD
        # publish beam, TBD

    def images_to_rays(self):
        # inputs
        depth_power_matrix = self.depth_power_matrix
        normals_power_matrix = self.normals_power_matrix
        lobe_power_matrix = self.lobe_power_matrix
        retro_power_matrix = self.retro_power_matrix

        # rays
        ray_matrix = cv2.multiply(depth_power_matrix, normals_power_matrix)
        ray_matrix = cv2.multiply(ray_matrix, lobe_power_matrix)
        ray_matrix = cv2.multiply(ray_matrix, retro_power_matrix)
        return ray_matrix

if __name__ == '__main__':
    sonar_maker = SonarCalculator()
    rospy.init_node('depth_and_normals_to_sonar', anonymous=True)

    # spin until Ctrl-C
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
#    cv2.destroyAllWindows()

