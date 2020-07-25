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
    parser.add_argument("num_rows", type=int, help="number of beams horiz")
    parser.add_argument("num_cols", type=int, help="number of beams vert")
    parser.add_argument("lobe_k1", type=float, help="Sonar lobe constant k1")
    parser.add_argument("lobe_k2", type=float, help="Sonar lobe constant k2")
    args = parser.parse_args()
    return args.width, args.height, args.num_rows, args.num_cols, \
           args.lobe_k1, args.lobe_k2

# calculate angle steps
def angle_steps(width, height, num_rows, num_cols):
    # horizontal and vertical starting angles and angle steps
    if num_rows < 2:
        width_0 = 0
        d_width = 0
    else:
        width_0 = -width/2.0
        d_width = width / (num_rows - 1)

    if num_cols < 2:
        height_0 = 0
        d_height = 0
    else:
        height_0 = -height/2.0
        d_height = height / (num_cols - 1)

    return width_0, d_width, height_0, d_height

# calculate Sonar head angle for each ray as tuples [height_angle, width_angle]
def camera_angle_matrix_constant(width, height, num_rows, num_cols):

    # horizontal and vertical starting angles and angle steps
    width_0, d_width, height_0, d_height = angle_steps(
                                        width, height, num_rows, num_cols)

    camera_angle_matrix = np.zeros((num_rows, num_cols, 2), np.float32)
    height_i = height_0
    for i in range(num_rows):
        width_j = width_0
        for j in range(num_cols):
            camera_angle_matrix[i,j] = [height_i, width_j]
            width_j += d_width
        height_i += d_height

    return camera_angle_matrix

# calculate lobe power matrix
def lobe_power_matrix_constant(width, height, num_rows, num_cols, k1, k2):

    # horizontal and vertical starting angles and angle steps
    width_0, d_width, height_0, d_height = angle_steps(
                                        width, height, num_rows, num_cols)

    lobe_power_matrix = np.zeros((num_rows, num_cols), np.float32)
    height_i = height_0
    for i in range(num_rows):
        width_j = width_0
        for j in range(num_cols):
            ray_power = 1.0 # use angles and constants instead
            lobe_power_matrix[i,j] = ray_power
            width_j += d_width
        height_i += d_height

    return lobe_power_matrix

# currently retro is hardcoded to a uniform constant
def retro_power_matrix_constant(num_rows, num_cols):
    retro_matrix = np.ones((num_rows, num_cols), np.float32)
    return retro_matrix

# calculate depth power matrix from depth values
def calculate_depth_power_matrix(depth_matrix, num_rows, num_cols):
    depth_power_matrix = np.zeros((num_rows, num_cols), np.float32)
    for i in range(num_rows):
        for j in range(num_cols):
            depth = depth_matrix[i,j]
            depth_power = 1 # use equation
            depth_power_matrix[i,j] = depth_power
    return depth_power_matrix

# calculate horizontal and vertical angles from x,y,z,1
def calculate_normals_matrix(normals_f4, num_rows, num_cols):
    normals_f2 = np.zeros((num_rows, num_cols, 2), np.float32)
    for i in range(num_rows):
        for j in range(num_cols):
            x,y,z = normals_f4[i,j,:3]
            # apply functions to get height and width angles
            normals_f2[i,j] = (0,0)
    return normals_f2

# calculate normals matrix as CV_32FC1 from CV_32FC4 x,y,z,1 image normals
def calculate_normals_power_matrix(normals_f2, num_rows, num_cols):
    normals_power_matrix = np.zeros((num_rows, num_cols), np.float32)
    for i in range(num_rows):
        for j in range(num_cols):
            normal_horiz, normal_vert = normals_f2[i,j]

            # apply math on normal_horiz, normal_vert to get some power value
            # for now hardcoded answer = 1
            normals_power_matrix[i,j] = 1.0

    return normals_power_matrix

class SonarMaker:
    def __init__(self):

        # user inputs warning: hardcoded in launch file, should match sdf
        self.width, self.height, self.num_rows, self.num_cols, \
                      self.lobe_k1, self.lobe_k2 = depth_camera_args()
        # depth and normals matrix from GPU
        self.depth_matrix = None
        self.normals_matrix = None

        # matrices that hold constants
        self.camera_angle_matrix = camera_angle_matrix_constant(
                       self.width, self.height, self.num_rows, self.num_cols)
        self.lobe_power_matrix = lobe_power_matrix_constant(
                       self.width, self.height, self.num_rows, self.num_cols,
                       self.lobe_k1, self.lobe_k2)

        # matrices that should be calculated but are not
        self.retro_power_matrix = retro_power_matrix_constant(
                                          self.num_rows, self.num_cols)

        # ROS subscribers
        self.depth_sub = rospy.Subscriber(
               "depth_camera_sonar_single_beam_sensor_camera/image_depth",
               Image, self.on_depth_image)
        self.normals_sub = rospy.Subscriber(
               "depth_camera_sonar_single_beam_sensor_camera/image_normals",
               Image, self.on_normals_image)

        # ROS publishers
        self.ray_pub = rospy.Publisher("image_sonar_rays_topic", Image)

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

    # get depth matrix from Gazebo and cache it
    def on_depth_image(self, depth_image):
        rospy.loginfo(rospy.get_caller_id() + "received depth_image")
        depth_matrix = bridge.imgmsg_to_cv2(depth_image)
        print("depth image shape: ", depth_matrix.shape)
        if depth_matrix.shape != (self.num_rows, self.num_cols):
            # bad
            print("Invalid depth image shape", self.depth_matrix.shape)
            return

        self.depth_power_matrix = calculate_depth_power_matrix(
                               depth_matrix, self.num_rows, self.num_cols)
    
    # get normals matrix from Gazebo and calculate Sonar
    def on_normals_image(self, normals_image):
        rospy.loginfo(rospy.get_caller_id() + "received normals_image")
        normals_matrix_f4 = bridge.imgmsg_to_cv2(normals_image)
        print("normals image shape: ", normals_matrix_f4.shape)
        if normals_matrix_f4.shape != (self.num_rows, self.num_cols, 4):
            # bad
            print("Invalid normals image shape", normals_matrix_f4.shape)
            return

        normals_matrix = calculate_normals_matrix(
                             normals_matrix_f4, self.num_rows, self.num_cols)
        compensated_normals_matrix = cv2.add(normals_matrix,
                                         self.camera_angle_matrix)
        self.normals_power_matrix = calculate_normals_power_matrix(
                    compensated_normals_matrix, self.num_rows, self.num_cols)

        # use depth and normals matrices to make rays
        ray_matrix = self.images_to_rays()

        # advertise ray_matrix to ROS, keep 32FC1
        self.ray_pub.publish(bridge.cv2_to_imgmsg(ray_matrix, "passthrough"))
        print("Ray power matrix", ray_matrix)

        # use rays to make one beam
        # make beam, TBD
        # publish beam, TBD

if __name__ == '__main__':
    sonar_maker = SonarMaker()
    rospy.init_node('depth_and_normals_to_sonar', anonymous=True)

    # spin until Ctrl-C
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
#    cv2.destroyAllWindows()

