#!/usr/bin/env python
#import sys
from argparse import ArgumentParser
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from sonar_equations import process_rays

# references:
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
# http://docs.ros.org/melodic/api/cv_bridge/html/python/index.html

# subscriber name constants, should be replaced with sdf inputs
IMAGE_DEPTH_TOPIC = "depth_camera_sonar_sensor_camera/image_depth"
IMAGE_NORMALS_TOPIC = "depth_camera_sonar_sensor_camera/image_normals"

# publisher name constants, should be replaced with sdf inputs
RAY_IMAGE_TOPIC = "sonar_ray_image"
RAY_POINT_CLOUD_TOPIC = "sonar_ray_point_cloud"
BEAM_IMAGE_TOPIC = "sonar_beam_image"
BEAM_POINT_CLOUD_TOPIC = "sonar_beam_point_cloud"

class SonarNode:
    def __init__(self):

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

    # calculate normals power matrix from Gazebo normals_image
    # then calculate ray power and beam power
    def on_normals_image(self, normals_image):
        rospy.loginfo(rospy.get_caller_id() + " received normals_image")
        self.normals_matrix_f4 = self.bridge.imgmsg_to_cv2(normals_image)
        print("normals image shape: ", self.normals_matrix_f4.shape)

        # generate all the outputs
        beam_matrix = process_rays(self.depth_matrix,
                                   self.normals_matrix_f4)

        # advertise ray_power_matrix to ROS, inheriting the 32FC1 format
        self.ray_pub.publish(self.bridge.cv2_to_imgmsg(beam_matrix,
                                                       "passthrough"))

if __name__ == '__main__':
    rospy.init_node('depth_and_normals_to_sonar', anonymous=True)
    ros_sonar_node = SonarNode()

    # spin until Ctrl-C
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

