#!/usr/bin/env python
# We use Python 2 instead of python3 bacause ROS uses Python 2.
# ref. https://numpy.org/doc/1.18/numpy-user.pdf

# Single Beam Sonar
#  Sonar Point-Scatter Model
# Contributors: Andreina Rascon, Derek Olson, Woeng-Sug Choi

from random import random
from math import sqrt, sin, cos, pi, log, acos
import numpy as np
import matplotlib.pyplot as plt
import rospy

# diagnostics
def _show_plots(nBeams, ray_nElevationRays, ray_nAzimuthRays,
                nFreq, nBuckets, time1f, P_beam_tf2, P_bucket_tf2):

    # Plots
    plt.figure(figsize=(14,10), dpi=80)
    plt.suptitle("%d beam(s), %d elevation rays, %d azimuth rays "
                 "%d frequencies, %d buckets"%(nBeams,
                   ray_nElevationRays, ray_nAzimuthRays, nFreq, nBuckets))

    # inverse fast fourier transform
    # figure (1)
    plt.subplot(2,2,1)
    plt.title("Power based on echo time")
    plt.grid(True)
    plt.plot(time1f, P_beam_tf2[0,:], linewidth=0.5)
    plt.xlabel('Time, [s]')
    plt.ylabel('Pressure, [Pa]')

    # Sound Pressure Level of Echo Level
    # figure (2)
    SPLf1 = 20 * np.log(np.abs(P_beam_tf2[0,:])) # sound pressure level, [dB]
    plt.subplot(2,2,2)
    plt.title("Sound pressure level based on echo time")
    plt.grid(True)
    plt.plot(time1f, SPLf1, linewidth=0.5)
    plt.xlabel('Time, [s]')
    plt.ylabel('Sound Pressure Level, [Pa]')

    # image for each nFreq
    plt.subplot(2,2,3)
    plt.title("Heatplot sound pressure level (SPL) based on frequency number")
    plt.xlabel('Beam number')
    plt.ylabel('Inverse FFT frequency number')
    plt.imshow(P_beam_tf2.T, aspect="auto")

    # image for each nBuckets
    plt.subplot(2,2,4)
    plt.title("Bucketed heatplot SPL based on bucketed frequency number")
    plt.xlabel('Beam number')
    plt.ylabel('Inverse FFT frequency bucket number')
    plt.imshow(P_bucket_tf2.T, aspect="auto")

    plt.show()

# unnormalized sinc function
def _unnormalized_sinc(t):
    try:
        return sin(t)/t
    except ZeroDivisionError:
        return 1.0

## physics constants
soundSpeed = 1500.0 # [m/s]
mu = 10e-4          # Surface Reflectivity

# Input Parameters
# The BlueView P900-45 is used as an example sonar for the purposes of
# demonstrating the point-scatter model

# Sonar properties
sonarFreq = 900E3 # Sonar frquency
bandwidth = 29.5e4 # [Hz]
freqResolution = 100e2
fmin = sonarFreq - bandwidth/2*4 # Calculated requency spectrum
fmax = sonarFreq + bandwidth/2*4 # Calculated requency spectrum

def _textf3(text, f):
    return "%s: %f, %f, %f"%(text,f[0],f[1],f[2])

# incidence angle is target's normal angle compensated by ray's
# azimuth and elevation
def _ray_incidence(azimuth, elevation, normalf4):
    # ray normal from camera azimuth and elevation
    camera_x = cos(-azimuth)*cos(elevation)
    camera_y = sin(-azimuth)*cos(elevation)
    camera_z = sin(elevation)
    ray_normal = np.array([camera_x, camera_y, camera_z])
    print("ray_normal", ray_normal)

    # target normal with axes compensated to camera axes zz verify
    target_normal = np.array([normalf4[2], -normalf4[0], -normalf4[1]])

    print("target_normal", target_normal)

    # dot product
#zz this fails because the dot product gets out of range.
#zz    return pi - acos(ray_normal.dot(target_normal))

    dot_product = ray_normal.dot(target_normal)
#    print("dot product: "%dot_product)
#    print(dot_product)
#    if dot_product < -1.0 or dot_product > 1.0:
#        text="ray_normal %f, %f, %f"%(ray_normal[0], ray_normal[1], ray_normal[2])
#        rospy.logwarn(_textf3("ray_normal", ray_normal))
#        rospy.logwarn(_textf3("target_normal", target_normal))
#        rospy.logwarn("dot product = %f"%dot_product)
#        dot_product=-1.0
    return pi - acos(dot_product)

def process_rays(ray_distancesf2, ray_normalsf2_4, show_plots=False):
#    for i in range(4):
#        rospy.logwarn("ray distance %d: %f, %f, %f"%(i,
#             ray_distancesf2[i,0], ray_distancesf2[i,1], ray_distancesf2[i,2]))
#
#        rospy.logwarn("ray normal %d: %f, %f, %f, %f"%(i,
#             ray_normalsf2_4[i,0,0], ray_normalsf2_4[i,0,1], ray_normalsf2_4[i,0,2], ray_normalsf2_4[i,0,3]))
#        rospy.logwarn("ray normal %d: %f, %f, %f, %f"%(i,
#             ray_normalsf2_4[i,1,2], ray_normalsf2_4[i,1,0], ray_normalsf2_4[i,1,1], ray_normalsf2_4[i,1,3]))
#        rospy.logwarn("ray normal %d: %f, %f, %f, %f"%(i,
#             ray_normalsf2_4[i,2,2], ray_normalsf2_4[i,2,0], ray_normalsf2_4[i,2,1], ray_normalsf2_4[i,2,3]))
#        rospy.logwarn(" ")

    for i in range(4):
        rospy.logwarn("ray distance %d: %f, %f, %f"%(i,
             ray_distancesf2[i,0], ray_distancesf2[i,1], ray_distancesf2[i,2]))

        rospy.logwarn("ray normal %d: %f, %f, %f"%(i,
             ray_normalsf2_4[i,0,2], ray_normalsf2_4[i,0,0], ray_normalsf2_4[i,0,1]))
        rospy.logwarn("ray normal %d: %f, %f, %f"%(i,
             ray_normalsf2_4[i,1,2], ray_normalsf2_4[i,1,0], ray_normalsf2_4[i,1,1]))
        rospy.logwarn("ray normal %d: %f, %f, %f"%(i,
             ray_normalsf2_4[i,2,2], ray_normalsf2_4[i,2,0], ray_normalsf2_4[i,2,1]))
        rospy.logwarn(" ")


    # Sonar sensor properties
    nBeams = 1
    beam_elevationAngle = 0.0175 # Beam looking down in elevation direction
    beam_azimuthAngle = 0.0 # Beam at center line in azimuth direction
    beam_elevationAngleWidth = 0.1 # radians
    beam_azimuthAngleWidth = 0.1 # radians
    ray_nElevationRays = 4
    ray_nAzimuthRays = 3

    nBuckets = 300

    if ray_distancesf2.shape != (ray_nElevationRays, ray_nAzimuthRays):
        print("bad distances shape ", ray_distancesf2.shape)
        return np.zeros(nBeams,nBuckets)

    # calculated Sonar sensor properties
    ray_elevationAnglesf1 = beam_elevationAngle + np.linspace(
                   -beam_elevationAngleWidth / 2, beam_elevationAngleWidth / 2,
                   ray_nElevationRays)
    ray_azimuthAnglesf1 = beam_azimuthAngle + np.linspace(
                   -beam_azimuthAngleWidth / 2, beam_azimuthAngleWidth / 2,
                   ray_nAzimuthRays)
    ray_elevationAngleWidth = beam_elevationAngleWidth/(ray_nElevationRays - 1)
    ray_azimuthAngleWidth = beam_azimuthAngleWidth/(ray_nAzimuthRays - 1)

    # azimuth angles for nAzimuthRays * nBeams
    full_sweep_azimuthAnglesf1 = beam_azimuthAngle + np.linspace(
                   nBeams * -beam_azimuthAngleWidth / 2,
                   nBeams * beam_azimuthAngleWidth / 2,
                   nBeams * ray_nAzimuthRays)

    # calculated sampling periods
    max_T = np.amax(ray_distancesf2)*2/soundSpeed
    _delta_f = 1/max_T

    # _delta_t = 1/(fmax - fmin)
    nFreq = int(round((fmax - fmin) / _delta_f))

    # reduce nFreq because calculated nFreq is too large for looping
    nFreq=10000
    _freq1f = np.linspace(fmin,fmax,nFreq)

    # calculated physics
    _absorption = 0.0354 # [dB/m]
    _attenuation = _absorption*log(10)/20
    _kw1f = 2*pi*_freq1f/soundSpeed # wave vector
    K1f = _kw1f + 1j*_attenuation # attenuation constant K1f

    # Transmit spectrum, frequency domain
    S_f1f = 1e11 * np.exp(-(_freq1f - sonarFreq)**2 * pi**2 / bandwidth**2)

    # Point Scattering model
    # Echo level using the point scatter model for P(f) and P(t) for beams
    P_ray_f2c = np.zeros((nBeams, nFreq), dtype=np.complex_)
    azimuthBeamPattern2f = np.zeros((ray_nElevationRays,ray_nAzimuthRays))
    elevationBeamPattern2f = np.zeros((ray_nElevationRays,ray_nAzimuthRays))
    for k in range(ray_nElevationRays):
        for i in range(ray_nAzimuthRays):
            azimuthBeamPattern2f[k,i] = (abs(_unnormalized_sinc(pi * 0.884
                 / ray_azimuthAngleWidth * sin(ray_azimuthAnglesf1[i]))))**2
            elevationBeamPattern2f[k,i] = (abs(_unnormalized_sinc(pi * 0.884
                 / ray_elevationAngleWidth * sin(ray_elevationAnglesf1[k]))))**2

    incidences_f2 = np.zeros((ray_nElevationRays, ray_nAzimuthRays), dtype=np.float32) # diagnostics message
    for k in range(ray_nElevationRays):
        for i in range(ray_nAzimuthRays):
            xi_z = random()   # generate a random number, (Gaussian noise)
            xi_y = random()   # generate another random number, (Gaussian noise)

            # ray in beam
            r = i % nBeams

            # angle between ray vector and object normal vector, [rad]
            incidence = _ray_incidence(full_sweep_azimuthAnglesf1[i],
                                       ray_elevationAnglesf1[k],
                                       ray_normalsf2_4[k, i])
            incidences_f2[k,i] = incidence

            distance = ray_distancesf2[k,i]
            amplitude = (((xi_z + 1j * xi_y)
                         / sqrt(2))
                         * (sqrt(mu * cos(incidence)**2 * distance**2
                                 * ray_azimuthAngleWidth
                                 * ray_elevationAngleWidth))
                         * azimuthBeamPattern2f[k,i]
                         * elevationBeamPattern2f[k,i])

            # Summation of Echo returned from a signal (frequency domain)
            b = int(i/ray_nAzimuthRays) # beam
            for m in range(nFreq):
                P_ray_f2c[b,m] = P_ray_f2c[b,m] + S_f1f[m] * amplitude \
                           * np.exp(-1j * K1f[m] * distance * 2) / (distance**2)

    # diagnostics incidences
    for i in range(4):
        rospy.logwarn("incidences %d: %f, %f, %f"%(i,
             incidences_f2[i,0], incidences_f2[i,1], incidences_f2[i,2]))

    # power level based on echo time for each beam
    P_beam_tf2 = np.zeros((nBeams, nFreq))
    for b in range(nBeams):
        P_beam_tf2[b,:] = np.fft.ifft(P_ray_f2c[b,:])

    # power into buckets
    P_bucket_tf2 = np.zeros((nBeams, nBuckets), dtype=np.float32)
    for b in range(nBeams):
        for f in range(nFreq):
            bucket = int(f*nBuckets/nFreq)
            P_bucket_tf2[b, bucket] += P_beam_tf2[b,f]

    if show_plots:
        time1f = np.linspace(0,max_T,nFreq) # for diagnostics plot
        _show_plots(nBeams, ray_nElevationRays, ray_nAzimuthRays,
                    nFreq, nBuckets, time1f, P_beam_tf2, P_bucket_tf2)

    return P_bucket_tf2, incidences_f2

# test
if __name__ == '__main__':
    # Note that dimensions must match dimensions hardcoded in process_rays.
    ray_distancesf2 = np.array([[15,5,10], [2,100,10], [15,15,15], [4,2,3]])
    ray_normalsf2_4= np.array([[[1.0,0,0,0], [1,0,0,0], [1,0,0,0]],
                               [[1,0,0,0], [1,0,0,0], [1,0,0,0]],
                               [[1,0,0,0], [1,0,0,0], [1,0,0,0]],
                               [[1,0,0,0], [1,0,0,0], [1,0,0,0]]])

    _image, _incidences = process_rays(ray_distancesf2, ray_normalsf2_4, True)

