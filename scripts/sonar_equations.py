#!/usr/bin/env python3
# ref. https://numpy.org/doc/1.18/numpy-user.pdf

# Single Beam Sonar
#  Sonar Point-Scatter Model
# Contributors: Andreina Rascon, Derek Olson, Woeng-Sug Choi

from random import random
from math import sqrt, sin, cos, pi, log
import numpy as np
import matplotlib.pyplot as plt

# unnormalized sinc function
def unnormalized_sinc(t):
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


def process_rays(distances, _normals):

    # Sonar sensor properties
    nBeams = 1
    beam_elevationAngle = 0.0175 # Beam looking down in elevation direction
    beam_azimuthAngle = 0.0 # Beam at center line in azimuth direction
    beam_elevationAngleWidth = 0.1 # radians
    beam_azimuthAngleWidth = 0.1 # radians
    ray_nElevationRays = 4
    ray_nAzimuthRays = 3

    nBuckets = 300

    if distances.shape != (ray_nElevationRays, ray_nAzimuthRays):
        print("bad distances shape ", distances.shape)
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

    # Sonar inputs
    ray_distancef2 = np.array([[15,5,10], [2,100,10], [15,15,15], [4,2,3]])
#    ray_alphaf2 = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0]]) # zz TBD
#    ray_distancef2 = distances
    ray_alphaf2 = np.zeros((ray_nElevationRays, ray_nAzimuthRays)) # TBD

    # calculated sampling periods
    max_T = np.amax(ray_distancef2)*2/soundSpeed
    _delta_f = 1/max_T
    # _delta_t = 1/(fmax - fmin)
    nFreq = int(round((fmax - fmin) / _delta_f))
    _freq1f = np.linspace(fmin,fmax,nFreq)
    time1f = np.linspace(0,max_T,nFreq) # for diagnostics plot

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
            azimuthBeamPattern2f[k,i] = (abs(unnormalized_sinc(pi * 0.884
                   / ray_azimuthAngleWidth * sin(ray_azimuthAnglesf1[i]))))**2
            elevationBeamPattern2f[k,i] = (abs(unnormalized_sinc(pi * 0.884
                   / ray_elevationAngleWidth * sin(ray_elevationAnglesf1[k]))))**2

    for k in range(ray_nElevationRays):
        for i in range(ray_nAzimuthRays):
            xi_z = random()   # generate a random number, (Gaussian noise)
            xi_y = random()   # generate another random number, (Gaussian noise)

            # angle between ray vector and object normal vector, [rad]
            alpha = ray_alphaf2[k,i]

            distance = ray_distancef2[k,i]
            amplitude = (((xi_z + 1j * xi_y)
                         / sqrt(2))
                         * (sqrt(mu * cos(alpha)**2 * distance**2
                                 * ray_azimuthAngleWidth
                                 * ray_elevationAngleWidth))
                         * azimuthBeamPattern2f[k,i]
                         * elevationBeamPattern2f[k,i])

            # Summation of Echo returned from a signal (frequency domain)
            b = int(i/ray_nAzimuthRays) # beam
            for m in range(nFreq):
                P_ray_f2c[b,m] = P_ray_f2c[b,m] + S_f1f[m] * amplitude \
                           * np.exp(-1j * K1f[m] * distance * 2) / (distance**2)

    # power level based on echo time for each beam
    P_beam_tf2 = np.zeros((nBeams, nFreq))
    #print("size1")
    #print(P_beam_tf2.shape)
    for b in range(nBeams):
        P_beam_tf2[b,:] = np.fft.ifft(P_ray_f2c[b,:])

    # power into buckets
    P_bucket_tf2 = np.zeros((nBeams, nBuckets))
    for b in range(nBeams):
        for f in range(nFreq):
            bucket = int(P_beam_tf2[b,nBuckets]*300/max_T)
            P_bucket_tf2[b, bucket] += P_beam_tf2[b,f]

    return P_bucket_tf2

