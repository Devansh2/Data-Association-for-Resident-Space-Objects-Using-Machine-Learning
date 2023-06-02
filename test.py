import numpy as np
import sys
from numba import jit
import os.path
from os import path

# import multiprocessing
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
import poliastro.core.elements
import scipy.io
from scipy.spatial.transform import Rotation as R
import pickle

from astropy.constants import GM_earth, R_earth
import math

GM_EARTH = GM_earth.value
R_EARTH = R_earth.value
mu = GM_EARTH / 1000000000

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord, generate_skycoord

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, GCRS, get_sun

import argparse
import time

def gcrs_to_radec(gcrs):
    '''
    Converts cartesian gcrs into pointing coordinates (eci_frame <-> ra/dec).
    Loose reference: www.astrosurf.com/jephem/library/li110spherCart_en.htm

    Parameters
    ----------
    gcrs : np.array (...,3) or (...,6)
        Cartesian coordinates (px,py,pz,(vx,vy,vz)).

    Returns
    -------
    radec : np.array (...,2)
        Spherical coordinates (ra, dec).
    '''
    # Separate and assign the gcrs components
    px, py, pz = gcrs.T[:3]

    # Convert positions to gcrs
    rho = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

    ra = np.arctan2(py, px)
    if ra < 0:
        ra = 2 * np.pi + ra

    dec = np.arcsin(pz / rho)

    radec = np.array([ra, dec]).T

    return radec

def two_body(mu, tau, ri, vi):
    """
    :param mu: gravitational constant(km ** 3 / sec ** 2)
    :param tau: propagation time interval(seconds)
    :param ri: initial eci position vector(kilometers)
    :param vi: initial eci velocity vector(kilometers / second)
    :return:
    rf = final eci position vector(kilometers)
    vf = final eci velocity vector(kilometers / second)
    """
    tolerance = 1.0e-12
    u = np.float64(0.0)
    uold = 100
    dtold = 100
    # imax = 20
    imax = 100

    # umax = sys.float_info.max
    # umin = -sys.float_info.max
    umax = np.float64(1.7976931348623157e+308)
    umin = np.float64(-umax)

    orbits = 0

    tdesired = tau

    threshold = tolerance * abs(tdesired)

    r0 = np.linalg.norm(ri)

    n0 = np.dot(ri, vi)

    beta = 2 * (mu / r0) - np.dot(vi, vi)

    if (beta != 0):
        umax = +1 / np.sqrt(abs(beta))
        umin = -1 / np.sqrt(abs(beta))

    if (beta > 0):
        orbits = beta * tau - 2 * n0
        orbits = 1 + (orbits * np.sqrt(beta)) / (np.pi * mu)
        orbits = np.floor(orbits / 2)

    for i in range(imax):
        q = beta * u * u
        q = q / (1 + q)
        n = 0
        r = 1
        l = 1
        s = 1
        d = 3
        gcf = 1
        k = -5

        gold = 0

        while (gcf != gold):
            k = -k
            l = l + 2
            d = d + 4 * l
            n = n + (1 + k) * l
            r = d / (d - n * r * q)
            s = (r - 1) * s
            gold = gcf
            gcf = gold + s

        h0 = 1 - 2 * q
        h1 = 2 * u * (1 - q)
        u0 = 2 * h0 * h0 - 1
        u1 = 2 * h0 * h1
        u2 = 2 * h1 * h1
        u3 = 2 * h1 * u2 * gcf / 3

        if (orbits != 0):
            u3 = u3 + 2 * np.pi * orbits / (beta * np.sqrt(beta))

        r1 = r0 * u0 + n0 * u1 + mu * u2
        dt = r0 * u1 + n0 * u2 + mu * u3
        slope = 4 * r1 / (1 + beta * u * u)
        terror = tdesired - dt

        if (abs(terror) < threshold):
            break

        if ((i > 1) and (u == uold)):
            break

        if ((i > 1) and (dt == dtold)):
            break

        uold = u
        dtold = dt
        ustep = terror / slope

        if (ustep > 0):
            umin = u
            u = u + ustep
            if (u > umax):
                u = (umin + umax) / 2
        else:
            umax = u
            u = u + ustep
            if (u < umin):
                u = (umin + umax) / 2

        if (i == imax):
            print('max iterations in twobody2 function')

    # usaved = u
    f = 1.0 - (mu / r0) * u2
    gg = 1.0 - (mu / r1) * u2
    g = r0 * u1 + n0 * u2
    ff = -mu * u1 / (r0 * r1)

    rf = f * ri + g * vi
    vf = ff * ri + gg * vi
    return rf, vf

def radec_to_xy(radec, corrected_wcs_dict):  # (2,),dataframe,dict
    pxcoord_obj = skycoord_to_pixel(radec, WCS(header=corrected_wcs_dict))
    return pxcoord_obj  # (2,)

def propogate_and_project(rv, rsite0, wcs_dict, rot_axis, earth_rot_deg):
    # generate two rsites given rotaxis
    rsite = np.zeros([2, 3])
    rsite[0, :] = rsite0

    curr_R = R.from_rotvec((5 * earth_rot_deg) * rot_axis)
    curr_R_mat = curr_R.as_matrix()
    rotated_rsite0 = np.matmul(curr_R_mat, rsite0)
    rsite[1, :] = rotated_rsite0

    rv2 = np.zeros([2, 6])
    radec = np.zeros([2, 2])

    tn = np.array([0, 5])
    for i in range(len(tn)):
        rv2[i, 0:3], rv2[i, 3:6] = two_body(mu, tn[i], rv[0, 0:3], rv[0, 3:6])
        radec[i, :] = gcrs_to_radec(rv2[i, 0:3] - rsite[i, :] / 1000)

    skycoord = generate_skycoord(np.rad2deg(radec[:, 0]), np.rad2deg(radec[:, 1]),
                                 WCS(header=wcs_dict))

    xy = radec_to_xy(skycoord, wcs_dict)

    return xy

if __name__ == "__main__":
    with open('b_120_240_nh_init.txt') as f:
        lines = f.readlines()
    
    lines  = lines[0].split()
    rv = [eval(i) for i in lines]
    print(rv)
    print(type(rv))