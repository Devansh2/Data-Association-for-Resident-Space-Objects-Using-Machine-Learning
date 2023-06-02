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

def sensor_gcrs_extractor(header, delta_time):
    time = datetime.strptime(header['OBSTIME'], '%Y-%m-%dT%H:%M:%S.%f') \
           + timedelta(seconds=delta_time)

    sensor_obj = EarthLocation(lon=header['SITELONG'] * u.deg,
                               lat=header['SITELAT'] * u.deg, height=header['SITEALT'] * u.m)

    sensor_obj = sensor_obj.get_gcrs(Time(time))
    sensor_gcrs = sensor_obj.cartesian.xyz.to(u.m).value

    return time, sensor_gcrs


def chip_identificaton(synth_orbits, chip_id, ori_wcs_dict, sp_t0=None, tp_t0=None):
    gcrs = oe_to_gcrs(synth_orbits[0:6])  # (6,)
    # gcrs = gcrs / 1000
    rv_0 = np.copy(gcrs[np.newaxis, :])
    rv = np.zeros_like(rv_0)

    if chip_id == 1:
        rsite = synth_orbits[6:9]
        rot_axis = synth_orbits[15:18]
        cam_axis = synth_orbits[24:27]
        r_1, v_1 = two_body(mu, -sp_t0, rv_0[0, 0:3], rv_0[0, 3:6])
        rv[0, 0:3] = r_1
        rv[0, 3:6] = v_1

    elif chip_id == 2:
        rsite = synth_orbits[9:12]
        rot_axis = synth_orbits[18:21]
        cam_axis = synth_orbits[27:30]
        rv = rv_0
    elif chip_id == 3:
        rsite = synth_orbits[12:15]
        rot_axis = synth_orbits[21:24]
        cam_axis = synth_orbits[30:33]
        r_2, v_2 = two_body(mu, tp_t0 - sp_t0, rv_0[0, 0:3], rv_0[0, 3:6])
        rv[0, 0:3] = r_2
        rv[0, 3:6] = v_2

    radec = gcrs_to_radec(cam_axis)

    wcs_dict = ori_wcs_dict.copy()
    mod_CRVAL_1 = np.rad2deg(radec[0])
    mod_CRVAL_2 = np.rad2deg(radec[1])
    wcs_dict['CRVAL1'] = mod_CRVAL_1
    wcs_dict['CRVAL2'] = mod_CRVAL_2

    return rv, rsite, rot_axis, wcs_dict


def header_to_wcs(header_file):  # str

    with fits.open(header_file) as hdul:
        header = hdul[0].header

    # Gather the WCS information from the header
    wcs_dict = {'NAXIS': header['NAXIS'],
                'NAXIS1': header['CRPIX1'] * 2, 'NAXIS2': header['CRPIX2'] * 2,
                'CTYPE1': header['CTYPE1'], 'CTYPE2': header['CTYPE2'],
                'CRPIX1': header['CRPIX1'], 'CRPIX2': header['CRPIX2'],
                'CRVAL1': header['CRVAL1'], 'CRVAL2': header['CRVAL2'],
                'CD1_1': header['CD1_1'], 'CD1_2': header['CD1_2'],
                'CD2_1': header['CD2_1'], 'CD2_2': header['CD2_2']}

    return header, wcs_dict  # dict, datetime, 3-array


def gcrs_to_coe(gcrs):
    r = gcrs[0:3]
    v = gcrs[3:6]
    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)

    epsilon = (vmag * vmag / 2) - (mu / rmag)
    a = 1 / (-2 * epsilon / mu)

    rhat = r / rmag
    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    hhat = hvec / hmag

    ecc = np.cross(v, hvec) / mu - rhat
    e = np.linalg.norm(ecc)

    khat = np.array([0, 0, 1])
    i = np.arccos(np.dot(khat, hhat))

    ihat = np.array([1, 0, 0])
    nvec = np.cross(khat, hvec)
    nhat = nvec / np.linalg.norm(nvec)
    if (nhat[1] < 0):
        ohm = 2 * np.pi - np.arccos(np.dot(ihat, nhat))
    else:
        ohm = np.arccos(np.dot(ihat, nhat))

    if (ecc[2] < 0):
        omega = 2 * np.pi - np.arccos(np.dot(nhat, ecc / e))
    else:
        omega = np.arccos(np.dot(nhat, ecc / e))

    if (np.dot(r, v) < 0):
        v = 2 * np.pi - np.arccos(np.dot(ecc / e, rhat))
    else:
        v = np.arccos(np.dot(ecc / e, rhat))

    rp = a * (1 - e)
    coe = np.array([rp, e, i, ohm, omega, v])

    return coe


def coe_to_gcrs(coe):
    rp = coe[0]  # takes km
    e = coe[1]
    i = coe[2]
    ohm = coe[3]
    omega = coe[4]
    v = coe[5]

    a = rp / (1 - e)
    p = a * (1 - e * e)

    r_pqw_x = (p * np.cos(v)) / (1 + e * np.cos(v))
    r_pqw_y = (p * np.sin(v)) / (1 + e * np.cos(v))
    r_pqw = np.array([r_pqw_x, r_pqw_y, 0])

    v_pqw_x = - np.sqrt(mu / p) * np.sin(v)
    v_pqw_y = np.sqrt(mu / p) * (e + np.cos(v))

    v_pqw = np.array([v_pqw_x, v_pqw_y, 0])

    rotm = np.zeros([3, 3])
    rotm[0, 0] = np.cos(ohm) * np.cos(omega) - np.sin(ohm) * np.sin(omega) * np.cos(i)
    rotm[0, 1] = -np.cos(ohm) * np.sin(omega) - np.sin(ohm) * np.cos(omega) * np.cos(i)
    rotm[0, 2] = np.sin(ohm) * np.sin(i)

    rotm[1, 0] = np.sin(ohm) * np.cos(omega) + np.cos(ohm) * np.sin(omega) * np.cos(i)
    rotm[1, 1] = -np.sin(ohm) * np.sin(omega) + np.cos(ohm) * np.cos(omega) * np.cos(i)
    rotm[1, 2] = -np.cos(ohm) * np.sin(i)

    rotm[2, 0] = np.sin(omega) * np.sin(i)
    rotm[2, 1] = np.cos(omega) * np.sin(i)
    rotm[2, 2] = np.cos(i)

    r = np.matmul(rotm, np.transpose(r_pqw))
    v = np.matmul(rotm, np.transpose(v_pqw))

    gcrs = np.zeros([6, 1])
    gcrs[0:3, 0] = r
    gcrs[3:6, 0] = v

    return gcrs


def oe_to_gcrs(oe):
    '''
    Converts orbital elements into position and velocity.

    Parameters
    ----------
    oe : np.array (...,6)
        Orbital elements (a,ex,ey,hx,hy,l).

    Returns
    -------
    gcrs : np.array (...,6)
        Position and velocity (px,py,pz,vx,vy,vz).
    '''
    # Assign the elements
    a, ex, ey, hx, hy, l = oe.T.copy()

    # Shorthand terms
    p = a * (1 - (ex ** 2 + ey ** 2))
    s2 = 1 + hx ** 2 + hy ** 2
    alpha2 = hx ** 2 - hy ** 2
    r = p / (1 + ex * np.cos(l) + ey * np.sin(l))

    # Positions
    px = np.cos(l) + alpha2 * np.cos(l) + 2 * hx * hy * np.sin(l)
    py = np.sin(l) - alpha2 * np.sin(l) + 2 * hx * hy * np.cos(l)
    pz = 2 * (hx * np.sin(l) - hy * np.cos(l))
    pos = r / s2 * np.array([px, py, pz])

    # Velocities
    vx = np.sin(l) + alpha2 * np.sin(l) - 2 * hx * hy * np.cos(l) + ey - 2 * ex * hx * hy + alpha2 * ey
    vy = -np.cos(l) + alpha2 * np.cos(l) + 2 * hx * hy * np.sin(l) - ex + 2 * ey * hx * hy + alpha2 * ex
    vz = -2 * (hx * np.cos(l) + hy * np.sin(l) + ex * hx + ey * hy)
    # vel = -1 / s2 * np.sqrt(GM_EARTH / p) * np.array([vx, vy, vz])
    vel = -1 / s2 * np.sqrt(mu / p) * np.array([vx, vy, vz])

    # Construct the position/velocity state matrix
    gcrs = np.array([pos[0], pos[1], pos[2],
                     vel[0], vel[1], vel[2]]).T

    return gcrs


def gcrs_to_oe(gcrs):
    r = gcrs[0:3]
    v = gcrs[3:6]
    rdv = np.dot(r, v)
    rhat = r / np.linalg.norm(r)
    rmag = np.linalg.norm(r)
    hvec = np.cross(r, v)
    hmag = np.linalg.norm(hvec)
    hhat = hvec / hmag
    # vhat = (rmag * v - rdv * rhat) / hmag
    vhat = v / np.linalg.norm(v)
    vmag = np.linalg.norm(v)
    p = (hmag * hmag) / mu
    k = hhat[0] / (1 + hhat[2])
    h = -hhat[1] / (1 + hhat[2])
    kk = k * k
    hh = h * h
    s2 = 1 + h + k
    tkh = 2 * k * h
    ecc = np.cross(v, hvec) / mu - rhat
    fhat = np.zeros([3])
    fhat[0] = 1 - kk + hh
    fhat[1] = tkh
    fhat[2] = -2 * k
    ghat = np.zeros([3])
    ghat[0] = tkh
    ghat[1] = 1 + kk - hh
    ghat[2] = 2 * h

    fhat = fhat / s2
    ghat = ghat / s2

    f = np.dot(ecc, fhat)
    g = np.dot(ecc, ghat)

    X1 = np.dot(r, fhat)
    Y1 = np.dot(r, ghat)

    epsilon = (vmag * vmag / 2) - (mu / rmag)
    a = 1 / (-2 * epsilon / mu)

    L = np.arctan2(rhat[1] - vhat[0], rhat[0] + vhat[1])
    oe = np.array([a, f, g, h, k, L])

    return oe


def coe_to_eq_oe(coe):
    rp = coe[0]  # takes km
    e = coe[1]
    i = coe[2]
    ohm = coe[3]
    omega = coe[4]
    v = coe[5]

    a = rp / (1 - e)
    ex = e * np.cos(omega + ohm)
    ey = e * np.sin(omega + ohm)
    hx = np.tan(i / 2) * np.cos(ohm)
    hy = np.tan(i / 2) * np.sin(ohm)
    lv = v + omega + ohm

    eq_oe = np.array([a, ex, ey, hx, hy, lv])
    return eq_oe


@jit(nopython=True, nogil=True, cache=True)
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


# def local_chip_gen(mu, tn, rv0, rsite, wcs_dict):
#     rv = np.zeros([len(tn), 6])
#     radec = np.zeros([len(tn), 2])
#
#     for i in range(len(tn)):
#         rv[i, 0:3], rv[i, 3:6] = two_body(mu, tn[i], rv0[0, 0:3], rv0[0, 3:6])
#         radec[i, :] = gcrs_to_radec(rv[i, 0:3] - rsite[i, :] / 1000)
#
#     skycoord = generate_skycoord(np.rad2deg(radec[:, 0]), np.rad2deg(radec[:, 1]),
#                                  WCS(header=wcs_dict))
#
#     xy = radec_to_xy(skycoord, wcs_dict)
#
#     return xy


@jit(nopython=True, nogil=True, cache=True)
def numba_image_generating_gaussian(all_x, all_y, min_y, max_y, min_x, max_x, h_offset, w_offset, sigma=0.8,
                                    alpha=0.001):
    # xy : float [N 2]
    # get bbox
    step = len(min_y)

    sigma_sqr = sigma * sigma
    sigma_sqrt_pi = sigma * np.sqrt(2 * np.pi)

    patch_min_y = int(np.min(min_y) - h_offset)
    patch_max_y = int(np.max(max_y) + h_offset)
    patch_min_x = int(np.min(min_x) - w_offset)
    patch_max_x = int(np.max(max_x) + w_offset)
    y_int = int(patch_max_y - patch_min_y)
    x_int = int(patch_max_x - patch_min_x)
    patch = np.zeros((y_int, x_int))

    for i in range(step):
        curr_centre_point = np.array([all_x[i], all_y[i]])
        for y_id in range(min_y[i], max_y[i]):
            for x_id in range(min_x[i], max_x[i]):
                curr_xy = np.array([x_id, y_id])
                curr_delta = np.linalg.norm(curr_xy - curr_centre_point)
                curr_i = alpha * (1 / (sigma_sqrt_pi) * np.exp(-0.5 * curr_delta / (sigma_sqr)))
                curr_x_normed = int(x_id - patch_min_x)
                curr_y_normed = int(y_id - patch_min_y)
                patch[curr_y_normed, curr_x_normed] = patch[curr_y_normed, curr_x_normed] + curr_i

    return patch, np.array([patch_min_x, patch_min_y]), np.array([patch_max_x, patch_max_y])




def local2global_chip(global_synthetic_patch, local_s_patch, ad_L_xy_min, ad_L_xy_max, ad_G_xy_min, ad_G_xy_max):
    global_synthetic_patch[ad_G_xy_min[1]:ad_G_xy_max[1], ad_G_xy_min[0]:ad_G_xy_max[0]] = \
        local_s_patch[ad_L_xy_min[1]:ad_L_xy_max[1], ad_L_xy_min[0]:ad_L_xy_max[0]]

    return global_synthetic_patch


def adjust_xy_coord(local_xy_min, local_xy_max, global_xy_min, global_xy_max):
    # this function clip the patch generated by func_eval with global chip corners
    # patch's corners generated by current orbit estimate - local_xy_min, local_xy_max
    # global chip's corners obtained with the measurement, contain the region of interest  - global_xy_min, global_xy_max

    adjusted_global_ymin = np.maximum(local_xy_min[1], global_xy_min[1])
    adjusted_global_ymin = np.minimum(adjusted_global_ymin, global_xy_max[1])
    adjusted_global_ymin = adjusted_global_ymin - global_xy_min[1]

    adjusted_global_xmin = np.maximum(local_xy_min[0], global_xy_min[0])
    adjusted_global_xmin = np.minimum(adjusted_global_xmin, global_xy_max[0])
    adjusted_global_xmin = adjusted_global_xmin - global_xy_min[0]

    adjusted_global_ymax = np.minimum(local_xy_max[1], global_xy_max[1])
    adjusted_global_ymax = np.maximum(adjusted_global_ymax, global_xy_min[1])
    adjusted_global_ymax = adjusted_global_ymax - global_xy_min[1]

    adjusted_global_xmax = np.minimum(local_xy_max[0], global_xy_max[0])
    adjusted_global_xmax = np.maximum(adjusted_global_xmax, global_xy_min[0])
    adjusted_global_xmax = adjusted_global_xmax - global_xy_min[0]

    adjusted_local_ymin = np.maximum(local_xy_min[1], global_xy_min[1])
    adjusted_local_ymin = np.minimum(adjusted_local_ymin, global_xy_max[1])
    adjusted_local_ymin = adjusted_local_ymin - local_xy_min[1]

    adjusted_local_xmin = np.maximum(local_xy_min[0], global_xy_min[0])
    adjusted_local_xmin = np.minimum(adjusted_local_xmin, global_xy_max[0])
    adjusted_local_xmin = adjusted_local_xmin - local_xy_min[0]

    adjusted_local_ymax = np.minimum(local_xy_max[1], global_xy_max[1])
    adjusted_local_ymax = np.maximum(adjusted_local_ymax, global_xy_min[1])
    adjusted_local_ymax = adjusted_local_ymax - local_xy_min[1]

    adjusted_local_xmax = np.minimum(local_xy_max[0], global_xy_max[0])
    adjusted_local_xmax = np.maximum(adjusted_local_xmax, global_xy_min[0])
    adjusted_local_xmax = adjusted_local_xmax - local_xy_min[0]

    adjusted_local_xy_min = np.array([adjusted_local_xmin, adjusted_local_ymin])
    adjusted_local_xy_max = np.array([adjusted_local_xmax, adjusted_local_ymax])

    adjusted_global_xy_min = np.array([adjusted_global_xmin, adjusted_global_ymin])
    adjusted_global_xy_max = np.array([adjusted_global_xmax, adjusted_global_ymax])

    return adjusted_local_xy_min, adjusted_local_xy_max, adjusted_global_xy_min, adjusted_global_xy_max



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


def radec_to_xy(radec, corrected_wcs_dict):  # (2,),dataframe,dict
    pxcoord_obj = skycoord_to_pixel(radec, WCS(header=corrected_wcs_dict))
    return pxcoord_obj  # (2,)


@jit(nopython=True, nogil=True, cache=True)
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


def local_chip_gen(mu, tn, rv0, rsite, wcs_dict, h_offset, w_offset, lc_width):
    rv = np.zeros([len(tn), 6])
    radec = np.zeros([len(tn), 2])
    los = np.zeros([len(tn), 3])
    for i in range(len(tn)):
        rv[i, 0:3], rv[i, 3:6] = two_body(mu, tn[i], rv0[0, 0:3], rv0[0, 3:6])
        los[i, :] = rv[i, 0:3] - rsite[i, :] / 1000
        los[i, :] = los[i, :] / np.linalg.norm(los[i, :])
        radec[i, :] = gcrs_to_radec(rv[i, 0:3] - rsite[i, :] / 1000)

    skycoord = generate_skycoord(np.rad2deg(radec[:, 0]), np.rad2deg(radec[:, 1]),
                                 WCS(header=wcs_dict))

    xy = radec_to_xy(skycoord, wcs_dict)

    all_x = xy[0]
    all_y = int(wcs_dict['NAXIS2']) - xy[1]
    min_y = (np.round(all_y - lc_width))
    max_y = (np.round(all_y + lc_width))
    min_x = (np.round(all_x - lc_width))
    max_x = (np.round(all_x + lc_width))
    patch, min_xy, max_xy = numba_image_generating_gaussian(all_x, all_y, min_y, max_y, min_x, max_x, h_offset,
                                                            w_offset)

    return patch, min_xy, max_xy, xy, los

def gt_chip_generation(rv, rsite0, earth_rot_deg, rot_axis, wcs_dict, lc_width, dt, kernel_size, tn=None, rsite=None,
                       min_chip_pc=None, max_chip_pc=None):
    ################# first get rsite N #################
    h = int(wcs_dict['NAXIS2'])
    w = int(wcs_dict['NAXIS2'])
    if min_chip_pc is None:  # part of preprocessing
        curr_R = R.from_rotvec((0 * earth_rot_deg) * rot_axis)
        curr_R_mat = curr_R.as_matrix()
        rsite_t0 = np.matmul(curr_R_mat, rsite0)

        curr_R = R.from_rotvec((dt * earth_rot_deg) * rot_axis)
        curr_R_mat = curr_R.as_matrix()
        rsite_tN = np.matmul(curr_R_mat, rsite0)

        rf_t0, vf_t0 = two_body(mu, 0, rv[0, 0:3], rv[0, 3:6])
        rf_tN, vf_tN = two_body(mu, dt, rv[0, 0:3], rv[0, 3:6])
        radec_t0 = gcrs_to_radec(rf_t0 - rsite_t0 / 1000)
        radec_tN = gcrs_to_radec(rf_tN - rsite_tN / 1000)

        skycoord0 = generate_skycoord(np.rad2deg(radec_t0[0]), np.rad2deg(radec_t0[1]),
                                      WCS(header=wcs_dict))
        skycoordN = generate_skycoord(np.rad2deg(radec_tN[0]), np.rad2deg(radec_tN[1]),
                                      WCS(header=wcs_dict))

        xy0_fc = radec_to_xy(skycoord0, wcs_dict)
        xyN_fc = radec_to_xy(skycoordN, wcs_dict)

        min_chip_x_pc = np.minimum(xy0_fc[0], xyN_fc[0])
        max_chip_x_pc = np.maximum(xy0_fc[0], xyN_fc[0])
        min_chip_y_pc = np.minimum(h - xy0_fc[1], h - xyN_fc[1])
        max_chip_y_pc = np.maximum(h - xy0_fc[1], h - xyN_fc[1])

        min_chip_pc = np.array([min_chip_x_pc, min_chip_y_pc])
        max_chip_pc = np.array([max_chip_x_pc, max_chip_y_pc])

    ''' ############ get chip size based on xy0 and xy5 here ##############'''
    chip_w = max_chip_pc[0] - min_chip_pc[0]
    chip_h = max_chip_pc[1] - min_chip_pc[1]

    h_offset = int(np.maximum(chip_h * 0.3, kernel_size))
    w_offset = int(np.maximum(chip_w * 0.3, kernel_size))

    ext_min_chip_x_pc = np.maximum(int(np.round(min_chip_pc[0] - w_offset)), 0)
    ext_max_chip_x_pc = np.minimum(int(np.round(max_chip_pc[0] + w_offset)), w)

    ext_min_chip_y_pc = np.maximum(int(np.round(min_chip_pc[1] - h_offset)), 0)
    ext_max_chip_y_pc = np.minimum(int(np.round(max_chip_pc[1] + h_offset)), h)

    tlc = np.array([ext_min_chip_x_pc, ext_min_chip_y_pc])
    brc = np.array([ext_max_chip_x_pc, ext_max_chip_y_pc])

    # dxy = np.sqrt((max_chip_x_pc - min_chip_x_pc) ** 2 + (max_chip_y_pc - min_chip_y_pc) ** 2)
    dxy = np.linalg.norm(brc - tlc)
    d_int = 1
    V = dxy / dt

    if tn is None and rsite is None:
        ''' ############### interpolate timestamps ##############'''
        discretize_step = int(np.ceil(dxy / d_int))
        tn = np.zeros(discretize_step)
        rsite = np.zeros([discretize_step, 3])
        for i in range(discretize_step):
            tn[i] = ((i * d_int) / V)
            curr_R = R.from_rotvec((tn[i] * earth_rot_deg) * rot_axis)
            curr_R_mat = curr_R.as_matrix()
            rotated_rsite0 = np.matmul(curr_R_mat, rsite0)
            rsite[i, :] = rotated_rsite0

    point = rv[0, 0:3] - rsite[0, :] / 1000
    rho = np.linalg.norm(point)

    ''' ############### render chip from orbit params here ##############'''
    local_patch, lc_min_xy, lc_max_xy, xy_fc, los = local_chip_gen(mu, tn, rv, rsite, wcs_dict, h_offset, w_offset,
                                                              lc_width)

    ''''############ generate global patch #################'''
    row_interval = int(brc[1] - tlc[1])
    col_interval = int(brc[0] - tlc[0])
    global_data_patch = np.zeros([row_interval, col_interval])

    localRef_tlc_pc, localRef_brc_pc, globalRef_tlc_pc, globalRef_brc_pc = \
        adjust_xy_coord(lc_min_xy, lc_max_xy, tlc, brc)

    global_data_patch = local2global_chip(global_data_patch, local_patch, localRef_tlc_pc, localRef_brc_pc,
                                          globalRef_tlc_pc, globalRef_brc_pc)

    coe = gcrs_to_coe(rv[0, :])

    chip = {'tlc': tlc,
            'brc': brc,
            'xy0_fc': np.array([xy_fc[0][0], xy_fc[1][0]]),
            'xyN_fc': np.array([xy_fc[0][-1], xy_fc[1][-1]]),
            'xy0_fc_noise': np.array([xy_fc[0][0] + np.random.normal(loc=0, scale=0.5), xy_fc[1][0] + np.random.normal(loc=0, scale=0.5)]),
            'xyN_fc_noise': np.array([xy_fc[0][-1] + np.random.normal(loc=0, scale=0.5), xy_fc[1][-1] + np.random.normal(loc=0, scale=0.5)]),
            'rsite': rsite,
            'rot_axis': rot_axis,
            'tn': tn,
            'h_offset': h_offset,
            'w_offset': w_offset,
            'xy_fc': xy_fc,
            'dt': dt,
            'wcs_dict': wcs_dict,
            'rv': rv,
            'rho': rho,
            'coe': coe}
    #
    # chip = {'patch': global_data_patch,
    #         'tlc': tlc,
    #         'brc': brc,
    #         'xy0_fc': np.array([xy_fc[0][0], xy_fc[1][0]]),
    #         'xyN_fc': np.array([xy_fc[0][-1], xy_fc[1][-1]]),
    #         'rsite': rsite,
    #         'rot_axis': rot_axis,
    #         'tn': tn,
    #         'h_offset': h_offset,
    #         'w_offset': w_offset,
    #         'xy_fc': xy_fc,
    #         'dt': dt,
    #         'wcs_dict': wcs_dict,
    #         'rv': rv,
    #         'rho': rho,
    #         'coe': coe}

    return chip


def xy_to_radec(xy, wcs_dict):  # (2,),dataframe,dict
    # Convert the pixel to sky coordinates
    skycoord_obj = pixel_to_skycoord(xy[0],
                                     xy[1], WCS(header=wcs_dict))
    gcrs_obj = skycoord_obj.transform_to(GCRS())
    radec = np.array([gcrs_obj.ra.deg, gcrs_obj.dec.deg])

    return radec


def radec_to_point(radec):
    point = np.array([np.cos(radec[1]) * np.cos((radec[0])),
                      np.cos(radec[1]) * np.sin(radec[0]),
                      np.sin(radec[1])])

    return point



if __name__ == "__main__":
    # for i in range(1000):
    i = 0
    streak_len_thres = 200 # minimum length of a streak

    sp_t0_mean = 30 # time interval between the first two patches (mean)
    sp_t0_scale = 10 # time interval between the first two patches (standard deviation)
    tp_t0 = 60 # time interval between the first and third patches (fixed)
    lc_width = 3 # width of the streak
    streak_len = []
    sp_t0_stack = []

    exposed_time = 5
    padding_size = 51
    test_case = 'b'

    print('sp_t0_mean: ' + str(sp_t0_mean) + ' scale: ' + str(sp_t0_scale))
    text_data_dir = 'test_case_'+test_case+'_eq_oe_'+str(sp_t0_mean)+ '_' + str(tp_t0)
    streak_images_dir = 'test_case_'+test_case+'_eq_oe_'+str(sp_t0_mean)+ '_' + str(tp_t0)

    while i < 5:
        if test_case == 'a': #different orbital test cases
            coe_oe = np.array([np.random.uniform(6880, 8380),
                           # 1e-8, 1e-8,  # Circular orbit
                           np.random.uniform(0, 0.01), #e
                           # 0,
                           np.random.uniform(np.deg2rad(1), np.deg2rad(179)), #i
                           np.random.uniform(0, 2 * np.pi), #ohm
                           np.random.uniform(0, 2 * np.pi), #omega
                           np.random.uniform(0, 2 * np.pi)]) #v
        elif test_case == 'b':
            coe_oe = np.array([np.random.uniform(8380, 9380),
                               # 1e-8, 1e-8,  # Circular orbit
                               np.random.uniform(0.01, 0.2),  # e
                               # 0,
                               np.random.uniform(np.deg2rad(1), np.deg2rad(179)),  # i
                               np.random.uniform(0, 2 * np.pi),  # ohm
                               np.random.uniform(0, 2 * np.pi),  # omega
                               np.random.uniform(0, 2 * np.pi)])  # v
        elif test_case == 'c':
            coe_oe = np.array([np.random.uniform(8380, 9380),
                               # 1e-8, 1e-8,  # Circular orbit
                               np.random.uniform(0.2, 0.4),  # e
                               # 0,
                               np.random.uniform(np.deg2rad(1), np.deg2rad(179)),  # i
                               np.random.uniform(0, 2 * np.pi),  # ohm
                               np.random.uniform(0, 2 * np.pi),  # omega
                               np.random.uniform(0, 2 * np.pi)])  # v
        elif test_case == 'd':
            coe_oe = np.array([np.random.uniform(8380, 9380),
                               # 1e-8, 1e-8,  # Circular orbit
                               np.random.uniform(0.4, 0.6),  # e
                               # 0,
                               np.random.uniform(np.deg2rad(1), np.deg2rad(179)),  # i
                               np.random.uniform(0, 2 * np.pi),  # ohm
                               np.random.uniform(0, 2 * np.pi),  # omega
                               np.random.uniform(0, 2 * np.pi)])  # v


        sp_t0 = int(np.random.normal(loc=sp_t0_mean, scale=sp_t0_scale))

        # intrinsic parameters of the cameras
        header_file = '016_2020-12-08_091618_E_DSC_0001_header.fits'
        header, ori_wcs_dict = header_to_wcs(header_file)
        h = int(ori_wcs_dict['NAXIS2'])
        w = int(ori_wcs_dict['NAXIS1'])
        dt = 5
        earth_rot_deg = np.deg2rad(0.02) / 5

        # convert orbital state to position and velocity vectors
        eq_oe = coe_to_eq_oe(coe_oe)
        gt_rv1 = oe_to_gcrs(eq_oe)
        gt_rv1 = gt_rv1[np.newaxis, :]

        # generate the middle camera location
        rsite1 = gt_rv1[0, 0:3] / np.linalg.norm(gt_rv1[0, 0:3])
        sp_rot_axis = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        sp_rot_axis = sp_rot_axis / np.linalg.norm(sp_rot_axis)
        rot_angle = np.random.uniform(0, (np.pi / 4))
        curr_R = R.from_rotvec(rot_angle * sp_rot_axis)
        curr_R_mat = curr_R.as_matrix()

        rotated_rsite1 = np.matmul(curr_R_mat, rsite1)
        rotated_rsite1 = rotated_rsite1 * (R_EARTH)

        gt_rho1 = np.linalg.norm(gt_rv1[0, 0:3] - (rotated_rsite1 / 1000))


        '''#################### project to first site #####################'''
        r_0, v_0 = two_body(mu, -sp_t0, gt_rv1[0, 0:3], gt_rv1[0, 3:6])
        gt_rv0 = np.zeros_like(gt_rv1)
        gt_rv0[0, 0:3] = r_0
        gt_rv0[0, 3:6] = v_0
        rsite0 = gt_rv0[0, 0:3] / np.linalg.norm(gt_rv0[0, 0:3])

        fp_rot_axis = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        fp_rot_axis = fp_rot_axis / np.linalg.norm(fp_rot_axis)
        rot_angle = np.random.uniform(0, (np.pi / 4))
        curr_R = R.from_rotvec(rot_angle * fp_rot_axis)
        curr_R_mat = curr_R.as_matrix()

        rotated_rsite0 = np.matmul(curr_R_mat, rsite0)
        rotated_rsite0 = rotated_rsite0 * (R_EARTH)
        gt_rho0 = np.linalg.norm(gt_rv0[0, 0:3] - rotated_rsite0 / 1000)

        '''#################### project to third site #####################'''
        r_2, v_2 = two_body(mu, tp_t0 - sp_t0, gt_rv1[0, 0:3], gt_rv1[0, 3:6])
        gt_rv2 = np.zeros_like(gt_rv1)
        gt_rv2[0, 0:3] = r_2
        gt_rv2[0, 3:6] = v_2
        rsite2 = gt_rv2[0, 0:3] / np.linalg.norm(gt_rv2[0, 0:3])

        tp_rot_axis = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        tp_rot_axis = tp_rot_axis / np.linalg.norm(tp_rot_axis)
        rot_angle = np.random.uniform(0, (np.pi / 4))
        curr_R = R.from_rotvec(rot_angle * tp_rot_axis)
        curr_R_mat = curr_R.as_matrix()
        rotated_rsite2 = np.matmul(curr_R_mat, rsite2)
        rotated_rsite2 = rotated_rsite2 * (R_EARTH)

        gt_rho2 = np.linalg.norm(gt_rv2[0, 0:3] - rotated_rsite2 / 1000)

        '''################## randomly offset the pointing direction for camera 1 #####################'''
        fp_cam_rot_axis = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        fp_cam_rot_axis = fp_cam_rot_axis / np.linalg.norm(fp_cam_rot_axis)
        rot_angle = np.random.uniform(0, np.deg2rad(3))
        curr_R = R.from_rotvec(rot_angle * fp_cam_rot_axis)
        curr_R_mat = curr_R.as_matrix()
        point0 = gt_rv0[0, 0:3] - rotated_rsite0 / 1000
        point0 = point0 / np.linalg.norm(point0)
        cam_axis0 = np.matmul(curr_R_mat, point0)
        radec0 = gcrs_to_radec(cam_axis0)

        '''################## randomly offset the pointing direction for camera 2 #####################'''
        sp_cam_rot_axis = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        sp_cam_rot_axis = sp_cam_rot_axis / np.linalg.norm(sp_cam_rot_axis)
        rot_angle = np.random.uniform(0, np.deg2rad(3))
        curr_R = R.from_rotvec(rot_angle * sp_cam_rot_axis)
        curr_R_mat = curr_R.as_matrix()
        point1 = gt_rv1[0, 0:3] - rotated_rsite1 / 1000
        point1 = point1 / np.linalg.norm(point1)
        cam_axis1 = np.matmul(curr_R_mat, point1)
        radec1 = gcrs_to_radec(cam_axis1)

        '''################## randomly offset the pointing direction for camera 3 #####################'''
        tp_cam_rot_axis = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        tp_cam_rot_axis = tp_cam_rot_axis / np.linalg.norm(tp_cam_rot_axis)
        rot_angle = np.random.uniform(0, np.deg2rad(3))
        curr_R = R.from_rotvec(rot_angle * tp_cam_rot_axis)
        curr_R_mat = curr_R.as_matrix()
        point2 = gt_rv2[0, 0:3] - rotated_rsite2 / 1000
        point2 = point2 / np.linalg.norm(point2)
        cam_axis2 = np.matmul(curr_R_mat, point2)
        radec2 = gcrs_to_radec(cam_axis2)

        '''################## edit the intrinsic parameter for camera_1 #####################'''
        wcs_dict_1 = ori_wcs_dict.copy()
        wcs_dict_2 = ori_wcs_dict.copy()
        wcs_dict_3 = ori_wcs_dict.copy()

        mod_CRVAL_1_c1 = np.rad2deg(radec0[0])
        mod_CRVAL_2_c1 = np.rad2deg(radec0[1])
        wcs_dict_1['CRVAL1'] = mod_CRVAL_1_c1
        wcs_dict_1['CRVAL2'] = mod_CRVAL_2_c1

        '''################## edit the intrinsic parameter for camera_2 #####################'''
        mod_CRVAL_1_c2 = np.rad2deg(radec1[0])
        mod_CRVAL_2_c2 = np.rad2deg(radec1[1])
        wcs_dict_2['CRVAL1'] = mod_CRVAL_1_c2
        wcs_dict_2['CRVAL2'] = mod_CRVAL_2_c2

        '''################## edit the intrinsic parameter for camera_3 #####################'''
        mod_CRVAL_1_c3 = np.rad2deg(radec2[0])
        mod_CRVAL_2_c3 = np.rad2deg(radec2[1])
        wcs_dict_3['CRVAL1'] = mod_CRVAL_1_c3
        wcs_dict_3['CRVAL2'] = mod_CRVAL_2_c3

        '''################## project the satellite position onto cameras #####################'''
        fp_gt_xy = propogate_and_project(gt_rv0, rotated_rsite0, wcs_dict_1, fp_rot_axis, earth_rot_deg)
        sp_gt_xy = propogate_and_project(gt_rv1, rotated_rsite1, wcs_dict_2, sp_rot_axis, earth_rot_deg)
        tp_gt_xy = propogate_and_project(gt_rv2, rotated_rsite2, wcs_dict_3, tp_rot_axis, earth_rot_deg)

        '''################## get only the first pixel locations #####################'''
        fp_gt_p1 = np.array([fp_gt_xy[0][0], fp_gt_xy[1][0]])
        fp_gt_p2 = np.array([fp_gt_xy[0][1], fp_gt_xy[1][1]])

        sp_gt_p1 = np.array([sp_gt_xy[0][0], sp_gt_xy[1][0]])
        sp_gt_p2 = np.array([sp_gt_xy[0][1], sp_gt_xy[1][1]])

        tp_gt_p1 = np.array([tp_gt_xy[0][0], tp_gt_xy[1][0]])
        tp_gt_p2 = np.array([tp_gt_xy[0][1], tp_gt_xy[1][1]])

        '''################## ensure these locations are within the field of view of the cameras #####################'''
        if ((fp_gt_p1[0] < 0) or (fp_gt_p1[0] > w)) or ((fp_gt_p1[1] < 0) or (fp_gt_p1[1] > h)):
            continue

        if ((fp_gt_p2[0] < 0) or (fp_gt_p2[0] > w)) or ((fp_gt_p2[1] < 0) or (fp_gt_p2[1] > h)):
            continue

        if ((sp_gt_p1[0] < 0) or (sp_gt_p1[0] > w)) or ((sp_gt_p1[1] < 0) or (sp_gt_p1[1] > h)):
            continue

        if ((sp_gt_p2[0] < 0) or (sp_gt_p2[0] > w)) or ((sp_gt_p2[1] < 0) or (sp_gt_p2[1] > h)):
            continue

        if ((tp_gt_p1[0] < 0) or (tp_gt_p1[0] > w)) or ((tp_gt_p1[1] < 0) or (tp_gt_p1[1] > h)):
            continue

        if ((tp_gt_p2[0] < 0) or (tp_gt_p2[0] > w)) or ((tp_gt_p2[1] < 0) or (tp_gt_p2[1] > h)):
            continue

        '''################## ensure the streak is long enough  #####################'''
        fp_streak_len = np.linalg.norm(fp_gt_p1 - fp_gt_p2)
        sp_streak_len = np.linalg.norm(sp_gt_p1 - sp_gt_p2)
        tp_streak_len = np.linalg.norm(tp_gt_p1 - tp_gt_p2)

        if (fp_streak_len < streak_len_thres) or (sp_streak_len < streak_len_thres) or (tp_streak_len < streak_len_thres):
            continue

        synth_orbits = np.array([eq_oe[0], eq_oe[1], eq_oe[2], eq_oe[3], eq_oe[4], eq_oe[5],
                                 rotated_rsite0[0], rotated_rsite0[1], rotated_rsite0[2],
                                 rotated_rsite1[0], rotated_rsite1[1], rotated_rsite1[2],
                                 rotated_rsite2[0], rotated_rsite2[1], rotated_rsite2[2],
                                 fp_rot_axis[0], fp_rot_axis[1], fp_rot_axis[2],
                                 sp_rot_axis[0], sp_rot_axis[1], sp_rot_axis[2],
                                 tp_rot_axis[0], tp_rot_axis[1], tp_rot_axis[2],
                                 cam_axis0[0], cam_axis0[1], cam_axis0[2],
                                 cam_axis1[0], cam_axis1[1], cam_axis1[2],
                                 cam_axis2[0], cam_axis2[1], cam_axis2[2]])

        chip_id = 1
        fp_rv, fp_rsite, fp_rot_axis, fp_wcs_dict = chip_identificaton(synth_orbits, chip_id, ori_wcs_dict,
                                                                       sp_t0=sp_t0,
                                                                       tp_t0=tp_t0)

        gt_fp_chip = gt_chip_generation(fp_rv,
                                        fp_rsite, earth_rot_deg,
                                        fp_rot_axis, fp_wcs_dict,
                                        lc_width, exposed_time,
                                        padding_size)

        gt_fp_chip['sp_t0'] = sp_t0
        gt_fp_chip['tp_t0'] = tp_t0
        
        img_1_dir = streak_images_dir + '_' + str(i) + '_' + str(chip_id) + '.pickle'
        with open(img_1_dir, 'wb') as f:
            pickle.dump(gt_fp_chip, f)

        # this is the loading function
        # with open(img_1_dir, 'rb') as f:
        #     read_dict = pickle.load(f)

        chip_id = 2
        sp_rv, sp_rsite, sp_rot_axis, sp_wcs_dict = chip_identificaton(synth_orbits, chip_id, ori_wcs_dict,
                                                                       sp_t0=sp_t0,
                                                                       tp_t0=tp_t0)

        gt_sp_chip = gt_chip_generation(sp_rv,
                                        sp_rsite, earth_rot_deg,
                                        sp_rot_axis, sp_wcs_dict,
                                        lc_width, exposed_time,
                                        padding_size)

        gt_sp_chip['sp_t0'] = sp_t0
        gt_sp_chip['tp_t0'] = tp_t0

        img_2_dir = streak_images_dir + '_' + str(i) + '_' + str(chip_id) + '.pickle'
        with open(img_2_dir, 'wb') as f:
            pickle.dump(gt_sp_chip, f)

        chip_id = 3
        tp_rv, tp_rsite, tp_rot_axis, tp_wcs_dict = chip_identificaton(synth_orbits, chip_id, ori_wcs_dict,
                                                                       sp_t0=sp_t0,
                                                                       tp_t0=tp_t0)

        lc_width = 3
        gt_tp_chip = gt_chip_generation(tp_rv,
                                        tp_rsite, earth_rot_deg,
                                        tp_rot_axis, tp_wcs_dict,
                                        lc_width, exposed_time,
                                        padding_size)

        gt_tp_chip['sp_t0'] = sp_t0
        gt_tp_chip['tp_t0'] = tp_t0

        img_3_dir = streak_images_dir + '_' + str(i) + '_' + str(chip_id) + '.pickle'
        with open(img_3_dir, 'wb') as f:
            pickle.dump(gt_tp_chip, f)

        '''########################### save text data ################################3'''
        curr_save_dir = text_data_dir + '.txt'
        with open(curr_save_dir, 'a') as f:
            f.write(
                str(eq_oe[0] * 1000) + ' ' + str(eq_oe[1]) + ' ' + str(eq_oe[2]) + ' ' + str(eq_oe[3]) + ' ' + str(
                    eq_oe[4]) + ' ' + str(eq_oe[5]) + ' ' + \
                str(coe_oe[0]) + ' ' + str(coe_oe[1]) + ' ' + str(coe_oe[2]) + ' ' + str(coe_oe[3]) + ' ' + str(
                    coe_oe[4]) + ' ' + str(coe_oe[5]) + ' ' + \
                str(rotated_rsite0[0]) + ' ' + str(rotated_rsite0[1]) + ' ' + str(rotated_rsite0[2]) + ' ' + \
                str(rotated_rsite1[0]) + ' ' + str(rotated_rsite1[1]) + ' ' + str(rotated_rsite1[2]) + ' ' + \
                str(rotated_rsite2[0]) + ' ' + str(rotated_rsite2[1]) + ' ' + str(rotated_rsite2[2]) + ' ' + \
                str(fp_rot_axis[0]) + ' ' + str(fp_rot_axis[1]) + ' ' + str(fp_rot_axis[2]) + ' ' + \
                str(sp_rot_axis[0]) + ' ' + str(sp_rot_axis[1]) + ' ' + str(sp_rot_axis[2]) + ' ' + \
                str(tp_rot_axis[0]) + ' ' + str(tp_rot_axis[1]) + ' ' + str(tp_rot_axis[2]) + ' ' + \
                str(cam_axis0[0]) + ' ' + str(cam_axis0[1]) + ' ' + str(cam_axis0[2]) + ' ' + \
                str(cam_axis1[0]) + ' ' + str(cam_axis1[1]) + ' ' + str(cam_axis1[2]) + ' ' + \
                str(cam_axis2[0]) + ' ' + str(cam_axis2[1]) + ' ' + str(cam_axis2[2]) + '\n')

        i = i + 1


