import numpy as np
from numpy.linalg import inv

import os
import pickle

test_case = 'b'
sp_t0_mean = 120
tp_t0 = 240

level = 'nh'

# data_dir = os.path.join('/home/ckchng/Dropbox/lc_orbit/lc_iod_direct_exp/synthetic_exp_w_holes/input_for_fig_3/data_for_baseline/', f"{test_case}_{sp_t0_mean}_{tp_t0}_{level}_los.txt")

data_dir = 'test_case_b_eq_oe_30_60.txt'
data = np.loadtxt(data_dir)

out_dir = os.path.join('', f"{test_case}_{sp_t0_mean}_{tp_t0}_{level}_init.txt")

rho1_err = []
rho2_err = []
rho3_err = []
L1_err = []
L2_err = []
L3_err = []

r1_err = []
v_err = []

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord, generate_skycoord

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, GCRS, get_sun

const_GM_Earth = 398600435436000  # m^3/s^2


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


def read_data(input_dir):
    # this is the loading function
    # extract:
    # state vectors: r1, r2, r3, v1, v2, v3
    # los vectors (noise): los1, los2, los3,
    # observation sites: rsite1, rsite2, rsite3

    with open(input_dir, 'rb') as f:
        data = pickle.load(f)

    rv = data['rv']
    rsite = data['rsite'][0, :] / 1000
    xy0_fc = data['xy0_fc']
    wcs_dict = data['wcs_dict']
    sp_t0 = data['sp_t0']
    tp_t0 = data['tp_t0']

    # backproject xy0 to los
    radec = xy_to_radec(xy0_fc, wcs_dict)
    radec = np.deg2rad(radec)
    los = radec_to_point(radec)

    return rv, rsite, los, sp_t0, tp_t0


def gibbs(r1, r2, r3):
    small = 1e-8
    theta = 0.0
    error = 'ok'
    theta1 = 0.0

    def unit(v):
        return v / np.linalg.norm(v)

    def angl(v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    magr1 = np.linalg.norm(r1)
    magr2 = np.linalg.norm(r2)
    magr3 = np.linalg.norm(r3)

    p = np.cross(r2, r3)
    q = np.cross(r3, r1)
    w = np.cross(r1, r2)
    pn = unit(p)
    r1n = unit(r1)
    copa = np.arcsin(np.dot(pn, r1n))

    if abs(np.dot(r1n, pn)) > 0.017452406:
        error = 'not coplanar'

    d = p + q + w
    magd = np.linalg.norm(d)
    n = magr1 * p + magr2 * q + magr3 * w
    magn = np.linalg.norm(n)
    nn = unit(n)
    dn = unit(d)

    # % -------------------------------------------------------------
    # % determine if the orbit is possible.both d and n must be in the same
    # direction, and non - zero.
    # % -------------------------------------------------------------
    if (abs(magd) < small) or (abs(magn) < small) or (np.dot(nn, dn) < small):
        error = 'impossible'
    else:
        theta = angl(r1, r2)
        theta1 = angl(r2, r3)

        # % ----------- perform gibbs method to find v2 - ----------
        r1mr2 = magr1 - magr2
        r3mr1 = magr3 - magr1
        r2mr3 = magr2 - magr3
        s = r1mr2 * r3 + r3mr1 * r2 + r2mr3 * r1
        b = np.cross(d, r2)
        l = np.sqrt(const_GM_Earth / (magd * magn))
        tover2 = l / magr2
        v2 = tover2 * b + l * s

    return v2, theta, theta1, copa, error

if __name__ == "__main__":
    fp_dir = 'test_case_b_eq_oe_30_60_0_1.pickle'
    sp_dir = 'test_case_b_eq_oe_30_60_0_2.pickle'
    tp_dir = 'test_case_b_eq_oe_30_60_0_3.pickle'

    rv1, rsite1, L1, sp_t0, tp_t0 = read_data(fp_dir)
    rv2, rsite2, L2, _, _ = read_data(sp_dir)
    rv3, rsite3, L3, _, _ = read_data(tp_dir)

    gt_r1 = rv1[0, 0:3]
    gt_v1 = rv1[0, 3:6]
    gt_r2 = rv2[0, 0:3]
    gt_v2 = rv2[0, 3:6]
    gt_r3 = rv3[0, 0:3]
    gt_v3 = rv3[0, 3:6]

    gt_rho1 = np.linalg.norm(gt_r1 - rsite1)
    gt_rho2 = np.linalg.norm(gt_r2 - rsite2)
    gt_rho3 = np.linalg.norm(gt_r3 - rsite3)

    L = np.column_stack((L1, L2, L3))

    mu = const_GM_Earth / 1e9

    Rs = np.vstack((rsite1, rsite2, rsite3))

    tau1 = -sp_t0
    tau3 = tp_t0 - sp_t0

    a1 = tau3 / (tau3 - tau1)
    a1u = (tau3 * ((tau3 - tau1)**2 - tau3**2)) / (6 * (tau3 - tau1))
    a3 = -tau1 / (tau3 - tau1)
    a3u = -tau1 * ((tau3 - tau1)**2 - tau1**2) / (6 * (tau3 - tau1))

    L_inv = inv(L)

    M = np.dot(L_inv, np.transpose(Rs))

    M21 = M[1, 0]
    M22 = M[1, 1]
    M23 = M[1, 2]

    d1 = M21 * a1 - M22 + M23 * a3
    d2 = M21 * a1u + M23 * a3u

    Ccye = 2 * np.dot(L2, Rs[1, :])

    poly = np.zeros(9)
    poly[0] = 1.0
    poly[2] = -(d1**2 + d1 * Ccye + (np.linalg.norm(Rs[1, :]))**2)

    poly[5] = -(const_GM_Earth / 1e9) * (d2 * Ccye + 2 * d1 * d2)
    poly[8] = -(const_GM_Earth / 1e9)**2 * d2**2
    rootarr = np.roots(poly)

    bigr2 = -99999990.0

    for val in rootarr:
        if val > bigr2 and np.isreal(val):
            bigr2 = np.real(val)

    u = (const_GM_Earth / 1e9) / (bigr2**3)

    C1 = a1 + a1u * u
    C2 = -1
    C3 = a3 + a3u * u

    temp = -np.dot(M, np.array([C1, C2, C3]))
    rho1 = temp[0] / (a1 + a1u * u)
    rho2 = -temp[1]
    rho3 = temp[2] / (a3 + a3u * u)

    est_r1 = rho1 * L1 + Rs[0, :]
    est_r2 = rho2 * L2 + Rs[1, :]
    est_r3 = rho3 * L3 + Rs[2, :]

    # evaluate error
    rho1_err.append(abs(rho1 - gt_rho1))
    rho2_err.append(abs(rho2 - gt_rho2))
    rho3_err.append(abs(rho3 - gt_rho3))

    v2, theta, theta1, copa, error = gibbs(est_r1 * 1000, est_r2 * 1000, est_r3 * 1000)
    v2 = v2 / 1000
    v_err.append(np.linalg.norm(v2 - gt_v2))


    v2, theta, theta1, copa, error = gibbs(est_r1 * 1000, est_r2 * 1000, est_r3 * 1000)
    v2 = v2 / 1000
    v_err.append(np.linalg.norm(v2 - gt_v2))
    
    with open(out_dir, 'a') as f:
        f.write(f"{est_r2[0]} {est_r2[1]} {est_r2[2]} {v2[0]} {v2[1]} {v2[2]}")
