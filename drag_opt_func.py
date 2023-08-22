import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import sin, cos, linspace as lin
from numpy.linalg import inv
import pandas as pd
import os
import openvsp as vsp
import degen_geom as dg
from scipy.interpolate import CubicSpline, interp1d


def spanload_opt_2d(le_y, le_z, le_x, chord, cl_req, bref, sref):
    c_avg = sref / bref
    ar = bref ** 2 / sref
    # Start the drag opt bookkeeping
    z_spacing = np.diff(le_z)  # Spacing between each geometry point
    y_spacing = np.diff(le_y)
    x_spacing = np.diff(le_x)
    xc4_sp = np.diff(le_x + chord / 4.0)
    y_vor = le_y[1:] - 0.5 * y_spacing  # Vortex locations in y and z
    z_vor = le_z[1:] - 0.5 * z_spacing
    theta_v = np.arctan(z_spacing / y_spacing)  # Angle of each vortex element
    sweep_v = np.arctan(x_spacing / y_spacing)  # LESweep of each vortex element
    sw_c4_v = np.arctan(xc4_sp / y_spacing)
    width_vor = np.sqrt(y_spacing ** 2 + z_spacing ** 2)  # Vortex strip widths
    s_p = width_vor / 2
    s = 2 * s_p / bref
    x_vor_le = le_x[1:] - np.diff(le_x) / 2
    chord_vor = chord[1:] - np.diff(chord) / 2
    x_vor_te = x_vor_le + chord_vor
    x_vor_ac = x_vor_le + 0.25 * chord_vor

    N_vor = np.shape(y_vor)[0]
    A = np.zeros((N_vor, N_vor))

    for i in range(N_vor):
        for j in range(N_vor):
            y_p = (y_vor[i] - y_vor[j]) * cos(theta_v[j]) \
                  + (z_vor[i] - z_vor[j]) * sin(theta_v[j])
            z_p = -(y_vor[i] - y_vor[j]) * sin(theta_v[j]) \
                  + (z_vor[i] - z_vor[j]) * cos(theta_v[j])
            r1 = z_p ** 2 + (y_p - s_p[j]) ** 2
            r2 = z_p ** 2 + (y_p + s_p[j]) ** 2
            a1 = ((y_p - s_p[j]) / r1 - (y_p + s_p[j]) / r2) * \
                 cos(theta_v[i] - theta_v[j]) + \
                 (z_p / r1 - z_p / r2) * sin(theta_v[i] - theta_v[j])

            y_p2 = (y_vor[i] + y_vor[j]) * cos(-theta_v[j]) \
                   + (z_vor[i] - z_vor[j]) * sin(-theta_v[j])
            z_p2 = -(y_vor[i] + y_vor[j]) * sin(-theta_v[j]) \
                   + (z_vor[i] - z_vor[j]) * cos(-theta_v[j])
            r12 = z_p2 ** 2 + (y_p2 - s_p[j]) ** 2
            r22 = z_p2 ** 2 + (y_p2 + s_p[j]) ** 2
            a2 = ((y_p2 - s_p[j]) / r12 - (y_p2 + s_p[j]) / r22) * \
                 cos(theta_v[i] + theta_v[j]) + \
                 (z_p2 / r12 - z_p2 / r22) * sin(theta_v[i] + theta_v[j])
            A[i, j] = -c_avg * (a1 + a2) / (4 * np.pi)
            # a(i, j) = -cavg / (4. * pi) * (a1 + a2);

    s_mat = np.array([list(s) for i in range(N_vor)])
    s_mat = s_mat.T
    A_bar = A * s_mat + A.T * s_mat.T  # Element-wise multiplication

    # Lift Constraint only
    A_aug = np.zeros((N_vor + 1, N_vor + 1))
    b = np.zeros((N_vor + 1, 1))
    b[-1, 0] = 0.5 * cl_req
    A_aug[:N_vor, :N_vor] = A_bar
    A_aug[N_vor, :N_vor] = s * cos(theta_v)
    A_aug[:N_vor, N_vor] = s * cos(theta_v)

    gamma = inv(A_aug) @ b
    lambdas = gamma[N_vor:]
    gamma = gamma[:N_vor]  # Column Vector

    CL_stripwise = gamma.T * 2 * s * cos(theta_v)

    CL = np.sum(CL_stripwise)

    c_n = gamma.T * c_avg / chord_vor

    # Induced drag and alpha_i calculation
    Vni_Vinf = A @ gamma  # sum over A[i, j] * gamma[j]
    CDI_stripwise = Vni_Vinf * gamma * np.atleast_2d(s).T
    alpha_I = CDI_stripwise.T / CL_stripwise
    cl_2d_stripwise = c_n[0] + 2 * np.pi * alpha_I[0]  # [int(N_vor/2)]

    cl_interp = CubicSpline(y_vor, cl_2d_stripwise)
    cl_3d_interp = CubicSpline(y_vor, c_n[0])

    c_cl_spanwise = c_n * chord_vor * cos(theta_v) / c_avg

    CDI_total = np.sum(CDI_stripwise)
    e = CL ** 2 / (np.pi * ar * CDI_total)
    results = {"cl_total": CL,
               "cl_strip": CL_stripwise.T,
               "cl_2d":cl_2d_stripwise,
               "cdi_total": CDI_total,
               "cdi_stripwise": CDI_stripwise,
               "alpha_i": alpha_I[0],
               "c_n": c_n[0],
               "e": e,
               "cl_interp_func": cl_interp,
               "cl_3d_func": cl_3d_interp,
               "y_vor": y_vor,
               "chord_vor": chord_vor,
               "theta_vor": theta_v,
               "lesweep_vor": sweep_v,
               "c4sweep_vor": sw_c4_v,
               "spanwise_ccl": c_cl_spanwise}
    return results
