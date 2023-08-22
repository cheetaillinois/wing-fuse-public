import numpy as np
import os, csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import sin, cos, tan, arctan
from numpy.linalg import inv
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib as mpl
import avlpy

mpl.use('Qt5Agg')
PI = np.pi
FILES_DIR = os.path.join(os.getcwd(), 'files')
VSP_DATA_DIR = os.path.join(os.getcwd(), 'vsp_data')


class Atmos:
    h_range = [11000, 20000, 32000, 47000, 51000, 71000, 80000]
    lapse_rates = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0]) * 1e-3
    gamma = 1.4
    g = 9.80665
    R = 287.04

    p0 = 101325  # Pa
    T0 = 288.15  # K
    rho0 = 1.225
    mu_ref = 1.716e-5
    t_ref = 273.15
    S_t = 110.4
    mu0 = mu_ref * ((T0 / t_ref) ** 1.5) * ((t_ref + S_t) / (T0 + S_t))
    S = 110.4

    def properties(self, h, feet=False):

        if feet:
            h = h / 3.281

        p, temp, rho = self.p0, self.T0, self.rho0

        for h_top in self.h_range:

            #     This section calculates height through which properties of
            #     the section need to be applied
            #     Option 1 - To the top, progress to the next section
            #     Option 2 - Until h is reached in this section

            h_previous = self.h_range[
                self.h_range.index(h_top) - 1] if h_top > 11000 else 0
            if h > h_top:
                h_next = h_top
                lapse_next = self.lapse_rates[self.h_range.index(h_top)]
                h_through = h_top - h_previous
                #                 print(h_next, lapse_next, h_through)
                h_reached = False

            else:
                h_through = h - h_previous
                lapse_next = self.lapse_rates[self.h_range.index(h_top)]
                #                 print("No", lapse_next, h_through)
                h_reached = True

            # This section calculates atmospheric properties based on lapse rate
            if lapse_next != 0.0:
                temp, t_base = temp + lapse_next * h_through, temp
                p, p_base = p * (temp / t_base) ** (
                        -self.g / self.R / lapse_next), p
                #                 print(p, (self.g / self.R / lapse_next))
                rho, rho_base = p / self.R / temp, rho
            else:
                p, p_base = p * np.exp(
                    -1 * self.g / self.R / temp * h_through), p
                rho, rho_base = p / self.R / temp, rho

            if h_reached:
                break

        mu = self.mu_ref * ((temp / self.t_ref) ** 1.5) * (
                (self.t_ref + self.S_t) / (temp + self.S_t))
        a = np.sqrt(self.gamma * self.R * temp)
        atm_prop = {"p": p, "rho": rho, "T": temp, "a": a, "mu": mu}
        return atm_prop

    def __init__(self, height, feet=False, prec=4):
        self.h = height
        atm_prop = self.properties(height, feet)
        self.p = np.round(atm_prop["p"], prec) if prec else atm_prop["p"]
        self.rho = np.round(atm_prop["rho"], prec) if prec else atm_prop["rho"]
        self.temp = np.round(atm_prop["T"], prec) if prec else atm_prop["T"]
        self.a = np.round(atm_prop["a"], prec) if prec else atm_prop["a"]
        self.mu = np.round(atm_prop["mu"], prec) if prec else atm_prop["mu"]


class ElementForcesOutput:
    """
    Output Wrapper for Element Forces
    """

    def __init__(self, fo_file):
        """

        :param fo_file: file name for the element forces output file to parse
        """

        with open(fo_file, "r") as f:
            self.num_lines = sum(1 for line in f)

        with open(fo_file, "r") as f:
            # Find start of main data
            data = []
            line = f.readline()
            while line != '':

                if "I        X           Y           Z           DX        Slope        dCp" in line:
                    line = f.readline()
                    while line.strip():
                        split_data = line.split()
                        line_data = split_data[1:]
                        line_data = [float(d) for d in line_data]
                        data.append(line_data)
                        line = f.readline()

                line = f.readline()

        data_edited = [arr for arr in data if len(arr) > 0]

        mat = np.array(data_edited)
        # print(data_edited)
        self.x = mat[:, 0]
        self.y = mat[:, 1]
        self.z = mat[:, 2]
        self.DX = mat[:, 3]
        self.slope = mat[:, 4]
        self.dCp = mat[:, 5]

        # Separate dCps by y and x
        yx_Cp = np.array([self.y, self.x, self.dCp]).T
        self.y_span_stations = np.unique(np.abs(self.y))
        span_Cp_x = {}
        chord_span = np.zeros_like
        for y_st in self.y_span_stations:
            span_Cp_x[y_st] = yx_Cp[np.where(yx_Cp[:, 0] == y_st), 1:][0]

        self.span_cp_x = span_Cp_x

    def plot(self):
        return


def reyn_calc(mach, height, chord, feet):
    if feet:
        chord /= 3.281
    atm = Atmos(height, feet, 10)
    rho = atm.rho
    v = mach * atm.a
    print(atm.mu)
    reyn = rho * v * chord / atm.mu
    return reyn


def cl_calc(weight, pounds, mach, height, feet, sref):
    if feet:
        sref /= 3.281 ** 2
    if pounds:
        weight /= 2.204
    weight *= 9.8
    atm = Atmos(height, feet, 10)
    rho = atm.rho
    v = mach * atm.a
    cl = (2 * weight) / (rho * (v ** 2) * sref)
    return cl, rho, v


def cp_flat(alpha, x):
    return 4 * alpha * np.sqrt((1 - x) / x)


def extract_sections(tri_file, save_file, array_output=False,
                     convex_hull=True, plot=False, normal_axis: str = 'y',
                    save_dir=os.path.join(FILES_DIR, "extracted_sections")):
    n_points = int(np.loadtxt(tri_file, max_rows=1)[0])
    xsecs_data = np.loadtxt(tri_file, skiprows=1, max_rows=n_points)
    if normal_axis.lower() == 'x':
        locs = np.unique(xsecs_data[:, 0])

        xsec_list = []
        for i, loc in enumerate(locs):
            xsec = xsecs_data[xsecs_data[:, 0] == loc][:, 1:]
            xsec_list.append(xsec)

    elif normal_axis.lower() in ['y', 'z']:
        locs = np.unique(xsecs_data[:, 1])

        xsec_list = []
        for i, loc in enumerate(locs):
            xsec = np.stack((xsecs_data[xsecs_data[:, 1] == loc]
                             [:, 0],
                             xsecs_data[xsecs_data[:, 1] == loc][:, -1]), 1)
            xsec_list.append(xsec)
    else:
        raise ValueError("Normal axis must be x, y, or z")
    section_list = []
    save_file_list = []

    for i, xsec in enumerate(xsec_list):
        if convex_hull:
            hull = ConvexHull(xsec)
            hull_outline = np.array(
                list(zip(xsec[hull.vertices, 0], xsec[hull.vertices, 1])))
            xsec_to_save = hull_outline
        else:
            raise Exception("Sorry not done yet")

        if plot:
            plt.plot(xsec_to_save[:, 0], xsec_to_save[:, 1],
                     label=f"Section {i}")

        save_file_list.append(os.path.join(save_dir, f"{save_file}_{i}.dat"))
        np.savetxt(os.path.join(save_dir, f"{save_file}_{i}.dat"), xsec_to_save)
        section_list.append(xsec_to_save)

    op = {"save_locs": save_file_list}
    if plot:
        plt.legend()
        plt.show()

    if array_output:
        op["arrays"] = section_list

    return op


def lifting_line(theta, y,  c_theta, ainc_theta, CLa_theta, b, aspect, lb_le=0):
    n_range = np.linspace(1, len(theta), len(theta))
    n_grid, theta_grid = np.meshgrid(n_range, theta)
    i_grid, c_grid = np.meshgrid(n_range, c_theta)
    y_grid, cla_grid = np.meshgrid(y, CLa_theta * cos(lb_le))
    # Construct vector Q such that P A_n = Q
    q_theta = np.atleast_2d(c_theta / (4 * b) * CLa_theta * cos(lb_le) * ainc_theta).T
    q_alpha = np.atleast_2d(c_theta / (4 * b) * CLa_theta * cos(lb_le)).T

    # Construct influence matrix P such that P A_n = Q
    p_new_theta = sin(n_grid * theta_grid) + (c_grid * cla_grid) * n_grid * sin(
        n_grid * theta_grid) / (4 * b * sin(theta_grid))

    # Solve for A_n
    A_n = inv(p_new_theta) @ q_theta
    A_n_alpha = inv(p_new_theta) @ q_alpha
    A_n_alpha_arr = A_n_alpha.T[0]
    A_n_arr = A_n.T[0]

    ind_alpha = np.sum(
        n_range * A_n_arr * sin(n_grid * theta_grid) / sin(theta_grid), axis=1)
    cl_dist = 4 * b / c_theta * np.sum(A_n_arr * sin(n_grid * theta_grid), axis=1)
    cdi_theta = cl_dist * ind_alpha

    CL_total = PI * aspect * A_n_arr[0]
    CDi_total = PI * aspect * sum(n_range * A_n_arr ** 2)
    e = CL_total ** 2 / (PI * aspect * CDi_total)
    results = {"theta": theta,
               "y": y,
               "cl_theta": cl_dist,
               "cdi_theta": cdi_theta,
               "aind_theta": ind_alpha,
               "CL_total": CL_total,
               "CL_alpha": PI * aspect * A_n_alpha_arr[0],
               "CDi_total": CDi_total,
               "e": e,
               "A_n": A_n.T[0],
               "A_n_alpha": A_n_alpha_arr}
    return results


def get_su2_cp(file_name):
    cpcol = 10  # column with cp values
    xcol = 6  # column with x values
    ycol = 7  # column with y values
    zcol = 8  # column with z values
    # functions to get values as lists
    extract_x = lambda x: x[xcol]
    extract_cp = lambda x: x[cpcol]
    extract_z = lambda x: x[zcol]
    with open(file_name, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    data.pop(0)  # removing the headers
    if len(data) == 0:
        return -1, -1, -1, -1
    y = -1 * float(data[0][ycol])  # extracting the y position of the slice
    # creating float lists for x, cp, and z
    x = np.array([extract_x(xi) for xi in data]).astype(float)
    cp = np.array([extract_cp(xi) for xi in data]).astype(float)
    z = np.array([extract_z(xi) for xi in data]).astype(float)

    # normalizing x from zero to one
    x = x - np.min(x)
    xmax = np.max(x)
    x = np.divide(x, xmax)
    # dividing the slice into top and bottom surfaces based on approximate chord
    zfront = z[np.argmin(x)]
    zback = z[np.argmax(x)]
    multiplier = -zfront + zback  # slope of the centerline
    centerline = x * multiplier + zfront
    xtop = []
    xbot = []
    cptop = []
    cpbot = []
    for i in range(x.size):
        if z[i] < centerline[i]:
            xbot.append(x[i])
            cpbot.append(cp[i])
        else:
            xtop.append(x[i])
            cptop.append(cp[i])
    # convert into np arrays to easily sort values
    xtop = np.array(xtop)
    xbot = np.array(xbot)
    cptop = np.array(cptop)
    cpbot = np.array(cpbot)
    # sort xtop and cptop and xbot and cpbot together
    idxtop = np.argsort(xtop)
    idxbot = np.argsort(xbot)
    xtop = xtop[idxtop]
    xbot = xbot[idxbot]
    cptop = cptop[idxtop]
    cpbot = cpbot[idxbot]

    return xtop, xbot, cptop, cpbot, y


if __name__ == "__main__":

    # external_bounds_file = "files/NewExternals.tri"
    # external_bounds = extract_sections(external_bounds_file, "new_externals",
    #                                    array_output=True,
    #                                    convex_hull=True, plot=True)
    # h = 35000
    # chord = 15
    # feet = True
    # mach = 0.773
    # reyn = reyn_calc(mach, h, chord, feet)
    #
    # el_forces = ElementForcesOutput(os.path.join(FILES_DIR,
    #                                              "sc_test_fe.txt"))
    # fig = plt.figure("3D Cp plot")
    # ax = plt.axes(projection='3d')
    # surf = ax.scatter(el_forces.x, el_forces.y, el_forces.z, c=el_forces.dCp,
    #                   cmap='viridis')
    # fig.colorbar(surf)
    # plt.show()

    # plt.figure("Cpx at 0")
    # plt.plot(el_forces.span_cp_x[el_forces.y[0]][:, 0] / np.max(
    #     el_forces.span_cp_x[el_forces.y[0]][:, 0]),
    #          el_forces.span_cp_x[el_forces.y[0]][:, 1], color='cornflowerblue')
    # x = el_forces.span_cp_x[el_forces.y[0]][:, 0] / np.max(
    #     el_forces.span_cp_x[el_forces.y[0]][:, 0])
    #
    # cl_0 = np.trapz(el_forces.span_cp_x[el_forces.y[0]][:, 1],
    #                 x=el_forces.span_cp_x[el_forces.y[0]][:, 0] / np.max(
    #                     el_forces.span_cp_x[el_forces.y[0]][:, 0]))
    # alpha_0 = cl_0 / 2 / np.pi
    # plt.plot(x, cp_flat(alpha_0, x), color='orange')
    #
    # strip = avlpy.StripForcesOutput(os.path.join(FILES_DIR,
    #                                              "gridconvflat",
    #                                              "wb_75_40",
    #                                              "flat_plate_strip"))
    # strip.parse_lines()
    # right_half = strip.strips[0]
    #
    # weight = 195000
    # pounds = True
    # sref = 1840
    # cl_req = cl_calc(weight, pounds, mach, h, feet, sref)
    b = 8
    c = 1
    # CL_alpha = 2 * PI
    v_infty = 80
    aspect = b / c
    n_theta = 41
    theta = np.linspace(0, PI, n_theta)[1:-1]
    CL_alpha = 2 * PI * np.ones_like(theta)
    # n_range = n_range[:len(theta)]
    c_theta = c * np.ones_like(theta)  # Constant chord
    CL_req = 0.6
    y = b / 2 * cos(theta)
    ainc_theta = (2 * CL_req * np.sqrt(1 - (2 * y / b) ** 2) / PI ** 2) + \
                 CL_req / (PI * aspect)

    llt_results = lifting_line(theta, y, c_theta, ainc_theta, CL_alpha, b, aspect)
    ainc_diff = llt_results["A_n"][0] / llt_results["A_n_alpha"][0]
    ainc_new = ainc_theta - ainc_diff
    llt_new = lifting_line(theta, y, c_theta, ainc_new, CL_alpha, b, aspect)
    plt.plot(y, llt_results["cl_theta"])
    plt.show()
