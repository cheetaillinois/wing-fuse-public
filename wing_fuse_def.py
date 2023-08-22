import json
import numpy as np
import os
from numpy import pi, sin, cos, tan, arctan, rad2deg, diff as steps
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from avl_spanload_gen import avl_sim_spanload
from drag_opt_func import spanload_opt_2d
from gen_utils import ElementForcesOutput
from pyPanair.preprocess import wgs_creator
from panair_utils import place_airfoil, run_panin
import openvsp as vsp
import scienceplots

plt.style.use(["science", "ieee"])

mpl.use('Qt5Agg')
FILES_DIR = os.path.join(os.getcwd(), 'files')
WINGFUSE_DIR = os.path.join(FILES_DIR, "WingFuse")
VSP_DATA_DIR = os.path.join(os.getcwd(), 'vsp_data')
PI = np.pi
COS2_CENT = (PI ** 2 - 4) / (4 * PI ** 2)


class WingFuse:

    def __init__(self, args: dict,
                 name="WingFuseDefault", is_inverse: bool = False):
        self.cl_alpha_spline = None
        self.args = args
        self.chord_b = None
        self.x_le_b = None
        self.y_body = None
        self.body_semiwidth = None
        self.chord_w = None
        self.results_avl = None
        self.results_2d_span = {}
        self.name = name
        self.y = None
        self.x_te = None
        self.x_le = None
        self.x_le_w = None
        self.x_te_w = None
        if "cg" in args.keys():
            self.cg = args["cg"]
        else:
            self.cg = None
        if not is_inverse:

            self.fuse_length = args['fuselage length']
            self.span = args['wing span']
            self.root_chord = args['wing root chord']
            self.wing_x_loc = args['wing loc ratio'] * \
                              (self.fuse_length - self.root_chord)
            self.taper_ratio = args['wing taper']
            self.wing_le_sweep = args['wing sweep']
            self.fuse_front_fraction = args['fuselage front fraction']
            self.fuse_rear_fraction = args['fuselage rear fraction']
            self.wing_ref_area = 0.5 * self.span * \
                                 self.root_chord * (1 + self.taper_ratio)
            self.wing_aspect = self.span ** 2 / self.wing_ref_area
            self.avg_chord = self.wing_ref_area / self.span
            self.fuse_extra_area = 0.5 * self.span * \
                                   (self.fuse_length - self.root_chord) * \
                                   self.fuse_front_fraction
            self.fuse_junc_area = self.span * self.fuse_front_fraction * \
                                  self.root_chord * \
                                  (1 + 0.5 * (1 - self.taper_ratio) *
                                   self.fuse_front_fraction)
            self.fuse_aspect = (self.span * self.fuse_front_fraction) ** 2 / \
                               (self.fuse_extra_area + self.fuse_junc_area)
            self.total_aspect = self.span ** 2 / \
                                (self.wing_ref_area + self.fuse_extra_area)

            tan_tesweep = tan(self.wing_le_sweep) + \
                          2 * (self.taper_ratio - 1) * self.root_chord / self.span
            self.wing_te_sweep = arctan(tan_tesweep)
            tan_c4sweep = tan(self.wing_le_sweep) + \
                          0.5 * (self.taper_ratio - 1) * self.root_chord / self.span
            self.wing_c4_sweep = arctan(tan_c4sweep)
            tan_c2sweep = tan(self.wing_le_sweep) + \
                          (self.taper_ratio - 1) * self.root_chord / self.span
            self.wing_c2_sweep = arctan(tan_c2sweep)
            self.wing_centroid = self.span * (self.taper_ratio + 0.5) / \
                                 (3 * (self.taper_ratio + 1))
            self.fuse_ex_centroid = COS2_CENT * self.span * \
                                    self.fuse_front_fraction
            self.r_a = self.fuse_extra_area / self.wing_ref_area
            self.ce_ratio = self.fuse_ex_centroid / self.wing_centroid
            self.phi = (self.r_a * self.ce_ratio + 1) / (self.r_a + 1)
            self.phi_min = 1 / (self.r_a + 1)
            k = 3 * COS2_CENT * (self.taper_ratio + 1) / (
                        self.taper_ratio + 0.5)
            self.phi_max = (self.r_a * k + 1) / (self.r_a + 1)
            self.g_phi = (self.phi - self.phi_min) / \
                         (self.phi_max - self.phi_min)
        else:
            self.span = args['wing span']
            self.root_chord = args['wing root chord']
            self.taper_ratio = args['wing taper']
            self.wing_le_sweep = args['wing sweep']
            self.wing_ref_area = 0.5 * self.span * \
                                 self.root_chord * (1 + self.taper_ratio)
            self.r_a = args['wing-fuse area ratio']
            self.fuse_extra_area = self.r_a * self.wing_ref_area
            self.g_phi = args['blending parameter']
            r_a, lb = self.r_a, self.taper_ratio
            self.phi_min = 1 / (self.r_a + 1)
            k = 3 * COS2_CENT * (self.taper_ratio + 1) / (
                    self.taper_ratio + 0.5)
            self.phi_max = (self.r_a * k + 1) / (self.r_a + 1)
            self.phi = self.phi_min + self.g_phi * (self.phi_max - self.phi_min)
            phi = self.phi
            f = (phi * (r_a + 1) - 1) * (lb + 0.5) / \
                (3 * r_a * (lb + 1) * COS2_CENT)
            self.fuse_front_fraction = f
            self.fuse_rear_fraction = f
            l_f = 2 * self.fuse_extra_area / (f * self.span) + self.root_chord
            self.fuse_length = l_f
            self.avg_chord = self.wing_ref_area / self.span
            self.fuse_junc_area = self.span * self.fuse_front_fraction * \
                                  self.root_chord * \
                                  (1 + 0.5 * (1 - self.taper_ratio) *
                                   self.fuse_front_fraction)
            self.fuse_aspect = (self.span * self.fuse_front_fraction) ** 2 / \
                               (self.fuse_extra_area + self.fuse_junc_area)
            self.total_aspect = self.span ** 2 / \
                                (self.wing_ref_area + self.fuse_extra_area)
            self.body_semiwidth = self.span * self.fuse_front_fraction * 0.25 * 1.0
            self.body_area = 2 * (self.root_chord * self.body_semiwidth -
                                  (1 - self.taper_ratio) * self.root_chord * self.body_semiwidth ** 2 / self.span +
                                  (self.fuse_length - self.root_chord) *
                                  (self.body_semiwidth / 2 + self.span * self.fuse_front_fraction / (4 * PI) *
                                   sin(2 * PI * self.body_semiwidth / (self.span * self.fuse_front_fraction))))
            self.body_aspect = 4 * self.body_semiwidth ** 2 / self.body_area
            tan_tesweep = tan(self.wing_le_sweep) + \
                          2 * (lb - 1) * self.root_chord / self.span
            self.wing_te_sweep = arctan(tan_tesweep)
            tan_c4sweep = tan(self.wing_le_sweep) + \
                          0.5 * (self.taper_ratio - 1) * self.root_chord / self.span
            self.wing_c4_sweep = arctan(tan_c4sweep)
            tan_c2sweep = tan(self.wing_le_sweep) + \
                          (self.taper_ratio - 1) * self.root_chord / self.span
            self.wing_c2_sweep = arctan(tan_c2sweep)
            self.wing_x_loc = args['wing loc ratio'] * \
                              (self.fuse_length - self.root_chord)
            self.wing_aspect = self.span ** 2 / self.wing_ref_area

    def le_func(self, y):
        ffrac = self.fuse_front_fraction
        
        return self.wing_x_loc + np.abs(y) * np.tan(self.wing_le_sweep) - \
                    (np.abs(y) <= (0.5 * self.span * ffrac)) * \
                    self.wing_x_loc * cos(pi * np.abs(y) / (self.span * ffrac)) ** 2

    def le_w_func(self, y):
        
        return self.wing_x_loc + np.abs(y) * np.tan(self.wing_le_sweep)

    def le_z_func(self, y):
        return np.zeros_like(y)

    def te_func(self, y):
        rfrac = self.fuse_rear_fraction \
            if self.fuse_rear_fraction is not None else self.fuse_front_fraction
        return self.wing_x_loc + self.root_chord + \
                    np.abs(y) * np.tan(self.wing_te_sweep) + \
                    (np.abs(y) <= (0.5 * self.span * rfrac)) * \
                    (self.fuse_length - self.wing_x_loc - self.root_chord) * \
                    cos(pi * np.abs(y) / (self.span * rfrac)) ** 2

    def te_w_func(self, y):
        return self.wing_x_loc + self.root_chord + \
               np.abs(y) * np.tan(self.wing_te_sweep)

    def chord_func(self, y):
        return self.te_func(y) - self.le_func(y)

    def chord_w_func(self, y):
        return self.te_w_func(y) - self.le_w_func(y)

    def leading_edge(self, y):
        ffrac = self.fuse_front_fraction
        self.x_le = self.wing_x_loc + y * np.tan(self.wing_le_sweep) - \
                    (y <= (0.5 * self.span * ffrac)) * \
                    self.wing_x_loc * cos(pi * y / (self.span * ffrac)) ** 2
        self.x_le_w = self.wing_x_loc + y * np.tan(self.wing_le_sweep)

    def trailing_edge(self, y):
        rfrac = self.fuse_rear_fraction \
            if self.fuse_rear_fraction is not None else self.fuse_front_fraction
        self.x_te = self.wing_x_loc + self.root_chord + \
                    y * np.tan(self.wing_te_sweep) + \
                    (y <= (0.5 * self.span * rfrac)) * \
                    (self.fuse_length - self.wing_x_loc - self.root_chord) * \
                    cos(pi * y / (self.span * rfrac)) ** 2
        self.x_te_w = self.wing_x_loc + self.root_chord + \
                      y * np.tan(self.wing_te_sweep)

    def chord_def(self, y):
        # self.chord = self.root_chord + \
        #              y * 2 * self.root_chord * \
        #              (self.taper_ratio - 1) / self.span + \
        #              (y <= (0.5 * self.span * self.fuse_fraction)) * \
        #              (self.fuse_length - self.root_chord) * \
        #              cos(pi * y / (self.span * self.fuse_fraction)) ** 2
        try:
            self.chord = self.x_te - self.x_le
            self.chord_w = self.x_te_w - self.x_le_w
        except:
            self.leading_edge(self.y)
            self.trailing_edge(self.y)
            self.chord = self.x_te - self.x_le
            self.chord_w = self.x_te_w - self.x_le_w

    def le_te_chord_def(self, y, plot=False):
        if y[-1] != self.span / 2:
            raise ValueError("Span Space is incorrectly defined")
        self.y = y
        self.leading_edge(self.y)
        self.trailing_edge(self.y)
        self.chord_def(self.y)

        if plot:
            plt.figure(f"Planform of {self.name}")
            plt.plot(self.y, self.x_le * 1,'-', color='blue',
                     label='Wing-Fuselage')
            plt.plot(self.y * -1, self.x_le * 1,'-', color='blue')
            plt.plot(self.y, self.x_te * 1,'-', color='blue')
            plt.plot(self.y * -1, self.x_te * 1,'-', color='blue')

            plt.plot(self.y, self.x_le_w * 1, '--', color='red',
                     label='Ref. Wing')
            plt.plot(self.y * -1, self.x_le_w * 1, '--', color='red')
            plt.plot(self.y, self.x_te_w * 1, '--', color='red')
            plt.plot(self.y * -1, self.x_te_w * 1, '--', color='red')
            plt.plot([self.y[-1], self.y[-1]],
                     [self.x_le[-1] * 1, self.x_te[-1] * 1],'-',
                     color='blue')
            plt.plot([self.y[-1] * -1, self.y[-1] * -1],
                     [self.x_le[-1] * 1, self.x_te[-1] * 1],'-',
                     color='blue')

            plt.fill_between(self.y, self.x_le * 1, self.x_le_w * 1,
                             where=(self.y <= (
                                     0.5 * self.span * self.fuse_front_fraction)),
                             hatch='/', facecolor='cornflowerblue',
                             label='Extra Area')
            plt.fill_between(self.y * -1, self.x_le * 1, self.x_le_w * 1,
                             where=(self.y <= (
                                     0.5 * self.span * self.fuse_front_fraction)),
                             hatch='/', facecolor='cornflowerblue')
            plt.fill_between(self.y, self.x_te * 1, self.x_te_w * 1,
                             where=(self.y <= (
                                     0.5 * self.span * self.fuse_front_fraction)),
                             hatch='/', facecolor='cornflowerblue')
            plt.fill_between(self.y * -1, self.x_te * 1, self.x_te_w * 1,
                             where=(self.y <= (
                                     0.5 * self.span * self.fuse_front_fraction)),
                             hatch='/', facecolor='cornflowerblue')
            plt.fill_between(self.y, self.x_te * 1, self.x_te_w * 1,
                             where=(self.y <= (
                                     0.5 * self.span * self.fuse_front_fraction)),
                             hatch='/', facecolor='cornflowerblue')
            plt.fill_between(self.y * -1, self.x_te * 1, self.x_te_w * 1,
                             where=(self.y <= (
                                     0.5 * self.span * self.fuse_front_fraction)),
                             hatch='/', facecolor='cornflowerblue')
            plt.legend(loc="upper right")
            plt.ylim((self.wing_x_loc + 0.6 * self.span,
                      self.wing_x_loc - 0.4 * self.span))
            plt.xlim((-0.6 * self.span, 1.1 * self.span))
            plt.annotate(fr"$g_\phi = {self.g_phi:.2f}$", (0.75, 0.1),
                         xycoords="axes fraction")
            plt.annotate(fr"$r_a = {self.r_a:.2f}$", (0.75, 0.2),
                         xycoords="axes fraction")
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.show()

    def json_dump(self):
        args_to_save = {'wing span': self.span,
                        'wing root chord': self.root_chord,
                        'wing taper': self.taper_ratio,
                        'wing sweep': self.wing_le_sweep,
                        'wing loc ratio': self.wing_x_loc /
                                          (self.fuse_length - self.root_chord),
                        'wing-fuse area ratio': self.r_a,
                        'blending parameter': self.g_phi,
                        "cg": self.cg}
        save_path = str(os.path.join(WINGFUSE_DIR, self.name, 'args.json'))
        if not os.path.isdir(os.path.join(WINGFUSE_DIR, self.name)):
            os.mkdir(os.path.join(WINGFUSE_DIR, self.name))
        with open(save_path, 'w') as f:
            json.dump(args_to_save, f)

    def spanload_eval(self, cl_req, plot=False, wing_only=False,
                      body_only=False):
        if self.y is None or self.chord is None:
            raise ValueError(f"le_te_chord_def has not been run for object\
             {self.name}")
        if not wing_only:
            r2d_span = spanload_opt_2d(self.y, np.zeros_like(self.x_le),
                                   self.x_le, self.chord, cl_req,
                                   self.span, self.wing_ref_area)
        elif wing_only and not body_only:
            r2d_span = spanload_opt_2d(self.y, np.zeros_like(self.x_le),
                                       self.x_le_w, self.chord_w, cl_req,
                                       self.span, self.wing_ref_area)
        # elif body_only and not wing_only:
        #     self.y_body = self.y[np.where(self.y <= self.body_semiwidth)]
        #     self.x_le_b = self.le_func(self.y_body)
        #     self.chord_b = self.chord_func(self.y_body)
        #     r2d_span = spanload_opt_2d(self.y_body, np.zeros_like(self.x_le_b),
        #                                self.x_le_b, self.chord_b, cl_req,
        #                                self.body_semiwidth * 2, self.wing_ref_area)
        self.results_2d_span = r2d_span

        if plot:

            plt.figure(f"2D Spanload results for {self.name}")
            plt.plot(r2d_span["y_vor"], r2d_span["c_n"],
                     label=r'Spanload $C_L$')
            plt.plot(r2d_span["y_vor"], r2d_span["cl_2d"],
                     color='orange', label=r'2D $C_l$')
            plt.legend()

    def avl_spanload_eval(self, cl_req, plot=False, wing_only=False,
                          save_skeleton=False,
                          body_only=False, alphas=[0.0]):
        if self.y is None or self.chord is None:
            raise ValueError(f"le_te_chord_def has not been run for object\
             {self.name}")
        elif len(self.results_2d_span.keys()) == 0:
            self.spanload_eval(cl_req, plot=plot, wing_only=wing_only, body_only=body_only)

        cl_cspline = self.results_2d_span["cl_interp_func"]

        if not wing_only and not body_only:
            res_avl = avl_sim_spanload(name=self.name, mach=0.0,
                                       sref=self.wing_ref_area, cref=self.avg_chord,
                                       bref=self.span,
                                       cg_wb=(0.5 * self.chord[0], 0,
                                              0) if self.cg is None else self.cg,
                                       nspan=40, nchord=30,
                                       sections_new_dir=os.path.join(WINGFUSE_DIR, self.name),
                                       base_dir=WINGFUSE_DIR,
                                       avl_binpath=r"C:\Work\CHEETA\OpenVSP_Projects\AVL\avl.exe",
                                       le_x_cspline=self.le_func,
                                       le_z_cspline=self.le_z_func,
                                       chord_cspline=self.chord_func,
                                       cl_cspline=cl_cspline,
                                       cla_func=self.cl_alpha_spline,
                                       alphas=alphas)
        elif wing_only and not body_only:
            res_avl = avl_sim_spanload(name=self.name, mach=0.0,
                                       sref=self.wing_ref_area,
                                       cref=self.avg_chord,
                                       bref=self.span,
                                       cg_wb=(0.5 * self.chord[0], 0,
                                              0) if self.cg is None else self.cg,
                                       nspan=40, nchord=30,
                                       sections_new_dir=os.path.join(WINGFUSE_DIR,
                                                                     self.name),
                                       base_dir=WINGFUSE_DIR,
                                       avl_binpath=r"C:\Work\CHEETA\OpenVSP_Projects\AVL\avl.exe",
                                       le_x_cspline=self.le_w_func,
                                       le_z_cspline=self.le_z_func,
                                       chord_cspline=self.chord_w_func,
                                       cl_cspline=cl_cspline)
        elif body_only and not wing_only:
            res_avl = avl_sim_spanload(name=self.name, mach=0.0,
                                       sref=self.wing_ref_area,
                                       cref=self.avg_chord,
                                       bref=self.body_semiwidth * 2,
                                       cg_wb=(0.5 * self.chord[0], 0,
                                              0) if self.cg is None else self.cg,
                                       nspan=40, nchord=30,
                                       sections_new_dir=os.path.join(WINGFUSE_DIR,
                                                                     self.name),
                                       base_dir=WINGFUSE_DIR,
                                       avl_binpath=r"C:\Work\CHEETA\OpenVSP_Projects\AVL\avl.exe",
                                       le_x_cspline=self.le_func,
                                       le_z_cspline=self.le_z_func,
                                       chord_cspline=self.chord_func,
                                       cl_cspline=cl_cspline)
        else:
            raise ValueError("wing_only and body_only flags can not be \
             True at the same time")

        self.results_avl = res_avl

        cl_half_2d = np.trapz(self.results_2d_span["c_n"] \
                              * cos(self.results_2d_span["theta_vor"]) \
                              * self.results_2d_span["chord_vor"] / self.avg_chord,
                              x=self.results_2d_span["y_vor"] * 2 / self.span)
        el_forces = self.results_avl["unp"].elementForces[0]
        strip = self.results_avl["unp"].stripForces[0]
        strip.parse_lines()
        right_half = strip.strips[0]
        cl_half_avl = np.trapz(right_half.c_cl / self.avg_chord,
                               x=right_half.yle * 2 / self.span)
        self.results_avl["spanwise_ccl"] = right_half.c_cl / self.avg_chord
        self.results_avl["spanwise_y"] = right_half.yle
        self.results_avl["spanwise_cl"] = right_half.cl

        if plot:
            plt.figure("Spanwise lift distributions")
            plt.plot(self.results_2d_span["y_vor"],
                     self.results_2d_span["c_n"] * self.results_2d_span["chord_vor"] / self.avg_chord,
                     label=fr"2D Spanload $C_l = {np.round(cl_half_2d, 3)}$")
            plt.plot(right_half.yle, right_half.c_cl / self.avg_chord, color='red',
                     label=fr"AVL $C_l = {np.round(cl_half_avl, 3)}$")
            plt.xlabel("Spanwise y-coordinate")
            plt.ylabel(r"$cC_L / c_{avg}$")
            plt.legend()

            fig_avl_res = plt.figure("3D Cp plot")
            ax = plt.axes(projection='3d')
            surf = ax.scatter(el_forces.x, el_forces.y, el_forces.dCp,
                              c=el_forces.dCp,
                              cmap='viridis')
            fig_avl_res.colorbar(surf)

            plt.figure("Cpx at station")
            y_s = el_forces.y_span_stations
            for i in range(0, 9):
                cl_i = np.trapz(el_forces.span_cp_x[y_s[i]][:, 1],
                                x=el_forces.span_cp_x[y_s[i]][:, 0] / (np.max(
                                    el_forces.span_cp_x[y_s[i]][:, 0]) - np.min(
                                    el_forces.span_cp_x[y_s[i]][:, 0])))
                plt.plot(el_forces.span_cp_x[y_s[i]][:, 0] / np.max(
                    el_forces.span_cp_x[y_s[0]][:, 0]),
                         el_forces.span_cp_x[y_s[i]][:, 1],
                         label=f"Cp at {y_s[i]}, CL = {cl_i}")

        if save_skeleton:
            args_to_save = {'wing span': self.span,
                            'wing root chord': self.root_chord,
                            'wing taper': self.taper_ratio,
                            'wing sweep': self.wing_le_sweep,
                            'wing loc ratio': self.wing_x_loc /
                                              (self.fuse_length - self.root_chord),
                            'wing-fuse area ratio': self.r_a,
                            'blending parameter': self.g_phi}
            save_path = str(os.path.join(WINGFUSE_DIR, self.name, 'args.json'))
            with open(save_path, 'w') as f:
                json.dump(args_to_save, f)

    def vsp_gen(self, wing_only=False, symm_only=True, custom_af=True, af_name="sc20010.af"):
        airfoils_dir = os.path.join(FILES_DIR, "sc_airfoils")
        af_sc20010 = os.path.join(airfoils_dir, af_name)
        num_secs = len(self.y) - 2
        vsp.VSPRenew()
        wing_id = vsp.AddGeom("WING")
        i=1
        while i <= num_secs:
            if custom_af:
                vsp.InsertXSec(wing_id, i, vsp.XS_FILE_AIRFOIL)
            else:
                vsp.InsertXSec(wing_id, i, vsp.XS_FOUR_SERIES)
            vsp.Update()
            i = i + 1

        wing_xs_surf = vsp.GetXSecSurf(wing_id, 0)
        if custom_af:
            vsp.ChangeXSecShape(wing_xs_surf, 0, vsp.XS_FILE_AIRFOIL)
            vsp.ChangeXSecShape(wing_xs_surf, 1, vsp.XS_FILE_AIRFOIL)
        vsp.Update()

        sec_spans = steps(self.y)
        sec_root_chord = self.chord[:-1]
        sec_tip_chord = self.chord[1:]
        sec_sweep = rad2deg(arctan(steps(self.x_le)/steps(self.y)))
        cl_cspline = self.results_2d_span["cl_interp_func"]
        af_inc = cl_cspline(self.y) / (2 * pi) * (180 / pi)
        yrot_0 = af_inc[0]
        sec_twist = steps(af_inc)

        #Stuff for y=0
        xsec_0 = vsp.GetXSec(wing_xs_surf, 0)
        if custom_af:
            vsp.ReadFileAirfoil(xsec_0, af_sc20010)
        vsp.SetParmVal(wing_id, "Y_Rel_Rotation", "XForm", yrot_0)
        # vsp.Update()
        for i in range(0, num_secs + 1):
            print(f"Generating section {i+1} for {self.name}")
            current_xsec = f"XSec_{i+1}"
            vsp.SetParmVal(wing_id, "Root_Chord", current_xsec, sec_root_chord[i])
            vsp.SetParmVal(wing_id, "Span", current_xsec, sec_spans[i])
            vsp.SetParmVal(wing_id, "Sweep", current_xsec, sec_sweep[i])
            vsp.SetParmVal(wing_id, "SectTess_U", current_xsec, 2)
            vsp.SetParmVal(wing_id, "Twist_Location", current_xsec, 0.5)
            vsp.SetParmVal(wing_id, "Twist", current_xsec, af_inc[i+1] - af_inc[0])
            vsp.SetParmVal(wing_id, "Tip_Chord", current_xsec, sec_tip_chord[i])
            xsec_i = vsp.GetXSec(wing_xs_surf, i + 1)
            if custom_af:
                vsp.ReadFileAirfoil(xsec_i, af_sc20010)
            vsp.Update()

        vsp_name = f"{self.name}.vsp3"
        vsp.WriteVSPFile(str(os.path.join(VSP_DATA_DIR, vsp_name)))

    def panair_gen(self, af_name="sc20010patch.af", plot=False,
                   panin_to_be_run=False, wing_only=False,
                   alpha=0):
        # airfoils_dir = os.path.join(FILES_DIR, "sc_airfoils")
        afname, extension = af_name.split(".")

        wing_fuse_wgs = wgs_creator.LaWGS(f"wgs_{self.name}")
        save_path = str(os.path.join(WINGFUSE_DIR, self.name))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        cl_cspline = self.results_2d_span["cl_interp_func"]
        if self.cl_alpha_spline is not None and not wing_only:

            af_inc = cl_cspline(self.y) / self.cl_alpha_spline(self.y)
        else:
            af_inc = cl_cspline(self.y) / (2 * pi)
        # af_inc = cl_cspline(self.y) / (2 * pi) if self.cl_alpha_spline is None
        # else cl_cspline(self.y) / self.cl_alpha_spline(self.y)
        savenames = []
        # Write out all airfoils
        for i, y_loc in enumerate(self.y):
            if not wing_only:
                xle, c, ainc = self.x_le[i], self.chord[i], af_inc[i]
            else:

                xle, c, ainc = self.x_le_w[i], self.chord_w[i], af_inc[i]
            current_af = place_airfoil(afname, extension, y_loc, xle, c, ainc,
                                       os.path.join(save_path, f"sec_{i}.csv"))
            savenames.append(current_af)

        # Make networks
        wing_sections = []
        wing_networks = []
        for i, y_loc in enumerate(self.y):
            wgs_wingsection = wgs_creator.read_airfoil(savenames[i],
                                                       y_coordinate=y_loc)
            wing_sections.append(wgs_wingsection)

            if i > 0:
                wing_net = wing_sections[i-1].linspace(wing_sections[i],
                                                       num=2)
                wing_networks.append(wing_net)

        root_net = wing_networks.pop(0)
        wing_networks = tuple(wing_networks)
        wing_fuse_panair = root_net.concat_row(wing_networks)

        tip_sec = wing_sections[-1]
        wingtip_us, wingtip_ls = tip_sec.split_half()
        wingtip_ls = wingtip_ls.flip()
        wingtip = wingtip_us.linspace(wingtip_ls, num=4)

        wake_length = 50 * self.avg_chord
        wingwake = wing_fuse_panair.make_wake(edge_number=3,
                                              wake_length=wake_length)
        wing_fuse_wgs.append_network("wingfuse", wing_fuse_panair, 1)
        wing_fuse_wgs.append_network("wingtip", wingtip, 1)
        wing_fuse_wgs.append_network("wake", wingwake, 18)
        wing_fuse_wgs.create_stl(filename=os.path.join(save_path,
                                                       f"{self.name}.stl"))
        wing_fuse_wgs.create_wgs(filename=os.path.join(save_path,
                                                       f"wgs_{self.name}.wgs"))
        wing_fuse_wgs.create_aux(alpha=alpha, mach=0.05, cbar=self.avg_chord,
                                 sref=self.wing_ref_area, xref=self.cg[0],
                                 zref=self.cg[2], span=self.span,
                                 filename=os.path.join(save_path,
                                                       f"{self.name}.aux"))

        if plot:
            wing_fuse_panair.plot_wireframe(show_normvec=False)

        if panin_to_be_run:
            base_dir = os.getcwd()
            os.chdir(save_path)
            run_panin(save_path, self.name)
            os.chdir(base_dir)

    def panair_spanload_eval(self):
        return

    def llt_based_design(self):
        # y_doub = np.concatenate((np.flip(self.y * -1)[:-1], self.y))
        # le_doub = np.concatenate((np.flip(self.x_le)[:-1], self.x_le))
        le_spline = CubicSpline(self.y, self.x_le)
        le_slope = le_spline.derivative()
        dx_dy = np.abs(le_slope(self.y))
        db_dx = 2 / dx_dy
        # plt.figure("dbdx")
        # plt.plot(self.y, db_dx)
        # plt.plot(self.y, np.sqrt(db_dx) + dx_dy / 4)
        cl_alpha_wf = np.clip(PI * (db_dx),
                              0, 2 * PI)
        cla_wf_avg = np.average(cl_alpha_wf[np.where(cl_alpha_wf < np.max(cl_alpha_wf))])
        cl_alpha_wf = np.clip(cl_alpha_wf, cla_wf_avg, 2 * PI)
        cl_alpha_w = PI * (0.5 * (self.chord_w / self.chord) + 1.5) \
                     * cos(self.wing_c2_sweep)
        cl_alpha_w2 = 0.5 * cl_alpha_wf + 0.5 * cl_alpha_w
        # cl_alpha_w3 = 2 * PI * (self.chord_w / self.chord) * cos(self.wing_c2_sweep)
        self.cl_alpha_spline = CubicSpline(self.y, cl_alpha_w2)

    def llt_spanload_eval(self, cl_req):
        if self.results_2d_span is None:
            self.spanload_eval(cl_req)

        if self.cl_alpha_spline is None:
            self.llt_based_design()

        cl_cspline = self.results_2d_span["cl_interp_func"]
        n_theta = 81
        theta = np.linspace(0, PI, n_theta)
        b_w = self.span
        y_llt = b_w / 2 * cos(theta)




def wing_fuse_plot_cp(name, yspan):
    wf_directory = os.path.join(WINGFUSE_DIR, name)
    # for file in os.listdir(wf_directory):
    #     if "element_forces" in file:
    #         ef_file = file
    wf_el_forces = ElementForcesOutput(os.path.join(wf_directory,
                                                    "element_forces_0.0_0.0.txt"))
    fig_cp = plt.figure(f"CP Surface Plot for {name}")
    ax_cp = plt.axes(projection='3d')
    surf_cp = ax_cp.scatter(wf_el_forces.x, wf_el_forces.y, wf_el_forces.dCp,
                         c=wf_el_forces.dCp, cmap="viridis")

    ax_cp.set_title(f"CP Surface Plot for {name}")
    cbar = fig_cp.colorbar(surf_cp)
    cbar.ax.set_ylabel(r"$C_p$")

    fig_cpx, (ax_cpx, ax_plan) = plt.subplots(2, 1, sharex=True,
                                              layout="constrained")
    y_s = wf_el_forces.y_span_stations

    for i, y in enumerate(np.take(y_s, (0, int((len(y_s)-1) / 2), -1))):
            cl_i = np.trapz(wf_el_forces.span_cp_x[y][:, 1],
                            x=wf_el_forces.span_cp_x[y][:, 0] /
                              (np.max(wf_el_forces.span_cp_x[y][:, 0]) -
                               np.min(wf_el_forces.span_cp_x[y][:, 0])))
            ax_cpx.plot(wf_el_forces.span_cp_x[y][:, 0],
                        wf_el_forces.span_cp_x[y][:, 1],
                        label=f"Cp at {np.round(y, 3)}, CL = {np.round(cl_i, 4)}")

    ax_cpx.set_title(r"$C_p$ v/s x at spanwise stations for %s" % name)
    ax_cpx.set_ylabel(r"$C_p$")
    ax_cpx.set_ylim((0, np.max(wf_el_forces.span_cp_x[y_s[0]][:, 1])))
    ax_cpx.legend()

    wf_temp = wing_fuse_from_save_json(name.replace("_reload", ""))
    wf_temp.le_te_chord_def(yspan)
    ax_plan.plot(wf_temp.x_le, wf_temp.y, color="blue")
    ax_plan.plot(wf_temp.x_te, wf_temp.y, color="blue")
    ax_plan.plot(wf_temp.x_le_w, wf_temp.y, "--", color="red")
    ax_plan.plot(wf_temp.x_te_w, wf_temp.y, "--", color="red")
    ax_plan.plot([wf_temp.x_le[-1] * 1, wf_temp.x_te[-1] * 1],
                 [wf_temp.y[-1], wf_temp.y[-1]], color='blue')
    ax_plan.plot([0, wf_temp.fuse_length],
                 [wf_temp.body_semiwidth, wf_temp.body_semiwidth],
                 color="green")
    ax_plan.set_xlabel(r"x")
    ax_plan.set_ylabel(r"Spanwise y")
    ax_plan.set_aspect("equal")
    return wf_el_forces.span_cp_x


def wing_fuse_from_save_json(name, is_inverse=True, custom_name=None):
    wf_directory = os.path.join(WINGFUSE_DIR, name)
    with open(os.path.join(wf_directory, "args.json"), "r") as f:
        wf_args = json.load(f)
    return_name = f"{name}_reload" if custom_name is None else custom_name
    wing_fuse = WingFuse(args=wf_args, name=return_name,
                         is_inverse=is_inverse)
    return wing_fuse


if __name__ == "__main__":
    # args_wf1 = {'fuselage length': 35.0,
    #             'wing span': 35.0,
    #             'wing root chord': 6.0,
    #             'wing taper': 0.2,
    #             'wing sweep': np.deg2rad(25),
    #             'wing loc ratio': 0.45,
    #             'fuselage front fraction': 0.2,
    #             'fuselage rear fraction': None}
    # wf_test1 = WingFuse(args=args_wf1, is_inverse=False,
    #                     name="TestWF_737")
    # y1 = np.linspace(0, 0.5 * wf_test1.span, 41)
    # wf_test1.le_te_chord_def(y1, plot=True)
    # wf_test1.spanload_eval(0.5, plot=True)
    #
    # args_wf2 = {'wing span': 35.0,
    #             'wing root chord': 6.0,
    #             'wing taper': 0.2,
    #             'wing sweep': np.deg2rad(25),
    #             'wing loc ratio': 0.45,
    #             'wing-fuse area ratio': wf_test1.r_a,
    #             'blending parameter': wf_test1.g_phi}
    #
    # wf_test2 = WingFuse(args=args_wf2, is_inverse=True,
    #                     name="TestWF_737_inv")
    # wf_test2.le_te_chord_def(y1, plot=True)
    #
    args_wf3 = {'fuselage length': 64.919 * 0.7,
                'wing span': 64.919,
                'wing root chord': 7.0 * 64.919 / 40,
                'wing taper': 0.16,
                'wing sweep': np.deg2rad(25),
                'wing loc ratio': 0.6,
                'fuselage front fraction': 0.4,
                'fuselage rear fraction': None}
    wf_test3 = WingFuse(args=args_wf3, name="TestWF_N2A")
    y3 = np.linspace(0, 0.5 * wf_test3.span, 41)
    wf_test3.le_te_chord_def(y3, plot=True)
    # wf_test3.spanload_eval(0.5, plot=True)
    #
    # args_wf4 = {'wing span': wf_test3.span,
    #             'wing root chord': wf_test3.root_chord,
    #             'wing taper': wf_test3.taper_ratio,
    #             'wing sweep': wf_test3.wing_le_sweep,
    #             'wing loc ratio': 0.6,
    #             'wing-fuse area ratio': wf_test3.r_a,
    #             'blending parameter': 1}
    # wf_test4 = WingFuse(args=args_wf4, is_inverse=True,
    #                     name="TestWF_N2AMax")
    # wf_test4.le_te_chord_def(y3, plot=True)
    # wf_test4.spanload_eval(0.5, plot=True)
    # a, b = 3, 7
    # print(a, b)
    # wf_reload = wing_fuse_from_save_json(f"Element_{a}_{b}", custom_name=f"{a}_{b}_panelgen")
    # y = np.linspace(0, 0.5 * wf_reload.span, 33)
    # wf_reload.le_te_chord_def(y)
    # wf_reload.spanload_eval(0.5)
    #
    # # wf_reload.avl_spanload_eval(0.5, body_only=False, plot=True)
    # # wf_reload.avl_spanload_eval(0.5, body_only=True, plot=True)
    # # cp_body = wing_fuse_plot_cp(f"Element_{a}_{b}_reload",
    # #                             np.linspace(0, 0.5 * wf_reload.span, 41))
    # wf_reload.vsp_gen(custom_af=True, af_name="sc20010patch.af")

    # wf_sweep_args = wf_reload.args
    # wf_sweep_args["wing sweep"] = np.deg2rad(25.5)
    # wf_sweep_args["cg"] = (20, 0, 0)
    # wf_sweep = WingFuse(args=wf_sweep_args, name=f"{a}_{b}_panelgen_sweep",
    #                     is_inverse=True)
    # wf_sweep.le_te_chord_def(y, plot=True)
    # wf_sweep.spanload_eval(0.5)
    # wf_sweep.vsp_gen(custom_af=True, af_name="sc20010.af")
    # wf_sweep.panair_gen(plot=True, panin_to_be_run=True)
    # wf_sweep.json_dump()

    # wf_sweep2_args = wf_reload.args
    # wf_sweep2_args["wing sweep"] = np.deg2rad(35)
    # wf_sweep2_args["wing loc ratio"] = 0.5
    # wf_sweep2 = WingFuse(args=wf_sweep2_args, name=f"{a}_{b}_panelgen_xtrasweep",
    #                     is_inverse=True)
    # wf_sweep2.le_te_chord_def(y)
    # wf_sweep2.spanload_eval(0.5)
    # wf_sweep2.vsp_gen(custom_af=True, af_name="sc20010patch.af")
    # cp_wingbody = wing_fuse_plot_cp(f"Element_{a}_{b}",
    #                                 np.linspace(0, 0.5 * wf_reload.span, 41))
    # body_key = sorted(cp_body.keys())[7]
    # wb_key = sorted(cp_wingbody.keys())[0]
    # cp_body_x, cp_body_y = cp_body[body_key][:, 0], cp_body[body_key][:, 1]
    # cp_wingbody_x, cp_wingbody_y = cp_wingbody[wb_key][:, 0], cp_wingbody[wb_key][:, 1]
    # cp_body_func = interp1d(cp_body_x, cp_body_y, fill_value='extrapolate')
    # cp_wingbody_interp = cp_body_func(cp_wingbody_x)
    #
    # plt.figure("Cp difference between body and body-wing")
    # plt.plot(cp_wingbody_x, cp_wingbody_y - cp_wingbody_interp)
    # plt.xlabel("x")
    # plt.ylabel(r"$\Delta C_p$")
    # plt.title(r"Wing-body $C_p$ - Body only $C_p$")
    # cl_diff = np.trapz(cp_wingbody_y - cp_wingbody_interp, x=cp_wingbody_x) / \
    #           np.max(cp_wingbody_x)


