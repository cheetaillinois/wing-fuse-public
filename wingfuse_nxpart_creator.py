import os
import json
import numpy as np
import subprocess
from panair_utils import place_airfoil_xy
from wing_fuse_def import WingFuse

FILES_DIR = os.path.join(os.getcwd(), 'files')
WINGFUSE_DIR = os.path.join(FILES_DIR, "WingFuse")
VSP_DATA_DIR = os.path.join(os.getcwd(), 'vsp_data')

def load_data(file):
    if os.path.splitext(file)[-1] in ['.json', '.jmea']:
        with open(file, 'r') as file:
            var = json.load(file)
        return var
    else:
        raise Exception('Invalid file extension for data load! Current available choices: .pkl, .dill, .json, .jmea')


def save_data(var, file):
    if os.path.splitext(file)[-1] in ['.json', '.jmea']:
        with open(file, 'w') as file:
            json.dump(var, file, indent=4)
    else:
        raise Exception('Invalid file extension for data save! Current available choices: .pkl, .dill, .json, .jmea')


def write_NX_macro(macro_file_name: str, control_point_json_file: str, nx_part_file_name: str,
                   wf: WingFuse, output_control_point_json: str = None,
                   wing_only=False, req_cl = 0.5, llt_designed=False):
    with open(macro_file_name, 'w') as f:
        for import_ in ['math', 'NXOpen', 'NXOpen.Features', 'NXOpen.GeometricUtilities', 'time', 'os']:
            f.write(f'import {import_}\n')

        with open('journal_functions.py', 'r') as g:
            f.writelines(g.readlines())

        ctrlpt_dict = load_data(control_point_json_file)
        newpt_dict = {}
        y_span = wf.y
        y_new = np.concatenate((np.flip(-1 * y_span)[:-1], y_span))
        wf.spanload_eval(req_cl, wing_only=wing_only)
        xle = wf.le_func(y_new) if not wing_only else wf.le_w_func(y_new)
        chord = wf.chord_func(y_new) if not wing_only else wf.chord_w_func(y_new)
        if llt_designed:
            wf.llt_based_design()
            ainc = wf.results_2d_span["cl_interp_func"](y_span) / wf.cl_alpha_spline(y_span)
        else:
            ainc = wf.results_2d_span["cl_interp_func"](y_span) / (2 * np.pi)
        ainc_new = np.concatenate((np.flip(ainc)[:-1], ainc))
        for i, y_loc in enumerate(y_new):
            xle_y = xle[i] * 1000
            chord_y = chord[i] * 1000
            twist = ainc_new[i]
            unscaled_xy = np.array(ctrlpt_dict["A0"])
            placed_xy = place_airfoil_xy(unscaled_xy, xle_y, chord_y, twist,
                                         None, coordinate_mode=True)

            new_curve = np.insert(placed_xy, 1, y_loc * 1000, axis=2)
            # print(new_curve)
            newpt_dict[y_loc] = new_curve.tolist()
        # for k, ctrlpts in ctrlpt_dict.items():
        #     new_ctrlpts = np.array(ctrlpts)
        #     for curve in new_ctrlpts:
        #         curve *= 1000
        #         # Do curve math here (these are N x 2 arrays where N is the number of Bezier control points in the curve)
        #         pass
        #     if k == 'A0':
        #         y = 0.0
        #     else:
        #         y = 3000.0
        #     new_ctrlpts = np.insert(new_ctrlpts, 1, y, axis=2)
        #     ctrlpt_dict[k] = new_ctrlpts.tolist()

        f.write('ctrlpts = {\n')
        for a_name, ctrlpts in newpt_dict.items():
            f.write(f'    "{a_name}": [\n')
            # print(a_name)
            # print(ctrlpts)
            for ctrlpt_set in ctrlpts:
                f.write(f'        [\n')
                for ctrlpt in ctrlpt_set:
                    f.write(f'            [{ctrlpt[0]}, {ctrlpt[1]}, {ctrlpt[2]}],\n')
                f.write(f'        ],\n')
            f.write(f'    ],\n')
        f.write('}\n\n')

        f.write(f'if os.path.exists(r\'{nx_part_file_name}\'):\n')
        f.write(f'    os.remove(r\'{nx_part_file_name}\')\n')

        f.write(f'the_session = create_new_file_nx(r\'{nx_part_file_name}\')\n')

        f.write('through_curve_args = create_bezier_curve_from_ctrlpts(ctrlpts, the_session)\n')

        f.write('workPart = through_curve_builder(*through_curve_args)\n')

        f.write('save_file(workPart)\n')

        if output_control_point_json is not None:
            save_data(ctrlpt_dict, output_control_point_json)


def main():
    output_macro_name = os.path.abspath('out_macro.py')
    input_ctrlpts = os.path.abspath(
        '../../../../Users/Karthik Mahesh/Box/CHEETA UIUC Only/MSES things/karthik_macro/control_points_sc20010.json')
    part_file = os.path.abspath('test_airfoil.prt')
    write_NX_macro(output_macro_name, input_ctrlpts, part_file, output_control_point_json=None)


if __name__ == '__main__':
    # wing_loc_ratio = 13.566 / (39.0 - 6.5755)
    # wf_args = {'fuselage length': 39.0,
    #             'wing span': 41.343,
    #             'wing root chord': 6.5755,
    #             'wing taper': 0.25749,
    #             'wing sweep': np.deg2rad(25.5),
    #             'wing loc ratio': wing_loc_ratio,
    #             'fuselage front fraction': 0.25,
    #             'fuselage rear fraction': None}

    wf_args = {"wing span": 41.343,
               "wing root chord": 6.5755,
               "wing taper": 0.25749,
               "wing sweep": 0.44505895925855404,
               "wing loc ratio": 0.41838733056793476,
               "wing-fuse area ratio": 0.980347338853566,
               "blending parameter": 0.25}
    wing_fuse = WingFuse(args=wf_args, is_inverse=True, name="WingFuse_NX")
    wing_only = False
    llt_design = True
    n_theta = 41
    CL_req = 0.5
    theta = np.linspace(0, np.pi, n_theta)
    b_w = wing_fuse.span
    y = np.linspace(0, b_w/2, 21)
    y_half = np.flip(y[np.where(y >= 0)])
    wing_fuse.le_te_chord_def(y)
    wing_fuse.spanload_eval(0.5, wing_only=wing_only)
    # wing_fuse.json_dump()
    output_macro_name = os.path.abspath(os.path.join(FILES_DIR, 'out_macro.py'))
    input_ctrlpts = os.path.abspath(
        '../../../../Users/Karthik Mahesh/Box/CHEETA UIUC Only/MSES things/karthik_macro/control_points_sc20010.json')
    part_file = os.path.abspath(os.path.join(FILES_DIR, f'wing_fuse_{wing_fuse.g_phi:.2f}.prt'))
    write_NX_macro(output_macro_name, input_ctrlpts, part_file, wing_fuse,
                   output_control_point_json=None, wing_only=wing_only,
                   llt_designed=llt_design)

    base_dir = os.getcwd()
    os.chdir(FILES_DIR)
    subprocess.run(["run_journal", "out_macro.py"])
    os.chdir(base_dir)
