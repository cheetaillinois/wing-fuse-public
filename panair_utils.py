import glob
import os
import numpy as np
from numpy import sin, cos
from pyPanair.preprocess import wgs_creator
from pyPanair.postprocess import calc_section_force
from gen_utils import read_xy_dat_files
import matplotlib as mpl, pandas as pd
from subprocess import Popen, PIPE
from wing_def import Wing
import matplotlib.pyplot as plt

mpl.use('Qt5Agg')
FILES_DIR = os.path.join(os.getcwd(), 'files')
WINGFUSE_DIR = os.path.join(FILES_DIR, "WingFuse")
VSP_DATA_DIR = os.path.join(os.getcwd(), 'vsp_data')
PI = np.pi


def twist_about(af_array, chord, twist_loc, twist_mag):

    twist_vector = np.array([twist_loc * chord, 0.0])
    twist_matrix = np.array([[cos(twist_mag), -1 * sin(twist_mag)],
                             [sin(twist_mag), cos(twist_mag)]])
    return (af_array - twist_vector) @ twist_matrix + twist_vector


def place_airfoil(af_coords_file, ext, y, xle, chord, twist, return_loc):

    # read airfoil
    af_dir = os.path.join(FILES_DIR, "sc_airfoils")
    airfoil_unscaled_xy = read_xy_dat_files(af_coords_file, af_dir, ext)

    # convert to upper and lower surface
    arg_le = np.argmin(airfoil_unscaled_xy[:, 0])
    upper_unscaled_xy = np.flip(airfoil_unscaled_xy[:arg_le+1, :], axis=0)
    lower_unscaled_xy = airfoil_unscaled_xy[arg_le:, :]

    # scale
    upper_scaled_xy = upper_unscaled_xy * chord
    lower_scaled_xy = lower_unscaled_xy * chord

    # twist
    upper_twisted_xy = twist_about(upper_scaled_xy, chord, 0.25, twist)
    lower_twisted_xy = twist_about(lower_scaled_xy, chord, 0.25, twist)

    # translate
    upper_placed_xy = upper_twisted_xy + np.array([xle, 0])
    lower_placed_xy = lower_twisted_xy + np.array([xle, 0])

    # make the pypanair csv
    concat = np.concatenate((upper_placed_xy, lower_placed_xy),
                                        axis=1)
    df_upper_lower = pd.DataFrame({"xup": concat[:, 0], "zup": concat[:, 1],
                                   "xlow": concat[:, 2], "zlow": concat[:, 3]})
    df_upper_lower.to_csv(return_loc)

    return return_loc


def place_airfoil_xy(airfoil_unscaled_xy, xle, chord, twist, return_loc,
                     coordinate_mode = True):

    # convert to upper and lower surface
    arg_le = np.argmin(airfoil_unscaled_xy[:, 0])
    upper_unscaled_xy = np.flip(airfoil_unscaled_xy[:arg_le+1, :], axis=0)
    lower_unscaled_xy = airfoil_unscaled_xy[arg_le:, :]

    # scale
    upper_scaled_xy = upper_unscaled_xy * chord
    lower_scaled_xy = lower_unscaled_xy * chord
    airfoil_scaled_xy = airfoil_unscaled_xy * chord

    # twist
    upper_twisted_xy = twist_about(upper_scaled_xy, chord, 0.25, twist)
    lower_twisted_xy = twist_about(lower_scaled_xy, chord, 0.25, twist)
    airfoil_twisted_xy = twist_about(airfoil_scaled_xy, chord, 0.25, twist)

    # translate
    upper_placed_xy = upper_twisted_xy + np.array([xle, 0])
    lower_placed_xy = lower_twisted_xy + np.array([xle, 0])
    airfoil_placed_xy = airfoil_twisted_xy + np.array([xle, 0])

    # make the pypanair csv

    if coordinate_mode:
        return airfoil_placed_xy
    else:
        concat = np.concatenate((upper_placed_xy, lower_placed_xy),
                                axis=1)
        df_upper_lower = pd.DataFrame({"xup": concat[:, 0], "zup": concat[:, 1],
                                       "xlow": concat[:, 2],
                                       "zlow": concat[:, 3]})
        df_upper_lower.to_csv(return_loc)
        return return_loc


def run_panin(directory, aux_name, print_output=False):
    panin_process = Popen("panin", stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=directory)
    (stdout, stderr) = panin_process.communicate(b"%s.aux" % aux_name.encode())
    if print_output:
        if stderr:
            print(stderr.decode('utf-8'))
        else:
            print(stdout.decode('utf-8'))


def run_panair(directory, print_output=False):
    if print_output:
        print(f"Starting panair run in directory: {directory}")
    panin_process = Popen("panair", stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=directory)
    if print_output:
        print("Feeding input file")
    (stdout, stderr) = panin_process.communicate(b"a502.in")
    if print_output:
        if stderr:
            print("ERROR")
            print(stderr.decode('utf-8'))
        else:
            print("OUTPUT")
            print(stdout.decode('utf-8'))
        print("done")


def process_ffmf(directory):
    """
    Processes a ffmf file. The ffmf file contains text output separated by lines
    , containing headers and values. The text output is horizontal and each
    value is in a separate column. This function parses by selecting the lines
    where the important data is stored, converts values to floats, and zips
    into a dictionary for further processing
    ONLY WORKS FOR SINGLE_SOLUTION FFMF FILES!!!
    :param directory: directory where the ffmf file is located
    :return: dictionary of solution quantities and values
    """
    with open(os.path.join(directory, "ffmf"), "r") as ffmf_file:
        ffmf_read = ffmf_file.readlines()

    keys1 = [val for val in ffmf_read[13].strip("\n").split(" ") if val]
    keys2 = [val for val in ffmf_read[14].strip("\n").split(" ") if val]
    vals1 = [float(val) for val in ffmf_read[17].strip("\n").split(" ") if val]
    vals2 = [float(val) for val in ffmf_read[18].strip("\n").split(" ") if val]
    keys1.extend(keys2)
    vals1.extend(vals2)
    solution_values = dict(zip(keys1, vals1))

    return solution_values


def process_aux(directory):
    """
    Processes an aux file, which is an input to panin. The aux file contains
    reference quantities fed to panair. Inputs are written vertically in the
    file, and each value is in a separate row. This function parses the aux
    files by selecting the lines where inputs are stored, converts values to
    floats, and zips into a dictionary for further processing
    :param directory: directory where the aux file is located
    :return: dictionary of reference quantities and values
    """
    aux_file_path = glob.glob(os.path.join(directory, "*.aux"))[0]
    with open(aux_file_path, "r") as aux_file:
        aux_read = aux_file.readlines()

    ref_quant_str_list = aux_read[2:-1]
    ref_quant_strs = [quant_str.strip("\n").split(" ")[0].lower()
                          for quant_str in ref_quant_str_list]
    ref_quant_vals = [float(quant_str.strip("\n").split(" ")[1])
                          for quant_str in ref_quant_str_list]
    return dict(zip(ref_quant_strs, ref_quant_vals))

def panair_postprocess(directory):

    ref_quantities = process_aux(directory)
    sol_quantities = process_ffmf(directory)
    all_vals = dict(ref_quantities)
    all_vals.update(sol_quantities)
    all_vals["ar"] = all_vals["span"] ** 2 / all_vals["sref"]
    all_vals["e"] = all_vals["cl"] ** 2 / (PI * all_vals["ar"] * all_vals["cdi"])
    return all_vals


if __name__ == "__main__":
    print("start")
    wgs_test = wgs_creator.LaWGS("Test_NASA_SC")
    panair_af_dir = os.path.join(FILES_DIR, "panair_stuff")
    files_to_clean = glob.glob(os.path.join(panair_af_dir, "*"))
    for f in files_to_clean:
        os.remove(f)
    wing_af1 = place_airfoil("sc20010patch", "af", 0, 0, 2.0, np.deg2rad(0.0),
                             os.path.join(panair_af_dir, "eta0000.csv"))
    wing_af2 = place_airfoil("sc20010patch", "af", 0, 0, 2.0, np.deg2rad(0.0),
                             os.path.join(panair_af_dir, "eta1000.csv"))

    wing_section1 = wgs_creator.read_airfoil(os.path.join(panair_af_dir, "eta0000.csv"), y_coordinate=0.0)
    wing_section2 = wgs_creator.read_airfoil(os.path.join(panair_af_dir, "eta1000.csv"), y_coordinate=10.0)
    wing_net1 = wing_section1.linspace(wing_section2, num=20)
    wingtip_upper, wingtip_lower = wing_section2.split_half()
    wingtip_lower = wingtip_lower.flip()
    wingtip = wingtip_upper.linspace(wingtip_lower, num=4)

    wing_net1.plot_wireframe(show_normvec=False)
    wgs_test.append_network("wing", wing_net1, 1)
    wgs_test.append_network("wingtip", wingtip, 1)
    wingwake = wing_net1.make_wake(edge_number=3, wake_length=50 * 2.0)
    wgs_test.append_network("wingwake", wingwake, 18)
    # wgs_test.create_stl()
    wgs_test.create_wgs(filename=os.path.join(panair_af_dir,
                                              f"{wgs_test.name}.wgs"))
    wgs_test.create_aux(alpha=2.0, mach=0.1, cbar=2.0, span=20, sref=40,
                        xref=0.0, zref=0.0, filename=os.path.join(panair_af_dir,
                                                       "Test.aux"))
    run_panin(panair_af_dir, "Test", print_output=True)
    run_panair(panair_af_dir, print_output=True)
    solution_vals = panair_postprocess(panair_af_dir)

    base_dir = os.getcwd()
    aoa, mac, rotcenter = solution_vals["alpha"], solution_vals["cbar"], (0.5, 0, 0)
    os.chdir(panair_af_dir)
    calc_section_force(aoa, mac, rotcenter)
    df_sec_force = pd.read_csv("section_force.csv")
    os.chdir(base_dir)

    plt.figure("Rect Wing Spanload")
    plt.plot(df_sec_force.pos,
             df_sec_force.cl * df_sec_force.chord / solution_vals["cbar"],
             "s")
    plt.xlabel("spanwise position")
    plt.ylabel("local lift coefficient")
    plt.grid()
    plt.show()


