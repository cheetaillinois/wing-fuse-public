import os, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt,\
    pandas as pd
from pyPanair.postprocess import write_tec, write_vtk, calc_section_force, read_agps
from wing_fuse_def import WingFuse, wing_fuse_from_save_json

mpl.use('Qt5Agg')
FILES_DIR = os.path.join(os.getcwd(), 'files')
WINGFUSE_DIR = os.path.join(FILES_DIR, "WingFuse")
VSP_DATA_DIR = os.path.join(os.getcwd(), 'vsp_data')


def wf_write_tec(wf_name, type="tec"):
    base_dir = os.getcwd()
    wf_dir = os.path.join(WINGFUSE_DIR, wf_name)
    os.chdir(wf_dir)
    if type == "tec":
        write_tec(n_wake=1)
    elif type == "vtk":
        write_vtk(n_wake=1)
    os.chdir(base_dir)


def wf_panair_spanload(wf_name, plot=False, add_directory=None,
                       return_center_cp=False, write_vis=None):
    base_dir = os.getcwd()
    wf_dir = os.path.join(WINGFUSE_DIR, wf_name) if add_directory is None\
        else os.path.join(WINGFUSE_DIR, wf_name, add_directory)
    os.chdir(wf_dir)
    wing_fuse_load = wing_fuse_from_save_json(wf_name)
    aoa, mac, rotcenter = 0., wing_fuse_load.avg_chord, wing_fuse_load.cg
    calc_section_force(aoa, mac, rotcenter)
    agps_raw_data = read_agps()
    df_sec_force = pd.read_csv("section_force.csv")
    if write_vis == "vtk":
        write_vtk(n_wake=1, outputname=f"panair_vtk_{wf_name}")
    os.chdir(base_dir)

    wing_network = agps_raw_data[0]
    n_y = np.shape(wing_network)[0]
    y = np.linspace(0, wing_fuse_load.span / 2, n_y)
    wing_fuse_load.le_te_chord_def(y)
    wing_fuse_load.spanload_eval(0.5)
    r2d_span = wing_fuse_load.results_2d_span
    center_section = wing_network[0]
    wing_section_mid = wing_network[17]
    if plot:
        plt.figure(f"WingFuse Spanload for {wf_name}")
        plt.plot(df_sec_force.pos, df_sec_force.cl * df_sec_force.chord / wing_fuse_load.avg_chord, label="PANAIR")
        plt.plot(r2d_span["y_vor"], r2d_span["c_n"] * r2d_span["chord_vor"] / wing_fuse_load.avg_chord, label="2D LLT")
        plt.xlabel("spanwise position")
        plt.ylabel("local lift coefficient")
        plt.ylim((0.0,  0.1 + np.max(r2d_span["c_n"] * r2d_span["chord_vor"] / wing_fuse_load.avg_chord)))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(wf_dir, f"panair_spanload_{wf_name}.png"))
        fig_tit = f"Planform and Cp for {wf_name}"
        fig_cpx, (ax_cpx, ax_plan) = plt.subplots(2, 1, sharex=True,
                                                  layout="tight", num=fig_tit)
        ax_cpx.plot(center_section[:, 1], center_section[:, 4] * -1)
        ax_cpx.plot(wing_section_mid[:, 1], wing_section_mid[:, 4] * -1)
        ax_cpx.set_ylabel(r"$C_p$")
        ax_cpx.set_ylim((-0.5, 0.5))
        ax_cpx.grid()
        ax_plan.plot(wing_fuse_load.x_le, wing_fuse_load.y, color="blue")
        ax_plan.plot(wing_fuse_load.x_te, wing_fuse_load.y, color="blue")
        ax_plan.plot(wing_fuse_load.x_le_w, wing_fuse_load.y, "--", color="red")
        ax_plan.plot(wing_fuse_load.x_te_w, wing_fuse_load.y, "--", color="red")
        ax_plan.plot([wing_fuse_load.x_le[-1] * 1, wing_fuse_load.x_te[-1] * 1],
                     [wing_fuse_load.y[-1], wing_fuse_load.y[-1]], color='blue')
        ax_plan.plot([0, wing_fuse_load.fuse_length],
                     [wing_fuse_load.body_semiwidth, wing_fuse_load.body_semiwidth],
                     color="green")
        ax_plan.set_xlabel(r"x")
        ax_plan.set_ylabel(r"Spanwise y")
        ax_plan.set_aspect("equal")



    if not return_center_cp:
        return df_sec_force
    else:
        return df_sec_force, center_section[:, 1], center_section[:, 4] * -1


if __name__ == "__main__":
    wf_name = "4_5_panelgen_sweep"
    # wf_write_tec(wf_name, type="vtk")
    sf = wf_panair_spanload(wf_name, plot=True, add_directory="panair_run_new")
