import numpy as np
import os
import glob
import shutil
from avlpy import AvlHeader, AvlSurface, AvlSection, AvlInput, runAvl


def cl_alpha_2pi(y):
    return 2 * np.pi


def avl_sim_spanload(name=None, mach=None, sref=None, cref=None, bref=None,
                     cg_wb=None, nspan=None, nchord=None, sections_new_dir=None,
                     base_dir=None, avl_binpath=None, le_x_cspline=None,
                     le_z_cspline=None, chord_cspline=None, cl_cspline=None,
                     h_aug_cspline=None, cla_func=None, alphas=[0.0]):

    # Start AVL file definition
    if not os.path.exists(sections_new_dir):
        os.mkdir(sections_new_dir)
    else:
        files_to_remove = glob.glob(f'{sections_new_dir}/*')
        print(files_to_remove)
        for f in files_to_remove:
            os.remove(f)
    header = AvlHeader(name, Mach=mach, Sref=sref,
                       Cref=cref,
                       Bref=bref, CGref=cg_wb)
    surfaces = []
    wb_surface = AvlSurface(name="WingBody", Nchord=nchord,
                            Component=1,
                            Nspan=None, Sspace=None, Ydupl=0.0)
    le_y_avl = np.linspace(0, bref/2, nspan)
    le_x_avl = le_x_cspline(le_y_avl)
    le_z_avl = le_z_cspline(le_y_avl)
    chord_avl = chord_cspline(le_y_avl)
    cl_avl = cl_cspline(le_y_avl)
    h_aug_avl = h_aug_cspline(le_y_avl) if h_aug_cspline is not None else np.zeros_like(le_y_avl)

    cl_alpha_func = cl_alpha_2pi if cla_func is None else cla_func
    for iy, yle in enumerate(le_y_avl):
        h_i = cl_avl[iy] / (cl_alpha_func(yle)) * (180 / np.pi) + h_aug_avl[iy]
        le_i = [le_x_avl[iy], yle, le_z_avl[iy]]
        chord_i = chord_avl[iy]
        section_i = AvlSection(le=le_i, chord=chord_i, ainc=h_i,
                               Nspan=1, Sspace=1.0)
        wb_surface.addSection(section_i)

    surfaces.append(wb_surface)
    avlinput = AvlInput(header, surfaces)

    avlinput.toFile(str(os.path.join(sections_new_dir,
                                     f"{name}_wingbody.avl")))
    os.chdir(sections_new_dir)
    results_avl, stdout = runAvl(avlinput, alpha=alphas,
                                 beta=[0.0], mach=0.0,
                                 savePlots=True,
                                 binPath=avl_binpath, change_dir=False,
                                 cleanup_flag=False,
                                 collect_element_forces=True)
    df_avl_results = results_avl.get_results_df()
    cl_avl_res = df_avl_results.CLtot.values[0]
    cdi_avl_res = df_avl_results.CDind.values[0]
    e_avl_res = df_avl_results.e.values[0]
    os.chdir(base_dir)
    results_we_need = {"cl_avl": cl_avl_res, "cdi_avl": cdi_avl_res,
                       "e_avl": e_avl_res,
                       "df_everything": df_avl_results,
                       "unp": results_avl,
                       "y_avl": le_y_avl
                       }
    return results_we_need
