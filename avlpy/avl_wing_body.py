from .avlInput import *


def process_wingbody(body_obj, wing_obj):
    # degen_obj = degen_objects[0]
    # component = "nothing"

    stick = wing_obj.sticks[0]
    le = np.array(stick.le)
    te = np.array(stick.te)
    chord = np.array(stick.chord)
    u = np.array(stick.u)

    wle_func = interp1d(le[:, 1], le[:, 0])
    wte_func = interp1d(te[:, 1], te[:, 0])
    wc_func = interp1d(le[:, 1], chord)

    sticks = body_obj.sticks
    h_stick = np.array(sticks[0].te)

    # Horizontal Surface

    le_x = h_stick[:, 0]
    le_y = h_stick[:, 1]
    le_z = h_stick[:, 2]
    tail_x = le_x[-1]

    le_f = interp1d(le_x, le_y, fill_value="extrapolate")
    wf_max = np.max(abs(h_stick[:, 1]))

    yle_h = np.linspace(0, wf_max, 6)
    print("Is 0 in yle", 0 in yle_h)
    xle_h = np.array([0.])
    chord_h = np.array([tail_x])

    for y in yle_h[1:]:
        xle = fsolve(lambda xx: le_f(xx) - np.abs(y), np.array([0]))
        xte = fsolve(lambda xx: le_f(xx) - np.abs(y), tail_x)
        chord = xte - xle
        xle_h = np.append(xle_h, xle)
        chord_h = np.append(chord_h, chord)

    zle_h = np.ones_like(xle_h) * max(le_z)
    le_h = np.array(list(zip(xle_h, yle_h, zle_h)))
    te_h = np.array(list(zip(xle_h + chord_h, yle_h, zle_h)))

    print(le_h, te_h)

    wingstart = wf_max + 0.1
    le_w1 = np.array([[wle_func(wingstart), wingstart, max(le_z)]])
    le_w2 = le[np.where(le[:, 1] > wf_max)]
    le_h = np.concatenate((le_h, le_w1, le_w2), axis=0)

    te_w1 = np.array([[wte_func(wingstart), wingstart, max(le_z)]])
    te_w2 = te[np.where(te[:, 1] > wf_max)]
    te_h = np.concatenate((te_h, te_w1, te_w2), axis=0)

    # Vertical Surface

    vl_stick = np.array(sticks[1].te)
    vu_stick = np.array(sticks[1].le)

    vl_x = vl_stick[:, 0]
    vl_z = vl_stick[:, 2]
    vl_f = interp1d(vl_x, vl_z, fill_value="extrapolate")

    vu_x = vu_stick[:, 0]
    vu_z = vu_stick[:, 2]
    vu_f = interp1d(vu_x, vu_z, fill_value="extrapolate")

    # plt.plot(vl_x, vl_z, vu_x, vu_z)
    le_int = vl_stick[0, 2]
    te_int = vl_stick[-1, 2]
    z_nt = np.array([le_int, te_int])

    # le lower than te => vu le, vl te
    # te lower than le => vl le, vu te

    z_min = np.min(vl_z)
    z_max = np.max(vu_z)
    zle_v = np.linspace(z_min, z_max, 11)
    # Insert exact locations of nose and tail for accuracy
    ii = np.searchsorted(zle_v, z_nt)
    zle_v = np.insert(zle_v, ii, z_nt)
    yle_v = np.zeros_like(zle_v)
    xle_v = np.array([])
    chord_v = np.array([])

    for z in zle_v:
        if z < le_int:
            xle = fsolve(lambda xx: vl_f(xx) - z, np.array([0]))
        else:
            xle = fsolve(lambda xx: vu_f(xx) - z, np.array([0]))

        if z < te_int:
            xte = fsolve(lambda xx: vl_f(xx) - z, tail_x)
        else:
            xte = fsolve(lambda xx: vu_f(xx) - z, tail_x)

        chord = xte - xle
        xle_v = np.append(xle_v, xle)
        chord_v = np.append(chord_v, chord)

    le_v = np.array(list(zip(xle_v, yle_v, zle_v)))
    te_v = np.array(list(zip(xle_v + chord_v, yle_v, zle_v)))

    surf_h = AvlSurface(name=body_obj.name, Nchord=len(le_h),
                        Component=1, Ydupl=0.0,
                        Nspan=None, Sspace=None)
    span_axis = le_h[-1, :] - le_h[0, :]
    nspan = 1
    for i in range(len(le_h)):
        nspan += 1
        # if abs(round(u[i]) - u[i]) > 1.0e-10:
        #     continue

        # Compute ainc
        x_vec = np.array([1.0, 0.0, 0.0])
        chord_vec = te_h[i, :] - le_h[i, :]
        cos_theta = np.dot(chord_vec, x_vec) / np.linalg.norm(chord_vec)
        rot_axis = np.cross(x_vec, chord_vec)
        angle_sign = 1.0

        cos_theta_span_axis = np.dot(span_axis, rot_axis) / (
                np.linalg.norm(span_axis) * np.linalg.norm(rot_axis))
        if np.abs(cos_theta_span_axis) < np.pi / 2.0:
            angle_sign = -1.0

        ainc = np.rad2deg(np.arccos(np.clip(cos_theta, -1, 1))) * angle_sign

        sect = AvlSection(le=le_h[i, :], chord=chord_h[i], ainc=ainc,
                          Nspan=nspan,
                          Sspace=1.0)
        surf_h.addSection(sect)
        if len(surf_h.sections) > 1:
            surf_h.sections[-2].Nspan = nspan
        nspan = 2

    surf_v = AvlSurface(name=body_obj.name, Nchord=len(le_v),
                        Component=2,
                        Nspan=None, Sspace=None)

    span_axis = le_v[-1, :] - le_v[0, :]
    nspan = 1
    for i in range(len(le_v)):
        nspan += 1
        # if abs(round(u[i]) - u[i]) > 1.0e-10:
        #     continue

        # Compute ainc
        x_vec = np.array([1.0, 0.0, 0.0])
        chord_vec = te_v[i, :] - le_v[i, :]
        cos_theta = np.dot(chord_vec, x_vec) / np.linalg.norm(chord_vec)
        rot_axis = np.cross(x_vec, chord_vec)
        angle_sign = 1.0

        cos_theta_span_axis = np.dot(span_axis, rot_axis) / (
                np.linalg.norm(span_axis) * np.linalg.norm(rot_axis))
        if np.abs(cos_theta_span_axis) < np.pi / 2.0:
            angle_sign = -1.0

        ainc = np.rad2deg(np.arccos(np.clip(cos_theta, -1, 1))) * angle_sign
        sect = AvlSection(le=le_v[i, :], chord=chord_v[i], ainc=ainc,
                          Nspan=nspan,
                          Sspace=1.0)
        surf_v.addSection(sect)
        if len(surf_v.sections) > 1:
            surf_v.sections[-2].Nspan = nspan
        nspan = 2

    return surf_h, surf_v


def create_input_from_dgwingbody(degen_objects=None, degen_set=None, title="DegenAvl", mach=0.0, Sref=1.0, Bref=1.0,
                                 Cref=1.0, cgRef=(0.0, 0.0, 0.0), cdp=0.0):
    """
    Creates an AvlInput object from a list of degen geometry objects

    The degen objects can be created by passing the vsp set from which to create them

    :param degen_objects: List of degen geom objects, these can be created from openvsp. If degen_objects is None,
    degen objects will be created by using OpenVSP on the input degen set
    :param degen_set: OpenVSP set to create degen objects from if :param degen_objects is None
    :param title: title of the avl geometry
    :param mach: mach number
    :param Sref: reference area
    :param Bref: span
    :param Cref: reference chord
    :param cgRef: moment calculation location
    :param cdp: fixed parasite drag value to add to computed induced drag
    :return: AvlInput object
    """
    import degen_geom as dg
    import numpy as np
    header = AvlHeader(title, Mach=mach, Sref=Sref, Cref=Cref, Bref=Bref,
                       CGref=cgRef, CDp=cdp)
    surfaces = []
    components = {}

    # Create degen objects from OpenVSP if no degen objects were passed in. Throw an exception if both degen_objects
    # and degen_set are None
    if degen_objects is None:
        raise ValueError(
            "degen_objects and degen_set cannot both be set to None")

    body_obj = degen_objects[0]
    wing_obj = degen_objects[1]
    surf, surf_v = process_wingbody(body_obj, wing_obj)
    surfaces.append(surf_v)
    surfaces.append(surf)

    return AvlInput(header, surfaces)

