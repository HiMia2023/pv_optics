"""Calculate optical absorption and carrier generatation rate in a PV device stack."""

import time
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.constants import Planck, speed_of_light, elementary_charge
import tmm


APPLICATION_FOLDER = pathlib.Path.cwd()
REFRACTIVE_INDEX_FOLDER = APPLICATION_FOLDER.joinpath("refractive_index")
ILLUMINATION_FOLDER = APPLICATION_FOLDER.joinpath("illumination")
INPUT_FOLDER = APPLICATION_FOLDER.joinpath("input")
OUTPUT_FOLDER = APPLICATION_FOLDER.joinpath("output")
FILE_EXT = "tsv"


def wavelength_to_energy(wavelength):
    """Convert wavelength/s in nm to energy in eV.

    Parameters
    ----------
    wavelength : float or array
        Wavelength/s in nm.

    Returns
    -------
    E : float or array
        Energy/s in eV.
    """
    return (Planck / elementary_charge) * speed_of_light / (wavelength * 1e-9)


def load_nk_data(filepath):
    """Load and interpolate refractive index data from a file.

    Files must be formatted as follows:
        - the first row must be reference information for the data, e.g. DOI, URL etc.
        - the second row must be a blank line
        - the third row must be three tab-separated column headers
        - all subsequent rows contain three columns of tab-separated data with column 0
        containing wavelengths in nm, column 1 containing n, and column 2 containing k.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to data file.

    Returns
    -------
    n_int : scipy.interpolate.interp1d
        scipy interpolation object for n.
    k_int : scipy.interpolate.interp1d
        scipy interpolation object for k.
    """
    # load data
    data = np.genfromtxt(filepath, skip_header=3, delimiter="\t")

    # create interpolation objects for n and k
    n_int = scipy.interpolate.interp1d(data[:, 0], data[:, 1], kind="cubic")
    k_int = scipy.interpolate.interp1d(data[:, 0], data[:, 2], kind="cubic")

    return n_int, k_int


def load_illumination_data(filepath):
    """Load and interpolate illumination data from a file.

    Files must be formatted as follows:
        - the first row must be reference information for the data, e.g. DOI, URL etc.
        - the second row must be a blank line
        - the third row must be two tab-separated column headers
        - all subsequent rows contain two columns of tab-separated data with column 0
        containing wavelengths in nm, and column 1 containing spectral irradiances in
        W/m^2/nm.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to data file.

    Returns
    -------
    F_int : scipy.interpolate.interp1d
        Scipy interpolation object for spectral irradiance.
    """
    # load data
    data = np.genfromtxt(filepath, skip_header=1, delimiter="\t")

    return scipy.interpolate.interp1d(data[:, 0], data[:, 1], kind="cubic")


def get_interpolated_n_list(layers):
    """Generate list of interpolation objects for each layer.

    Parameters
    ----------
    layers : list of str
        List of layer names corresponding to file names.

    Returns
    -------
    int_n_list : list of scipy.interpolate.interp1d
        List of interpolation objects for each layer.
    """
    # get interpolation objects for refractive indices for each layer
    int_n_list = []
    for layer in layers:
        path = REFRACTIVE_INDEX_FOLDER.joinpath(f"{layer}.{FILE_EXT}")
        int_n_list.append(load_nk_data(path))

    return int_n_list


def calculate_tmm_spectra(int_n_list, d_list, c_list, wavelengths, th_0, polarisation):
    """Perform tmm calculation for each wavelength.

    Parameters
    ----------
    int_n_list : list of scipy.interpolate.interp1d
        List of interpolation objects for each layer.
    d_list : list of float or int
        List of thicknesses in nm for each layer in layers.
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    wavelengths : np.array
        Array of wavelengths in nm at which to perform tmm calculations.
    th_0 : numeric
        Incident angle in degrees, 0 deg = normal incidence.
    polarisation : str
        Polarisation state: "s" or "p".

    Returns
    -------
    inc_tmm_pol : dict of dict
        Dictionary of output dictionaries from the tmm.inc_tmm function for a
        particular polarisation state. Keys are each wavelength in `wavelengths`.
    """
    # perform tmm calculation for each wavelength
    inc_tmm_pol = {}
    for wavelength in wavelengths:
        # build list of complex refractive indices at current wavelength
        n_list = [
            n_int(wavelength) + 1j * k_int(wavelength) for n_int, k_int in int_n_list
        ]

        # perform tmm calculation
        inc_tmm_pol[wavelength] = tmm.inc_tmm(
            polarisation, n_list, d_list, c_list, th_0, wavelength
        )

    return inc_tmm_pol


def calculate_rta_spectra(inc_tmm_pol):
    """Caculate absorptance spectra for all layers.

    Parameters
    ----------
    inc_tmm_pol : dict of dict
        Dictionary of output dictionaries from the tmm.inc_tmm function for a
        particular polarisation state. Keys are each wavelength in `wavelengths`.

    Returns
    -------
    rta_pol : np.array
        Array of data containing absorption within each layer ("absorption" in
        layer 0 and the last layer being the stack reflectance and transmittance
        respectively) for a particular polarisation state.
    """
    rta_pol = [
        [wavelength] + tmm.inc_absorp_in_each_layer(inc_data)
        for wavelength, inc_data in inc_tmm_pol.items()
    ]

    return np.array(rta_pol)


def calculate_total_rta_spectra(rta_s, rta_p, s_fraction=0.5, p_fraction=0.5):
    """Calcualte total reflection, transmission, and absorption spectra.

    This function accounts for the fraction of s- and p- polarised light in the
    illumination source.

    Parameters
    ----------
    rta_s : array
        Reflection, transmission, and absorption spectra from tmm calculations for
        s-polarised light.
    rta_p : array
        Reflection, transmission, and absorption spectra from tmm calculations for
        p-polarised light.
    s_fraction : float
        Fraction of s-polarised light incident on the stack. The fractions of s- and p-
        polarised light must sum to 1.
    p_fraction : float
        Fraction of s-polarised light incident on the stack. The fractions of s- and p-
        polarised light must sum to 1.

    Returns
    -------
    rta : array
        Reflection, transmission, and absorption spectra from tmm calculations
        accounting for s- and p- polarisation fractions in the illumination source.
    """
    # initialise rta array
    rta = np.ones(np.shape(rta_s))

    # set column 0 to wavelengths
    rta[:, 0] = rta_s[:, 0]

    # set remaining columns as weighted sum of s- and p- polarisations
    rta[:, 1:] = s_fraction * rta_s[:, 1:] + p_fraction * rta_p[:, 1:]

    return rta


def calculate_total_generation_profile(
    gen_x_s, gen_x_p, s_fraction=0.5, p_fraction=0.5
):
    """Calcualte total carrier generation rate profiles.

    This function accounts for the fraction of s- and p- polarised light in the
    illumination source.

    Parameters
    ----------
    gen_x_s : dict
        Carrier generation rate profile for s-polarised light.
    gen_x_p : dict
        Carrier generation rate profile for p-polarised light.
    s_fraction : float
        Fraction of s-polarised light incident on the stack. The fractions of s- and p-
        polarised light must sum to 1.
    p_fraction : float
        Fraction of s-polarised light incident on the stack. The fractions of s- and p-
        polarised light must sum to 1.

    Returns
    -------
    gen_x : dict
        Carrier generation rate profile accounting for s- and p- polarisation fractions
        in the illumination source.
    """
    return {
        wavelength: (
            s_fraction * gen_x_s[wavelength] + p_fraction * gen_x_p[wavelength]
        )
        for wavelength in gen_x_s.keys()
    }


# replaces equivalent broken function in tmm module
def inc_find_absorp_analytic_fn(layer, inc_data):
    """
    Outputs an absorp_analytic_fn object for a coherent layer within a
    partly-incoherent stack.
    inc_data is output of incoherent_main()
    """
    j = inc_data["stack_from_all"][layer]
    if np.isnan(j).any():
        raise ValueError("layer must be coherent for this function!")
    [stackindex, withinstackindex] = j

    forwardfunc = tmm.absorp_analytic_fn()
    forwardfunc.fill_in(inc_data["coh_tmm_data_list"][stackindex], withinstackindex)
    forwardfunc.scale(inc_data["stackFB_list"][stackindex][0])

    backfunc = tmm.absorp_analytic_fn()
    backfunc.fill_in(inc_data["coh_tmm_bdata_list"][stackindex], -1 - withinstackindex)
    backfunc.scale(inc_data["stackFB_list"][stackindex][1])
    backfunc.flip()

    return forwardfunc.add(backfunc)


def calculate_absorption_profile(inc_tmm_pol, c_list, d_list, dx):
    """Calculate an absorption profile in coherent layers in the stack.

    Parameters
    ----------
    inc_tmm_pol : dict
        Incoherent tmm dictionary.
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    d_list : list of float or int
        List of thicknesses in nm for each layer in layers.
    dx : float
        Position step in nm.

    Returns
    -------
    abs_x_pol : np.array
        Position resolved absorption through the thickness of coherent layers in the
        stack.
    abs_x_pol_int : np.array
        Integrated absorption profile in each coherent layer.
    """
    abs_x_pol = np.array([])
    abs_x_pol_int = np.array([])
    for cix, coh in enumerate(c_list):
        # only calculate absorption profile for coherent layers
        if coh == "c":
            thickness = d_list[cix]
            layer_x_list = np.linspace(0, thickness, dx)
            layer_abs_an = inc_find_absorp_analytic_fn(cix, inc_tmm_pol)
            layer_abs_x = layer_abs_an.run(layer_x_list)
            layer_abs_x_int = scipy.integrate.simpson(layer_abs_x, layer_x_list)

            abs_x_pol = np.concatenate((abs_x_pol, layer_abs_x))
            abs_x_pol_int = np.concatenate((abs_x_pol_int, layer_abs_x_int))

    return abs_x_pol, abs_x_pol_int


def get_x_list(c_list, d_list, dx):
    """Calculate list of x positions used for the generation profile.

    Parameters
    ----------
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    d_list : list of float or int
        List of thicknesses in nm for each layer in layers.
    dx : float
        Position step in nm.

    Returns
    -------
    x_list : np.array
        X positions in nm at which generation profile is calculated.
    """
    x_list = np.array([])
    for cix, coh in enumerate(c_list):
        # only calculate positions for coherent layers
        if coh == "c":
            thickness = d_list[cix]
            layer_x_list = np.linspace(0, thickness, dx)

            x_list = np.concatenate([x_list, layer_x_list])

    return x_list


def calculate_total_integrated_absorption(
    abs_x_int_s, abs_x_int_p, s_fraction, p_fraction
):
    """Calculate total integrated absorption in a layer from a absorption profile.

    This function accounts for the fraction of s- and p- polarised light in the
    illumination source.

    Parameters
    ----------
    abs_x_int_s : dict
        Integrated absorption profile in each coherent layer for s-polarised light.
    abs_x_int_p : dict
        Integrated absorption profile in each coherent layer for p-polarised light.
    s_fraction : float
        Fraction of s-polarised light incident on the stack. The fractions of s- and p-
        polarised light must sum to 1.
    p_fraction : float
        Fraction of s-polarised light incident on the stack. The fractions of s- and p-
        polarised light must sum to 1.

    Returns
    -------
    abs_x_int : np.array
        Integrated absorption profile in each coherent layer accounting for s- and p-
        polarisation fractions in the illumination source. Column 0 is wavelengths,
        then one column for each coherent layer.
    """
    abs_x_int = []
    for wavelength in abs_x_int_s.keys():
        total_abs_x_int = (
            s_fraction * abs_x_int_s[wavelength] + p_fraction * abs_x_int_p[wavelength]
        )
        abs_x_int.append([wavelength] + total_abs_x_int.tolist())

    return np.array(abs_x_int)


def plot_rta(rta):
    """Plot cumulative reflection, transmission, and absorption data for all layers.

    Parameters
    ----------
    rta : array
        Reflection, transmission, and absorption spectra from tmm calculations.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))

    cum_sum = np.zeros(np.shape(rta)[0])
    for col in range(1, np.shape(rta)[1]):
        # get legend prefix indicating R and T for first and last layers respectively
        if col == 1:
            prefix = "R, "
        elif col == np.shape(rta)[1] - 1:
            prefix = "T, "
        else:
            prefix = ""

        ax1.fill_between(
            rta[:, 0], cum_sum, cum_sum + rta[:, col], label=f"{prefix}{layers[col-1]}"
        )
        cum_sum += rta[:, col]

    ax1.set_xlim(np.min(rta[:, 0]), np.max(rta[:, 0]))
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Cumulative fractional R, T, and A")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    fig.show()


def plot_eqe(rta, active_layer_ix, c_list=None, abs_x_int=None):
    """Plot the ideal external quantum efficiency spectrum.

    Assuming internal quantum efficiency is 100%, i.e. all absorbed photons in the
    active layer generate free electron-hole pairs, all of which get collected.

    Parameters
    ----------
    rta : array
        Reflection, transmission, and absorption spectra from tmm calculations.
    active_layer_ix : int
        Index of the active layer in the layer list.
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    abs_x_int : array
        Array of integrated position resolved absorption. Used to validate consistency
        between tmm absorption calculation and position resolved absorption
        calculation.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(rta[:, 0], rta[:, active_layer_ix + 1], label="tmm absorption")

    if (c_list is not None) and (abs_x_int is not None):
        # find number of incoherent layers before active layer to help get index of
        # active layer in coherent substack
        inc_preceding = sum(c_list[index] == "i" for index in range(active_layer_ix))
        ax1.plot(
            abs_x_int[:, 0],
            abs_x_int[:, active_layer_ix + 1 - inc_preceding],
            label="integrated absorption profile",
        )

    ax1.set_xlim(np.min(rta[:, 0]), np.max(rta[:, 0]))
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fractional ideal EQE")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    fig.show()


def plot_generation_profile(x_list, gen_x):
    """Plot generation profiles.

    Parameters
    ----------
    x_list : array
        X positions in nm at which generation profile is calculated.
    gen_x : dict
        Carrier generation rate profiles. Dictionary keys are wavelengths in nm.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))

    for wavelength, gen_x_wl in gen_x.items():
        ax1.plot(x_list, gen_x_wl, label=f"{wavelength} nm")

    ax1.set_xlim(np.min(x_list), np.max(x_list))
    ax1.set_ylim(0)
    ax1.set_ylabel("Carrier generation rate (cm^-3)")
    ax1.set_xlabel("Distance (nm)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    fig.show()


def export_rta(rta, layers, timestamp):
    """Export reflection, transmission, and absorption data.

    Parameters
    ----------
    rta : array
        Reflection, transmission, and absorption spectra from tmm calculations
        accounting for s- and p- polarisation fractions in the illumination source.
    layers : list of str
        List of layer names corresponding to file names.
    timestamp : int
        Calculation time stamp.
    """
    path = OUTPUT_FOLDER.joinpath(f"{timestamp}").joinpath(
        f"{timestamp}_{rta}.{FILE_EXT}"
    )

    header = "wavelength (nm)\t" + "\t".join(layers)

    np.savetxt(
        path, rta, header=header, delimiter="\t", newline="\n", comments="", fmt="%.9f"
    )


def export_generation_profiles(x_list, gen_x, timestamp):
    """Export carrier generation rate profiles for all wavelengths.

    Parameters
    ----------
    x_list : np.array
        X positions in nm at which generation profile is calculated.
    gen_x : dict
        Carrier generation rate profile accounting for s- and p- polarisation fractions
        in the illumination source.
    timestamp : int
        Calculation time stamp.
    """
    header = "distance (nm)\tG (cm^-3)"

    for wavelength, gen in gen_x.items():
        path = OUTPUT_FOLDER.joinpath(f"{timestamp}").joinpath(
            f"{timestamp}_G_{wavelength}nm.{FILE_EXT}"
        )

        data = np.array([x_list, gen]).T

        np.savetxt(
            path,
            data,
            header=header,
            delimiter="\t",
            newline="\n",
            comments="",
            fmt="%.9f",
        )


def run_tmm(
    layers,
    d_list,
    c_list,
    wavelength_min,
    wavelength_max,
    wavelength_step,
    active_layer_name=None,
    th_0=0,
    s_fraction=0.5,
    p_fraction=0.5,
    show_plots=True,
    profiles=True,
    dx=1,
    illumination=None,
    export_data=False,
):
    """Run the transfer matrix calculation.

    Parameters
    ----------
    layers : list of str
        List of layer names corresponding to file names.
    d_list : list of float or int
        List of thicknesses in nm for each layer in layers.
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    wavelength_min : float
        Lower bound of the wavelength range in nm.
    wavelength_max : float
        Upper bound of the wavelength range in nm.
    wavelength_step : float
        Wavelength step between wavelengths in the wavelength range in nm.
    active_layer_name : str
        Name of the active layer in the device stack.
    th_0 : numeric
        Incident angle in degrees, 0 deg = normal incidence.
    s_fraction : float
        Fraction of s-polarised light incident on the stack. The fractions of s- and p-
        polarised light must sum to 1.
    p_fraction : float
        Fraction of s-polarised light incident on the stack. The fractions of s- and p-
        polarised light must sum to 1.
    show_plots : bool
        Flag indicating whether or not to display plots of calculation output.
    profiles : bool
        Flag indicating whether or not to perform position resolved calculations.
    dx : float
        Position step in nm for position resolved calculations.
    illumination : str or None
        Name of illumination source used for calculating carrier generation rate
        profiles. If set to `None` the generation profile will not be calculated.
    export_data : bool
        Flag indicating whether or not to export calculation output to files.

    Returns
    -------
    output : dict
        Dictionary of all input settings and output calculations.
    """
    timestamp = int(time.time())

    # get wavelegnths
    wavelength_n = int(((wavelength_max - wavelength_min) / wavelength_step) + 1)
    wavelengths = np.linspace(
        wavelength_min, wavelength_max, wavelength_n, endpoint=True
    )

    # load and interpolate layer data
    int_n_list = get_interpolated_n_list(layers)

    # validate polarisation fractions
    if not (
        (s_fraction + p_fraction == 1)
        and (1 >= s_fraction >= 0)
        and (1 >= p_fraction >= 0)
    ):
        raise ValueError(
            "Invalid s_fraction and/or p_fraction. Both must be in the range 0-1 and "
            + "sum to 1."
        )

    # perform incoherent tmm calculations for each polarisation state as required
    if s_fraction > 0:
        inc_tmm_s = calculate_tmm_spectra(
            int_n_list, d_list, c_list, wavelengths, th_0, "s"
        )
        rta_s = calculate_rta_spectra(inc_tmm_s)

    if p_fraction > 0:
        inc_tmm_p = calculate_tmm_spectra(
            int_n_list, d_list, c_list, wavelengths, th_0, "p"
        )
        rta_p = calculate_rta_spectra(inc_tmm_p)

    # calculate net RTA accounting for s and p fractions
    if s_fraction == 1:
        rta = rta_s
    elif p_fraction == 1:
        rta = rta_p
    else:
        rta = calculate_total_rta_spectra(rta_s, rta_p, s_fraction, p_fraction)

    # calculate absorption profiles
    if profiles is True:
        abs_x_s = {}
        abs_x_p = {}
        abs_x_int_s = {}
        abs_x_int_p = {}
        for wavelength in wavelengths:
            # only calculate for a polarisation state if required
            if s_fraction > 0:
                inc_data = inc_tmm_s[wavelength]
                abs_x_s_wl, abs_x_int_s_wl = calculate_absorption_profile(
                    inc_data, c_list, d_list, dx
                )
                abs_x_s[wavelength] = abs_x_s_wl
                abs_x_int_s[wavelength] = abs_x_int_s_wl

            if p_fraction > 0:
                inc_data = inc_tmm_p[wavelength]
                abs_x_p_wl, abs_x_int_p_wl = calculate_absorption_profile(
                    inc_data, c_list, d_list, dx
                )
                abs_x_p[wavelength] = abs_x_p_wl
                abs_x_int_p[wavelength] = abs_x_int_p_wl

        # calculate net integrated absorption profiles
        if s_fraction == 1:
            abs_x_int = np.array(
                [[wavelength] + abs_int for wavelength, abs_int in abs_x_int_s.items()]
            )
        elif p_fraction == 1:
            abs_x_int = np.array(
                [[wavelength] + abs_int for wavelength, abs_int in abs_x_int_p.items()]
            )
        else:
            abs_x_int = calculate_total_integrated_absorption(
                abs_x_int_s, abs_x_int_p, s_fraction, p_fraction
            )

        # calculate x positions for generation profile
        x_list = get_x_list(c_list, d_list, dx)

    # calculate generation profiles for each wavelegnth
    if (profiles is True) and (illumination is not None):
        illumination_path = REFRACTIVE_INDEX_FOLDER.joinpath(
            f"{illumination}.{FILE_EXT}"
        )
        illumination_data = load_illumination_data(illumination_path)

        gen_x_s = {}
        gen_x_p = {}
        for wavelength in wavelengths:
            # only calculate for a polarisation state if required
            if s_fraction > 0:
                gen_x_s[wavelength] = (
                    abs_x_s[wavelength]
                    * illumination_data(wavelength)
                    / wavelength_to_energy(wavelength)
                )

            if p_fraction > 0:
                gen_x_p[wavelength] = (
                    abs_x_p[wavelength]
                    * illumination_data(wavelength)
                    / wavelength_to_energy(wavelength)
                )

        # calculate net generation profiles
        if s_fraction == 1:
            gen_x = gen_x_s
        elif p_fraction == 1:
            gen_x = gen_x_p
        else:
            gen_x = calculate_total_generation_profile(
                gen_x_s, gen_x_p, s_fraction, p_fraction
            )

    # show plots
    if show_plots is True:
        # plot cumulative absorbtion
        plot_rta(rta)

        # look up ideal EQE, i.e. assume 100% IQE
        if active_layer_name is not None:
            # look up active layer index in full stack
            active_layer_ix = layers.index(active_layer_name)

            if profiles is True:
                # plot eqe compared to integrated absorption profiles
                plot_eqe(
                    rta,
                    active_layer_ix,
                    c_list,
                    abs_x_int,
                )
            else:
                # don't attempt to plot integrated absorption
                plot_eqe(
                    rta,
                    active_layer_ix,
                )

        # plot generation profiles for each wavelength
        if (profiles is True) and (illumination is not None):
            plot_generation_profile(x_list, gen_x)

    if export_data is True:
        export_rta(rta, layers, timestamp)

        if (profiles is True) and (illumination is not None):
            export_generation_profiles(x_list, gen_x, timestamp)

    return {
        "layers": layers,
        "d_list": d_list,
        "c_list": c_list,
        "wavelength_min": wavelength_min,
        "wavelength_max": wavelength_max,
        "wavelength_step": wavelength_step,
        "th_0": th_0,
        "s_fraction": s_fraction,
        "p_fraction": p_fraction,
        "active_layer": active_layer,
        "show_plots": show_plots,
        "profiles": profiles,
        "dx": profiles,
        "illumination": illumination,
        "export_data": export_data,
    }


if __name__ == "__main__":
    # list of layer names corresponding to file names
    layers = [
        "air",
        "soda_lime_glass",
        "ito",
        "ptaa",
        "wbg-perovskite",
        "pcbm",
        "bcp",
        "au",
        "air",
    ]
    active_layer = "wbg-perovskite"

    # thickness in nm for each layer in layers
    d_list = [np.inf, 1.1e6, 110, 20, 500, 30, 10, 100, np.inf]

    # coherent ("c")/incoherent ("i") label for each layer in layers
    c_list = ["i", "i", "c", "c", "c", "c", "c", "c", "i"]

    # wavelength range of interest in nm
    wavelength_min = 310
    wavelength_max = 890
    wavelength_step = 1

    # incident angle in degrees, 0 deg = normal incidence
    th_0 = 0

    # name of file containing spectral irradiance for the illumination source
    illumination = "am15g"

    # spacing for calculating generation profiles
    dx = 1

    # s and p polarisation state fractions
    s_fraction = 0.5
    p_fraction = 0.5

    show_plots = True
    profiles = True
    export_data = False

    calc = run_tmm(
        layers,
        d_list,
        c_list,
        wavelength_min,
        wavelength_max,
        wavelength_step,
        th_0,
        s_fraction,
        p_fraction,
        active_layer,
        show_plots,
        profiles,
        dx,
        illumination,
        export_data,
    )
