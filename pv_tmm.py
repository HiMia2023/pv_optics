"""Calculate optical absorption and carrier generatation rate in a PV device stack."""

import argparse
import os
import pathlib
import shutil
import time
from typing import Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy.constants
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import tmm
import yaml


APPLICATION_FOLDER = pathlib.Path.cwd()
REFRACTIVE_INDEX_FOLDER = APPLICATION_FOLDER.joinpath("refractive_index")
ILLUMINATION_FOLDER = APPLICATION_FOLDER.joinpath("illumination")
INPUT_FOLDER = APPLICATION_FOLDER.joinpath("input")
OUTPUT_FOLDER = APPLICATION_FOLDER.joinpath("output")
DATA_FILE_EXT = "tsv"

if OUTPUT_FOLDER.exists() is False:
    os.mkdir(OUTPUT_FOLDER)


def wavelength_to_energy(wavelength: Union[float, NDArray]) -> Union[float, NDArray]:
    """Convert wavelength/s in nm to energy in eV.

    Parameters
    ----------
    wavelength : float or np.array
        Wavelength/s in nm.

    Returns
    -------
    E : float or array
        Energy/s in J.
    """
    return scipy.constants.Planck * scipy.constants.speed_of_light / (wavelength * 1e-9)


def integrated_jsc(wavelengths: NDArray, eqe: NDArray, irradiance: NDArray) -> float:
    """Calculate integrated Jsc in mA/cm^2.

    Parameters
    ----------
    wavelengths : np.array
        Array of wavelengths in nm.
    eqe : np.array
        Fractional external quantum efficiency spectrum.
    irradiance : np.array
        Irradiance spectrum in W/m^2/nm.

    Returns
    -------
    jsc : float
        Short-circuit current density in mA/cm^2.
    """
    flux = irradiance / wavelength_to_energy(wavelengths)
    integrand = scipy.integrate.simpson(eqe * flux, wavelengths)

    return scipy.constants.elementary_charge * integrand * 1000 / 10000


def load_config(filename: str) -> Dict:
    """Load a configuration settings file.

    A configuration settings file is a yaml file whose keys correspond to arguments of
    the `run_tmm()` function.

    Parameters
    ----------
    filename : str
        Name of config file.

    Returns
    -------
    tmm_config : dict
        Dictionary of settings to be passed to the `run_tmm()` function.
    """
    # load config file
    path = INPUT_FOLDER.joinpath(f"{filename}")
    with open(path, encoding="utf-8") as file:
        tmm_config = yaml.load(file, Loader=yaml.SafeLoader)

    # cast elements of d_list to correct type
    tmm_config["d_list"] = [
        np.inf if d == "inf" else float(d) for d in tmm_config["d_list"]
    ]

    return tmm_config


def load_nk_data(
    filepath: Union[str, pathlib.Path]
) -> Tuple[
    scipy.interpolate._interpolate.interp1d, scipy.interpolate._interpolate.interp1d
]:
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


def load_illumination_data(
    illumination: str,
) -> scipy.interpolate._interpolate.interp1d:
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
    illumination : str or None
        Name of illumination source used for calculating carrier generation rate
        profiles.

    Returns
    -------
    F_int : scipy.interpolate.interp1d
        Scipy interpolation object for spectral irradiance.
    """
    illumination_path = ILLUMINATION_FOLDER.joinpath(f"{illumination}.{DATA_FILE_EXT}")
    data = np.genfromtxt(illumination_path, skip_header=3, delimiter="\t")

    return scipy.interpolate.interp1d(data[:, 0], data[:, 1], kind="cubic")


def get_interpolated_n_list(layers: List[str]) -> List:
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
        path = REFRACTIVE_INDEX_FOLDER.joinpath(f"{layer}.{DATA_FILE_EXT}")
        int_n_list.append(load_nk_data(path))

    return int_n_list


def calculate_tmm_spectra(
    int_n_list: List,
    d_list: List[float],
    c_list: List[str],
    wavelengths,
    th_0: float,
    polarisation: str,
) -> Dict:
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
    th_0 : float
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


def calculate_rta_spectra(inc_tmm_pol: Dict) -> NDArray:
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


def calculate_total_rta_spectra(
    rta_s: NDArray, rta_p: NDArray, s_fraction: float = 0.5, p_fraction: float = 0.5
) -> NDArray:
    """Calcualte total reflection, transmission, and absorption spectra.

    This function accounts for the fraction of s- and p- polarised light in the
    illumination source.

    Parameters
    ----------
    rta_s : np.array
        Reflection, transmission, and absorption spectra from tmm calculations for
        s-polarised light.
    rta_p : np.array
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
    rta : np.array
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
    gen_x_s: Dict, gen_x_p: Dict, s_fraction: float = 0.5, p_fraction: float = 0.5
) -> Dict:
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


def calculate_absorption_profile(
    inc_tmm_pol: Dict, c_list: List[str], d_list: List[float], xstep: float
) -> Tuple[NDArray, NDArray]:
    """Calculate an absorption profile in coherent layers in the stack.

    Parameters
    ----------
    inc_tmm_pol : dict
        Incoherent tmm dictionary.
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    d_list : list of float
        List of thicknesses in nm for each layer in layers.
    xstep : float
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
    abs_x_pol_int = []
    for cix, coh in enumerate(c_list):
        # only calculate absorption profile for coherent layers
        if coh == "c":
            thickness = d_list[cix]
            n_x = int((thickness / xstep))
            layer_x_list = np.linspace(0, thickness, n_x, endpoint=False)
            layer_abs_an = inc_find_absorp_analytic_fn(cix, inc_tmm_pol)

            # calculate absorption profile
            # returns a complex number with 0 imaginary part so discard it
            layer_abs_x = layer_abs_an.run(layer_x_list).real

            # integrate absorption profile for sanity check against tmm output
            layer_abs_x_int = scipy.integrate.simpson(layer_abs_x, layer_x_list)

            abs_x_pol = np.concatenate((abs_x_pol, layer_abs_x))
            abs_x_pol_int.append(layer_abs_x_int)

    return abs_x_pol, np.array(abs_x_pol_int)


def get_x_list(c_list: List[str], d_list: List[float], xstep: float) -> NDArray:
    """Calculate list of x positions used for the generation profile.

    Parameters
    ----------
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    d_list : list of float
        List of thicknesses in nm for each layer in layers.
    xstep : float
        Position step in nm.

    Returns
    -------
    x_list : np.array
        X positions in nm at which generation profile is calculated.
    """
    x_list = np.empty(0)
    cum_thickness = 0
    for cix, coh in enumerate(c_list):
        # only calculate positions for coherent layers
        if coh == "c":
            thickness = d_list[cix]
            n_x = int((thickness / xstep))
            layer_x_list = np.linspace(
                cum_thickness, cum_thickness + thickness, n_x, endpoint=False
            )
            x_list = np.concatenate([x_list, layer_x_list])
            cum_thickness += thickness

    return x_list


def calculate_total_integrated_absorption(
    abs_x_int_s: Dict, abs_x_int_p: Dict, s_fraction: float, p_fraction: float
) -> NDArray:
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


def plot_rta(rta: NDArray, layers: List[str]):
    """Plot cumulative reflection, transmission, and absorption data for all layers.

    Parameters
    ----------
    rta : np.array
        Reflection, transmission, and absorption spectra from tmm calculations.
    layers : list of str
        List of layer names.
    """
    fig, ax1 = plt.subplots()

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


def plot_eqe(
    rta: NDArray,
    active_layer_ixs: List[int],
    layers: List[str],
    c_list: Optional[List[str]] = None,
    abs_x_int: Optional[NDArray] = None,
):
    """Plot the ideal external quantum efficiency spectrum.

    Assuming internal quantum efficiency is 100%, i.e. all absorbed photons in the
    active layer generate free electron-hole pairs, all of which get collected.

    Parameters
    ----------
    rta : np.array
        Reflection, transmission, and absorption spectra from tmm calculations.
    active_layer_ixs : list of int
        List of indices of the active layers in the layer list.
    layers : list of str
        List of layer names.
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    abs_x_int : np.array
        Array of integrated position resolved absorption. Used to validate consistency
        between tmm absorption calculation and position resolved absorption
        calculation.
    """
    fig, ax1 = plt.subplots()

    eqe_sum = np.zeros(len(rta[:, 0]))
    for active_layer_ix in active_layer_ixs:
        ax1.plot(
            rta[:, 0],
            rta[:, active_layer_ix + 1],
            label=f"{layers[active_layer_ix]} tmm",
        )
        eqe_sum += rta[:, active_layer_ix + 1]

        if (c_list is not None) and (abs_x_int is not None):
            # find number of incoherent layers before active layer to help get index of
            # active layer in coherent substack
            inc_preceding = sum(
                c_list[index] == "i" for index in range(active_layer_ix)
            )
            ax1.plot(
                abs_x_int[:, 0],
                abs_x_int[:, active_layer_ix + 1 - inc_preceding],
                label=f"{layers[active_layer_ix]} profile",
            )

    if len(active_layer_ixs) > 1:
        ax1.plot(rta[:, 0], eqe_sum, label="sum tmm")

    ax1.set_xlim(np.min(rta[:, 0]), np.max(rta[:, 0]))
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fractional ideal EQE")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    fig.show()


def plot_generation_profile(x_list: NDArray, gen_x: Dict):
    """Plot generation profiles.

    Parameters
    ----------
    x_list : np.array
        X positions in nm at which generation profile is calculated.
    gen_x : dict
        Carrier generation rate profiles. Dictionary keys are wavelengths in nm.
    """
    fig, ax1 = plt.subplots()

    for wavelength, gen_x_wl in gen_x.items():
        ax1.plot(x_list, gen_x_wl, label=f"{wavelength} nm")

    ax1.set_xlim(np.min(x_list), np.max(x_list))
    ax1.set_ylim(0)
    ax1.set_ylabel("Carrier generation rate (m^-3)")
    ax1.set_xlabel("Distance (nm)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    fig.show()


def export_rta(rta: NDArray, layers: List[str], timestamp: int):
    """Export reflection, transmission, and absorption data.

    Parameters
    ----------
    rta : np.array
        Reflection, transmission, and absorption spectra from tmm calculations
        accounting for s- and p- polarisation fractions in the illumination source.
    layers : list of str
        List of layer names corresponding to file names.
    timestamp : int
        Calculation time stamp.
    """
    os.mkdir(OUTPUT_FOLDER.joinpath(f"{timestamp}"))

    path = OUTPUT_FOLDER.joinpath(f"{timestamp}").joinpath(
        f"{timestamp}_rta.{DATA_FILE_EXT}"
    )

    header = "wavelength (nm)\t" + "\t".join(layers)

    np.savetxt(
        path, rta, header=header, delimiter="\t", newline="\n", comments="", fmt="%.9f"
    )


def export_generation_profiles(x_list: NDArray, gen_x, timestamp: int):
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
    header = "distance (nm)"
    data = x_list
    for wavelength, gen in gen_x.items():
        data = np.column_stack((data, gen))
        header += f"\tG_{wavelength}nm (m^-3)"

    path = OUTPUT_FOLDER.joinpath(f"{timestamp}").joinpath(
        f"{timestamp}_G.{DATA_FILE_EXT}"
    )

    np.savetxt(
        path,
        data,
        header=header,
        delimiter="\t",
        newline="\n",
        comments="",
        fmt="%.9e",
    )


def get_wavelengths(
    wavelength_min: float, wavelength_max: float, wavelength_step: float
) -> NDArray:
    """Generate wavelengths array.

    Parameters
    ----------
    wavelength_min : float
        Lower bound of the wavelength range in nm.
    wavelength_max : float
        Upper bound of the wavelength range in nm.
    wavelength_step : float
        Wavelength step between wavelengths in the wavelength range in nm.

    Returns
    -------
    wavelengths : np.array
        Wavelengths in nm.
    """
    wavelength_n = int(((wavelength_max - wavelength_min) / wavelength_step) + 1)
    return np.linspace(wavelength_min, wavelength_max, wavelength_n, endpoint=True)


def run_tmm(
    layers: List[str],
    d_list: List[float],
    c_list: List[str],
    wavelength_min: float,
    wavelength_max: float,
    wavelength_step: float,
    config_filename: str,
    active_layer_names: Optional[List[str]] = None,
    th_0: float = 0,
    s_fraction: float = 0.5,
    p_fraction: float = 0.5,
    show_plots: bool = True,
    profiles: bool = True,
    xstep: float = 1,
    illumination: Optional[str] = None,
    export_data: bool = False,
) -> Dict:
    """Run the transfer matrix calculation.

    Parameters
    ----------
    layers : list of str
        List of layer names corresponding to file names.
    d_list : list of float
        List of thicknesses in nm for each layer in layers.
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    wavelength_min : float
        Lower bound of the wavelength range in nm.
    wavelength_max : float
        Upper bound of the wavelength range in nm.
    wavelength_step : float
        Wavelength step between wavelengths in the wavelength range in nm.
    config_filename : str
        Name of calculation configuration file.
    active_layer_names : list of str
        List of active layer (layers that produce photocurrent) names in the device
        stack.
    th_0 : float
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
    xstep : float
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

    timestamp = int(time.time())

    wavelengths = get_wavelengths(wavelength_min, wavelength_max, wavelength_step)

    # load and interpolate layer data
    int_n_list = get_interpolated_n_list(layers)

    # perform incoherent tmm calculations for each polarisation state as required
    if s_fraction > 0:
        inc_tmm_s = calculate_tmm_spectra(
            int_n_list, d_list, c_list, wavelengths, th_0, "s"
        )
        rta_s = calculate_rta_spectra(inc_tmm_s)
    else:
        inc_tmm_s = {}
        rta_s = np.empty(0)

    if p_fraction > 0:
        inc_tmm_p = calculate_tmm_spectra(
            int_n_list, d_list, c_list, wavelengths, th_0, "p"
        )
        rta_p = calculate_rta_spectra(inc_tmm_p)
    else:
        inc_tmm_p = {}
        rta_p = np.empty(0)

    # calculate net RTA accounting for s and p fractions
    if s_fraction == 1:
        rta = rta_s
    elif p_fraction == 1:
        rta = rta_p
    else:
        rta = calculate_total_rta_spectra(rta_s, rta_p, s_fraction, p_fraction)

    # calculate absorption profiles
    abs_x_s = {}
    abs_x_p = {}
    abs_x_int_s = {}
    abs_x_int_p = {}
    if profiles is True:
        for wavelength in wavelengths:
            # only calculate for a polarisation state if required
            if s_fraction > 0:
                inc_data = inc_tmm_s[wavelength]
                abs_x_s_wl, abs_x_int_s_wl = calculate_absorption_profile(
                    inc_data, c_list, d_list, xstep
                )
                abs_x_s[wavelength] = abs_x_s_wl
                abs_x_int_s[wavelength] = abs_x_int_s_wl

            if p_fraction > 0:
                inc_data = inc_tmm_p[wavelength]
                abs_x_p_wl, abs_x_int_p_wl = calculate_absorption_profile(
                    inc_data, c_list, d_list, xstep
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
        x_list = get_x_list(c_list, d_list, xstep)
    else:
        abs_x_int = np.empty(0)
        x_list = np.empty(0)

    # calculate generation profiles for each wavelegnth
    gen_x_s = {}
    gen_x_p = {}
    if (profiles is True) and (illumination is not None):
        illumination_data = load_illumination_data(illumination)

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
    else:
        illumination_data = None
        gen_x = {}

    # show plots
    if show_plots is True:
        # plot cumulative absorbtion
        plot_rta(rta, layers)

        # look up ideal EQE, i.e. assume 100% IQE
        if active_layer_names is not None:
            # look up active layer index in full stack
            active_layer_ixs = [layers.index(name) for name in active_layer_names]

            if profiles is True:
                # plot eqe compared to integrated absorption profiles
                plot_eqe(
                    rta,
                    active_layer_ixs,
                    layers,
                    c_list,
                    abs_x_int,
                )
            else:
                # don't attempt to plot integrated absorption
                plot_eqe(
                    rta,
                    active_layer_ixs,
                    layers,
                )

        # plot generation profiles for each wavelength
        if (profiles is True) and (illumination is not None):
            plot_generation_profile(x_list, gen_x)

    if export_data is True:
        export_rta(rta, layers, timestamp)

        if (profiles is True) and (illumination is not None):
            export_generation_profiles(x_list, gen_x, timestamp)

        # copy config into output
        src = INPUT_FOLDER.joinpath(config_filename)
        dst = OUTPUT_FOLDER.joinpath(f"{timestamp}").joinpath(config_filename)
        shutil.copy2(src, dst)

    return {
        "layers": layers,
        "d_list": d_list,
        "c_list": c_list,
        "wavelength_min": wavelength_min,
        "wavelength_max": wavelength_max,
        "wavelength_step": wavelength_step,
        "wavelengths": wavelengths,
        "th_0": th_0,
        "s_fraction": s_fraction,
        "p_fraction": p_fraction,
        "active_layer_names": active_layer_names,
        "show_plots": show_plots,
        "profiles": profiles,
        "xstep": profiles,
        "illumination": illumination,
        "export_data": export_data,
        "inc_tmm_s": inc_tmm_s,
        "inc_tmm_p": inc_tmm_p,
        "rta_s": rta_s,
        "rta_p": rta_p,
        "rta": rta,
        "x_list": x_list,
        "abs_x_s": abs_x_s,
        "abs_x_p": abs_x_p,
        "abs_x_int_s": abs_x_int_s,
        "abs_x_int_p": abs_x_int_p,
        "illumination_data": illumination_data,
        "gen_x_s": gen_x_s,
        "gen_x_p": gen_x_p,
        "gen_x": gen_x,
        "timestamp": timestamp,
    }


def optimise_thicknesses(
    layers: List[str],
    d_list: List[float],
    c_list: List[str],
    wavelength_min: float,
    wavelength_max: float,
    wavelength_step: float,
    active_layer_names: List[str],
    optimisation_layer_names: List[str],
    d_min_list: List[float],
    d_max_list: List[float],
    illumination: str,
    config_filename: str,
    th_0: float = 0,
    s_fraction: float = 0.5,
    p_fraction: float = 0.5,
    show_plots: bool = True,
    profiles: bool = True,
    xstep: float = 1,
    export_data: bool = False,
) -> Dict:
    """Optimise layer thicknesses using transfer matrix calculations.

    Parameters
    ----------
    layers : list of str
        List of layer names corresponding to file names.
    d_list : list of float
        List of thicknesses in nm for each layer in layers.
    c_list : list of str
        Coherent ("c")/incoherent ("i") label for each layer in layers.
    wavelength_min : float
        Lower bound of the wavelength range in nm.
    wavelength_max : float
        Upper bound of the wavelength range in nm.
    wavelength_step : float
        Wavelength step between wavelengths in the wavelength range in nm.
    active_layer_names : list of str
        List of active layer (layers that produce photocurrent) names in the device
        stack.
    optimisation_layer_names : list of str
        List of layer names to optimise.
    d_min_list : List of float
        List of minimum thicknesses in nm for each layer in optimisation_layer_names.
    d_max_list : list of float
        List of maximum thicknesses in nm for each layer in optimisation_layer_names.
    illumination : str
        Name of illumination source used for calculating carrier generation rate
        profiles. If set to `None` the generation profile will not be calculated.
    config_filename : str
        Name of calculation configuration file.
    th_0 : float
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
    xstep : float
        Position step in nm for position resolved calculations.
    export_data : bool
        Flag indicating whether or not to export calculation output to files.

    Returns
    -------
    output : dict
        Dictionary of optimisation output.
    """

    def minimisation_function(d_guess: Union[List[float], NDArray], *min_args) -> float:
        """Minimise this function to find optimal thicknesses.

        Parameters
        ----------
        d_guess : list of float or np.array
            Latest guess at optimal thickness in nm.
        *min_args : list
            Additional arguments required to define the function.
        """
        layers = min_args[0]
        d_list = min_args[1]
        c_list = min_args[2]
        wavelength_min = min_args[3]
        wavelength_max = min_args[4]
        wavelength_step = min_args[5]
        illumination = min_args[6]
        optimisation_layer_ixs = min_args[7]
        active_layer_ixs = min_args[8]
        active_layer_names = min_args[9]
        wavelengths = min_args[10]
        irradiance = min_args[11]
        th_0 = min_args[12]
        s_fraction = min_args[13]
        p_fraction = min_args[14]
        illumination = min_args[15]
        config_filename = min_args[16]

        for optimisation_layer_ix, guess in zip(optimisation_layer_ixs, d_guess):
            d_list[optimisation_layer_ix] = guess

        tmm_output = run_tmm(
            layers,
            d_list,
            c_list,
            wavelength_min,
            wavelength_max,
            wavelength_step,
            config_filename,
            active_layer_names,
            th_0,
            s_fraction,
            p_fraction,
            show_plots=False,
            profiles=False,
            illumination=illumination,
            export_data=False,
        )

        eqes = [tmm_output["rta"][:, layer_ix + 1] for layer_ix in active_layer_ixs]
        jscs = [integrated_jsc(wavelengths, eqe, irradiance) for eqe in eqes]

        return -min(jscs)

    # calculate derived args required for minimiser
    active_layer_ixs = [layers.index(name) for name in active_layer_names]
    optimisation_layer_ixs = [layers.index(name) for name in optimisation_layer_names]
    wavelengths = get_wavelengths(wavelength_min, wavelength_max, wavelength_step)
    f_irradiance = load_illumination_data(illumination)
    irradiance = f_irradiance(wavelengths)

    # set up minimiser arguments
    bounds = list(zip(d_min_list, d_max_list))
    min_args = (
        layers,
        d_list,
        c_list,
        wavelength_min,
        wavelength_max,
        wavelength_step,
        illumination,
        optimisation_layer_ixs,
        active_layer_ixs,
        active_layer_names,
        wavelengths,
        irradiance,
        th_0,
        s_fraction,
        p_fraction,
        illumination,
        config_filename,
    )
    d_init = [d_list[opt_layer_ix] for opt_layer_ix in optimisation_layer_ixs]

    # run the minimisation
    result = scipy.optimize.differential_evolution(
        minimisation_function,
        bounds=bounds,
        args=min_args,
        workers=1,
        disp=True,
        init="sobol",
        x0=d_init,
    )

    # report the results
    print(result)
    if result.success:
        print("\nOptimised thicknesses\n---------------------")
        for d_res, optimisation_layer_name in zip(result.x, optimisation_layer_names):
            print(f"{optimisation_layer_name} thickness = {d_res} nm")

        # insert optimal thicknesses into d_list
        d_opt = d_list
        for opt_layer_ix, d_res in zip(optimisation_layer_ixs, result.x):
            d_opt[opt_layer_ix] = d_res

        # calculate tmm for optimised thickness
        opt_output = run_tmm(
            layers,
            d_opt,
            c_list,
            wavelength_min,
            wavelength_max,
            wavelength_step,
            config_filename,
            active_layer_names,
            th_0,
            s_fraction,
            p_fraction,
            show_plots,
            profiles,
            xstep,
            illumination,
            export_data,
        )

        eqes = [opt_output["rta"][:, layer_ix + 1] for layer_ix in active_layer_ixs]
        jscs = [integrated_jsc(wavelengths, eqe, irradiance) for eqe in eqes]

        print("\nOptimised Jsc's\n---------------")
        for active_layer_name, jsc in zip(active_layer_names, jscs):
            print(f"{active_layer_name} Jsc = {jsc} mA/cm^2")

        # export optimsation results
        if export_data:
            opt_output_dict = {
                "optimisation_layer_names": optimisation_layer_names,
                "optimised_thicknesses": result.x.tolist(),
                "optimised_jscs": [float(jsc) for jsc in jscs],
                "optimiser_output": str(result),
            }

            opt_output_path = OUTPUT_FOLDER.joinpath(
                f"{opt_output['timestamp']}"
            ).joinpath(f"{opt_output['timestamp']}_optimisation_output.yaml")

            with open(opt_output_path, "w", encoding="utf-8") as file:
                yaml.dump(opt_output_dict, file)

        return {"result": result, "tmm_output": opt_output}

    raise ValueError("Optimisation failed!")


def get_args():
    """Get command line arguments.

    Returns
    -------
    args : namedTuple
        Command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        help="Name of configuration yaml file, including extension",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # get cli args, i.e. config filename
    args = get_args()

    # load config dictionary from yaml file
    config = load_config(args.filename)

    # run tmm calculations
    try:
        # check if running an optimisation
        if len(config["optimisation_layer_names"]) > 0:
            print("Running optimisation...\n")
            calc = optimise_thicknesses(
                config["layers"],
                config["d_list"],
                config["c_list"],
                config["wavelength_min"],
                config["wavelength_max"],
                config["wavelength_step"],
                config["active_layer_names"],
                config["optimisation_layer_names"],
                config["d_min_list"],
                config["d_max_list"],
                config["illumination"],
                args.filename,
                config["th_0"],
                config["s_fraction"],
                config["p_fraction"],
                config["show_plots"],
                config["profiles"],
                config["xstep"],
                config["export_data"],
            )
    except KeyError as err:
        print(f"Cannot run optimisation, invalid key: {err}")
        # not optimising so just do standard tmm
        calc = run_tmm(
            config["layers"],
            config["d_list"],
            config["c_list"],
            config["wavelength_min"],
            config["wavelength_max"],
            config["wavelength_step"],
            args.filename,
            config["active_layer_names"],
            config["th_0"],
            config["s_fraction"],
            config["p_fraction"],
            config["show_plots"],
            config["profiles"],
            config["xstep"],
            config["illumination"],
            config["export_data"],
        )

        # report integrated Jscs
        active_layer_ixs = [
            config["layers"].index(name) for name in config["active_layer_names"]
        ]

        wavelengths = get_wavelengths(
            config["wavelength_min"],
            config["wavelength_max"],
            config["wavelength_step"],
        )
        f_irradiance = load_illumination_data(config["illumination"])
        irradiance = f_irradiance(wavelengths)

        eqes = [calc["rta"][:, layer_ix + 1] for layer_ix in active_layer_ixs]
        jscs = [integrated_jsc(wavelengths, eqe, irradiance) for eqe in eqes]

        print("\nJsc's\n-----")
        for active_layer_name, jsc in zip(config["active_layer_names"], jscs):
            print(f"{active_layer_name} Jsc = {jsc} mA/cm^2")

    # make sure plot windows don't close
    if config["show_plots"] is True:
        plt.show()
