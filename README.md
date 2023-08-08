# pv_optics

Simulate and optimise the optical properties of solar cells using the transfer matrix method. Simulated parameters include spectra of the reflection, transmission, absorption in each layer, and carrier generation profiles, as well as total current generation under specific illumination conditions. The thicknesses of active layers in the stack can also be optimised to maximise photocurrent generation. This is particularly important in multijunction solar cells.

## Installation

Create and activate a new virtual environment e.g. using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), [venv](https://docs.python.org/3/library/venv.html) etc., with Python version 3.10 (may not work with other versions). Then either [download](https://github.com/jmball/pv_optics/archive/refs/heads/main.zip) and unzip (or clone using [git](https://git-scm.com)) this repository to a location of your choice. Using a terminal, e.g. Anaconda Prompt, with the virtual environment activated, navigate to the newly created directory using e.g.:

```
cd C:\Users\Name\Documents\Git\pv_optics
```

The exact path you use here will depend on where the pv_optics folder is located on your computer.

Then install the dependencies of the program into the virtual environment using:

```
pip install -r requirements.txt
```

## Usage

### Data files

Optical simulations of solar cells requires data files for the complex refractive index spectra of the layers of the device stack. Typically, complex refractive index data will be derived from spectroscopic ellipsometry measurements. Some literature data can be found in online databases e.g. [RefractiveIndex.info](https://refractiveindex.info). Please see the "refractive_index" folder for examples of the data format required by the program when including new material data for a simulation.

Simulating carrier generation profiles and total current generation additionally requires a data file for the spectral irradiance of the illumination source. Please see the "illumination" folder for examples of the data format required by the program when including new illumination source data for a simulation.

### Configuration files

The configuration of a simulation and/or optimisation of stack of layers is handled with configuration files located in the "input" folder. Please see the example configuration files, which can be used as templates, for how they should be written.

### GUI

From the command line, navigate to the pv_optics folder and activate the Python virtual environment created above. Then run the GUI by calling:

```
python pv_optics_gui.py
```

### Non-GUI

From the command line, navigate to the pv_optics folder and activate the Python virtual environment created above. Then run without the GUI by calling:

```
python pv_optics.py --filename [config_file_name.yaml]
```

where `[config_file_name.yaml]` is the name of the simulation configuration file to run, which must be located in the "input" folder.

### Output files

If required, the output data created by a simulation/optimisation can be saved for further processing without needing to re-run the simulation. This data can be found in the "output" folder, which is automatically created on first use of the program.
