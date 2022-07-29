# pv_optics
Simulate the reflection, transmission, absorption in each layer, and carrier generation profiles of solar cells using the transfer matrix method.

## Installation and Usage
Create and activate a new Python (version >= 3.10) virtual environment e.g. using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), [venv](https://docs.python.org/3/library/venv.html) etc. Then clone this repository using [git](https://git-scm.com) and navigate to its newly created directory:
```
git clone https://github.com/jmball/pv_optics.git
cd pv_optics
```
Install the dependencies into the virtual environment using:
```
pip install -r requirements.txt
```
To run the program call:
```
python pv_optics.py --filename [config_file_name.yaml]
```
where `[config_file_name.yaml]` is the name of the simulation configuration file to run, which must be located in the "input" folder.