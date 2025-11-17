[![pull requests](https://img.shields.io/github/issues-pr/mb4512/ezcascades.svg)](https://github.com/mb4512/ezcascades/pulls)
[![issues](https://img.shields.io/github/issues/mb4512/ezcascades.svg)](https://github.com/mb4512/ezcascades/issues/)
[![l## icense: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)

# ezcascades
LAMMPS Python script for simulating high-dose irradiation damage

Full details of the method are available at:
* Simultaneous cascades for large-scale high dose simulation: [10.1103/PhysRevMaterials.6.063601](https://doi.org/10.1103/PhysRevMaterials.6.063601)
* Varying recoil energies in different materials:             [10.1038/s41598-022-27087-w](https://doi.org/10.1038/s41598-022-27087-w)
* Cascades with recoil energies drawn from a spectrum:        [10.1038/s43246-024-00655-5](https://doi.org/10.1038/s43246-024-00655-5)

See the [project Wiki](../../wiki) for detailed information on features, installation, file structures, and working examples.

## Features

The ezcascades repository contains collection of Python scripts for driving the [LAMMPS Molecular Dynamics Simulator](https://github.com/lammps/lammps), compiled as a shared library, in order to simulate the evolution of microstructure under the influence of irradiation, simulated in the form of highly energetic atomic recoils. Simulation settings such as file paths, interatomic potential, system size, etc., are specified in json files, allowing for a quick and repeatable simulation setup. 

These scripts require LAMMPS to be built with the EXTRA-FIX and MANYBODY packages, for electronic stopping and embedded atom model potentials. LAMMPS must be compiled as a shared library linked to Python 3+ with the numpy, scipy, and mpi4py packages. See the [LAMMPS documentation](https://docs.lammps.org/Python_head.html) for more information on how to set this up.

The scripts support simulating single element materials and alloys, starting from the pristine crystal or a supplied structure, at thermal or athermal conditions, under displacement or stress constraints. The collection includes scripts for simulating large-scale high-dose microstructures through initialisation of multiple non-overlapping recoils per cascade iteration, and also scripts for simulating single-cascade events for the purpose of gathering defect generation statistics.

The collection contains the following scripts for high-dose simulations:
* [ezcascades.py: High-dose simulations via overlap collision cascades](../../wiki/High‐dose:-collision-cacades)
* [ezcra.py: High-dose simulations via creation relaxation algorithm](../../wiki/High‐dose:-Creation-relaxation-algorithm)

The collection also contains single-cascade scripts
* `geted.py`:   single-cascade simulations for obtaining displacement threshold statistics
* `getcasc.py`: single-cascade simulations for obtaining Frenkel pair generation statistics
* `getmelt.py`: single-cascade simulations for obtaining cascade melt size statistics

## Getting started

To get started, you need to build LAMMPS with the EXTRA-FIX package for electronic stopping, and optionally with the MANYBODY package to enable embedded atom model potentials (or other for machine-learned potentials). LAMMPS must be compiled as a shared library linked to Python 3+ with the numpy, scipy, and mpi4py packages. See the [LAMMPS documentation](https://docs.lammps.org/Python_head.html) for more information on how to set this up. To avoid compatability issues, the Python environment should be set up with the same modules (compilers, MPI version, ...) used for compiling the LAMMPS shared library. In many HPC systems, you can set up the Python environment locally as a virtual environment, installing the neccessary package via the `pip` command. 

Once LAMMPS is set up, clone this repository
```shell
$ git clone https://github.com/mb4512/ezcascades.git
```

You can test if your Python, mpi4py, LAMMPS environment works with the script **TODO**
```shell
$ mpirun -n 4 python tests/mpitest.py
```

Note that some HPC systems do not permit executing parallel processes on the login node. In that case, you can test the configuration by submitting this script as a batch job.

## Further information

See the [project Wiki](../../wiki) for detailed information on features, installation, file structures, and working examples.


