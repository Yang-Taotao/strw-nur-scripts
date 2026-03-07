# strw-nur-scripts

[![Linux](https://img.shields.io/badge/platform-Linux-green?logo=linux&logoColor=white)](https://www.kernel.org/) [![Python](https://img.shields.io/badge/python=3.9.25-blue?logo=python&logoColor=yellow)](https://www.python.org/downloads/release/python-3925/) [![License](https://img.shields.io/github/license/Yang-Taotao/strw-nur-scripts?color=yellow)](LICENSE)

This is the documentation for **Numerical Recipes in Astrophysics** course at [Leiden Observatory](https://local.strw.leidenuniv.nl/):

[![NUR-A](https://img.shields.io/badge/NUR%20A-2026-blue)](https://studiegids.universiteitleiden.nl/courses/130599/numerical-recipes-in-astrophysics-a) [![NUR-B](https://img.shields.io/badge/NUR%20B-2026-blue)](https://studiegids.universiteitleiden.nl/courses/130600/numerical-recipes-in-astrophysics-b)

**Dated:** 2026-February-10

## Comment

- This is a collection of lecture notes and scripts used during the courses
- Containing scripts used for tutorial problems and assignments

## Additional

- scripts are written and tested under `conda` environment as specified through `environment.yml`
- scripts can also be run natively with base `python3` through shell script `run.sh` -> ***preferred***

## Usage

Start by cloning the repository:

```sh
git clone https://github.com/Yang-Taotao/strw-nur-scripts.git
cd strw-nur-scripts/
```

Example usage for running the `.py` scripts for assingmnets and compiling the associated `.tex` files:

```sh
chmod +x ./run.sh
./run.sh
```

Comments
- Relevant section of `run.sh` are kept in its comments for executing associated scripts for specific assignments. 
- See `file structure` below for details on where the files are located.

## file structure

```sh
.
├── src                     # source code
│   └── nur_a
│       ├── assignment_1
│       ├── assignment_2
│       └── tutorial
├── data                    # data: text based
├── media                   # data: images for tutorials and assignments
├── notes                   # lecture notes
├── output                  # output: text based, code, calculations
├── plots                   # output: plots
├── tex                     # latex files and rendered pdf
├── run.sh     
├── environment.yml
├── README.md
└── LICENSE
```
