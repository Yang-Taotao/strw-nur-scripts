# strw-nur-scripts

This is the documentation for [**_Numerical Recipes in Astrophysics A_**](https://studiegids.universiteitleiden.nl/courses/130599/numerical-recipes-in-astrophysics-a) and [**_Numerical Recipes in Astrophysics B_**](https://studiegids.universiteitleiden.nl/courses/130600/numerical-recipes-in-astrophysics-b) at [**Leiden Observatory**](https://local.strw.leidenuniv.nl/).

**Dated:** 2026-February-10

## notes

- This is a collection of lecture notes and scripts used during the courses
- Containing scripts used for tutorial problems and assignments

## additional

- scripts are written and tested under `conda` environment as specified through `environment.yml`
- scripts can also be ran natively with base `python3` through shell script `run_*.sh`

## usage

Preferred OS

![Linux](https://img.shields.io/badge/-Linux-4c566a?logo=linux?link=https://github.com/user/repo)

Star by cloning the repository to your device:

```sh
git clone https://github.com/Yang-Taotao/strw-nur-scripts.git
cd strw-nur-scripts/
```

Example usage for generating results for `nur_a_handin_1`:

```sh
chmod +x ./run_assignment_1.sh
./run_assignment_1.sh
```

See `file structure` below for details.

## file structure

```sh
.
├── src                     # source code
│   └── nur_a
│       ├── assignment_1
│       └── tutorial
├── data                    # data: text based
├── media                   # data: images for tutorials and assignments
├── notes                   # lecture notes
├── output                  # output: text based
├── plots                   # output: plots
├── tex                     # latex files and rendered pdf
├── run_assignment_1.sh     
├── environment.yml
├── README.md
└── LICENSE
```
