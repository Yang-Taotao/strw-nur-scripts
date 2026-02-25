#!/bin/bash

# If you get a permission denied error for run.sh itself, run this line in the terminal:
chmod +x ./run_assignment_1.sh

# If you get errors about weird (return) characters,
# and/or you edited run.sh on Windows, run this command:
dos2unix ./run_assignment_1.sh &>/dev/null

# Make sure you do NOT run in a virtual environment (e.g. conda, uv), 
# or your results may be different than when we run your code
# On the STRW computers, you may need to run "module purge" 
# if you load any modules at startup
echo "================================================================================="
pythonversion="$(python3 --version | cut -d' ' -f2)"
if [ "${pythonversion}" != "3.9.25" ]; then
	echo "WARNING: python version ${pythonversion} != default vdesk one (v3.9.25)"
fi
matplotlibversion="$(python3 -m pip list | grep "matplotlib " | tr -s ' ' | cut -d' ' -f2)"
if [ "${matplotlibversion}" != "3.9.0" ]; then
	echo "WARNING: matplotlib version ${matplotlibversion} != default vdesk one (v3.9.0)"
fi
numpyversion="$(python3 -m pip list | grep "numpy " | tr -s ' ' | cut -d' ' -f2)"
if [ "${numpyversion}" != "1.26.4" ]; then
	echo "WARNING: numpy version ${numpyversion} != default vdesk one (v1.26.4)"black
fi

# Check if black formatter is installed
if ! python3 -m black --version &>/dev/null; then
	echo "INFO: black not found. Installing via pip ...."
	python3 -m pip install black
fi

# Format all python files 
# (note that this assumes your python files are all in the same directory as run.sh)
echo "================================================================================="
echo "INFO: run $(black) formatter on code base ...."
python3 -m black .

echo "================================================================================="
echo "INFO: initializing new ./data   directory.."
mkdir -p ./data
rm -rf ./data/*

echo "INFO: initializing new ./plots  directory.."
mkdir -p ./plots
rm -rf ./plots/*

echo "INFO: initializing new ./output directory.."
mkdir -p ./output
rm -rf ./output/*

echo "================================================================================="
echo "INFO: data downloading to ./data/vandermonde.txt"
wget -q -O ./data/vandermonde.txt "https://home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt"

echo "================================================================================="
echo "INFO: running scripts to solve assignment1-q1..."
python3 ./src/nur_a/assignment_1/q1.py

# Copy the code to a text file which will be shown in the PDF
cat ./src/nur_a/assignment_1/q1.py >./output/a1q1_poisson_code.txt

echo "INFO: running scripts to solve assignment1-q2..."
python3 ./src/nur_a/assignment_1/q2.py

# Copy the code to a text file which will be shown in the PDF
cat ./src/nur_a/assignment_1/q2.py >./output/a1q2_vandermonde_all_code.txt

echo "================================================================================="
echo "INFO: compiling tex via pdflatex..."
pdflatex -interaction=batchmode "$(pwd)/tex/nur_a_handin_1.tex"
# Run a second time to fix links/references
pdflatex -interaction=batchmode "$(pwd)/tex/nur_a_handin_1.tex" &>/dev/null 

echo "INFO: purging temp tex comp file..."
rm -rf ./*.aux ./*.log ./*.out
rm -rf ./tex/%OUTDIR%

echo "================================================================================="
echo "INFO: run_assignment_1.sh completed!"
echo "WARNING: don't forget to hand in a *clean* version of this directory"
echo "         (remove downloaded data, the plots and output directories)."
echo "================================================================================="
