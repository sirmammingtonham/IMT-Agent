# IMT-Agent
Integrated Multi-Task Reinforcement Learning Agent Architecture

## Installation

1. `git submodule update --init --recursive`
   * pull the go game repository

2. `conda env create -f environment.yml`
	* installs all the required packages

3. `conda activate IMT-Agent`
	* activate the conda environment

4. `pip install GymGo/.`
	* install the gym_go environment

## Running

1. `python src/go.py`
	* will run the imt agent (black) against a standard q-table (white)
	* to change this, modify `src/go.py` line 91