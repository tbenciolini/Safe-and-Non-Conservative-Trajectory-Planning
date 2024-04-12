# Safe and Non-Conservative Trajectory Planning for Autonomous Driving Handling Unanticipated Behaviors of Traffic Participants

## Description

This repository contains the code to reproduce the simulations of the manuscript "Safe and Non-Conservative Trajectory Planning for Autonomous Driving Handling Unanticipated Behaviors of Traffic Participants" by Tommaso Benciolini, Michael Fink, Nehir Güzelkaya, Dirk Wollherr, Marion Leibold.

## Usage

### Requirements

Run `pip install -r requirements.txt` to intall required libraries.


### Run files

Change in the main.py file the Start-up section depending on your goals and run the main script with python3.  
The file plot_all.py plots stored data.  
With plots_for_paper.py plots in pdf and png format of defined time steps are generated.  
The file print_statistics_multi_runs.py computes the mean and standard deviation of the run time for multiple runs.  

### Parameters
You can find the used parameters and settings in the following files:

main.py
line 21: MPC Horizon  
line 22: Flag, iterative recomputation of cosy  
line 24: Show plot after run  
line 25-26: Number of Simulations to evaluate statistical properties of the computation time.  
line 30-31: Select the file name of the scenario  
line 63-64: Linear dynamics of the obstacles  
line 81-85: Initialize controller, set reference trajectory and weight matrices for MPC controller  

controller.py  
line 7-8: Use debug plot, plot for each predictet time step with feasible area.  
line 9-10: use plot, Plot of the current situation  

ev.py  
line 12: Maximum velocity  
line 13: Minimum velocity  
line 14: Maximum acceleration  
line 15: Minimum acceleration  
line 16: Maximum change in acceleration  
