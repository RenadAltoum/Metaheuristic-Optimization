# Metaheuristic-Optimization
Knapsack Problem Using Genetic Algorithm (GA) and Simulated Annealing (SA)
## Project Description
Our project aims to solve the 0/1 Knapsack Problem (KP) using two metaheuristic optimization algorithms:
(GA),
(SA)
The goal is to select items that maximize the total profit without exceeding the weight capacity of the knapsack.
Both algorithms are implemented, tested on multiple datasets, and compared in terms of solution quality, execution time, and convergence behavior.
## Input Data Format
Each dataset file follows this structure: -First line: Profit Weight, 
-Next lines : Each line contains two numbers → profit and weight for one item,
-Last line: Capacity: <value>

## data file names
knapPI_1_100_1000_1_clean.txt
knapPI_2_200_1000_1_clean.txt
knapPI_3_500_1000_1_clean.txt

These files contain data for 100, 200, and 500 items respectively.
Each dataset has its own initial temperature and cooling rate tuned for best performance.

## Requirements
Make sure you have the following installed before running the project:

Python 3.8+

Libraries:

Built-in: os, math, time, random, csv

External: matplotlib

To install matplotlib: pip install matplotlib


## How to Run the Project

1.Place all scripts and data files in the same folder.

2.Open the folder in Visual Studio Code.

3.Open the file sa_knapsack_batch.py.

4.Press Run  (or run in terminal with): python sa_knapsack_batch.py

5.The default parameters at the end of the file are already set correctly,
but you can modify them in the function: run_batch_simulated_annealing(...)
## What Happens During Execution
When you run the program, it performs these steps automatically:

1.Reads the input dataset (e.g., 100, 200, 500 items).

2.Runs the Simulated Annealing algorithm with pre-defined parameters.

3.Displays in the terminal: Number of items ,Best profit ,Best weight ,Execution time.

Example output: 
Items: 100
Best Profit: 2594
Best Weight: 998
Time: 2.14 seconds

4. Generates the result files and visualizations automatically.


## Team Members
-Renad Abdullah Altoum
-Rahaf Mussed Almutairi
-Sara Faiz Babgi 
-Ohood Mohammad Al-Magedi 
-Munira Mohammad Alangari
