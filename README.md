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

## Output Files
After running, the program creates:

1. sa_results.csv
   Contains summary of all datasets:

File name

Number of items

Capacity

Best profit

Best weight

Fill ratio

Average runtime

Iterations

Number of improvements

Path to plot image

2. sa_plots/ folder 
   For each dataset, this folder contains:

*_history.png → Graph of iterations vs. best profit (improvement over time).

*_history.csv → Same data used for plotting.

*_best_vector.txt → Binary vector (0/1) showing selected items in the best solution.



## Recommended SA Parameters

| Dataset Size | initial_temperature | cooling_rate | iterations_per_item_factor | early_stop_no_improve | adaptive_cooling |
|---------------|---------------------|---------------|-----------------------------|-----------------------|------------------|
| 100 items | 220 | 0.985 | 35 | 5 | yes |
| 200 items | 240 | 0.987 | 30 | 5 | yes |
| 500 items | 260 | 0.989 | 25 | 6 | yes |

## Result Metrics
Each run reports:

best_profit: Highest profit for a feasible solution.

best_weight: Weight of the best solution.

fill_ratio: best_weight / capacity.

avg_time_sec: Average runtime in seconds.

iterations: Total number of iterations.

num_improvements: How many times the best solution improved.

plot_path: File path of the improvement graph.

## Genetic Algorithm (GA)
A second part of the project implements the Genetic Algorithm (GA) to solve the same Knapsack datasets.
GA uses:

Population-based search

Selection, Crossover, and Mutation operators

Multiple generations to evolve better solutions

After running GA:

The program prints best profit, best weight, and average runtime.

Results can be compared directly with SA to evaluate algorithm performance.

## Comparison Between SA and GA  

| Feature / Metric | Simulated Annealing (SA) | Genetic Algorithm (GA) |
|------------------|---------------------------|--------------------------|
| **Type** | Single-solution (cooling-based) | Population-based |
| **Search Strategy** | Explores one solution and gradually cools down | Evolves a population of solutions using selection, crossover, and mutation |
| **Key Parameters** | Initial temperature, cooling rate, iterations per level | Population size, crossover rate, mutation rate |
| **Exploration** | Local search with occasional random jumps | Global exploration across multiple candidates |
| **Convergence Speed** | Faster convergence | Slower but more diverse search |
| **Computation Cost** | Low (focuses on one solution) | Higher (multiple solutions per generation) |
| **Best Use Case** | When we need a quick, near-optimal solution | When we need more exploration and global optimization |
| **Output** | Best single solution + improvement plot | Best evolved solution after multiple generations |
| **Strengths** | Simple, efficient, adaptable | Strong global search, avoids local minima |
| **Weaknesses** | May get stuck in local optima | Requires tuning and longer runtime |

## Team Members
-Renad Abdullah Altoum

-Rahaf Mussed Almutairi

-Sara Faiz Babgi 

-Ohood Mohammad Al-Magedi 

-Munira Mohammad Alangari
