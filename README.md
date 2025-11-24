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

knapPI_1_200_1000_1_clean.txt

knapPI_1_500_1000_1_clean.txt

knapPI_2_100_1000_1_clean.txt

knapPI_2_200_1000_1_clean.txt

knapPI_2_500_1000_1_clean.txt

These files contain data for 100, 200, and 500 items respectively.


## Requirements
Make sure you have the following installed before running the project:

Python 3.8+

### Libraries:

Built-in:
- os
- math
- time
- random
- csv

External: - matplotlib pandas

Install with:

    pip install matplotlib pandas




## How to Run the Project


### Simulated Annealing (SA)

1. Place all scripts and data files in the same folder.  
2. Open the folder in **Visual Studio Code**.  
3. Open `sa_knapsack_batch.py`.  
4. Press **Run** (or run in terminal with):  
   ```bash
   python sa_knapsack_batch.py
5.  Default parameters are set inside the function:\
    `run_batch_simulated_annealing(...)`

### Genetic Algorithm (GA)

1. Place all scripts and data files in the same folder.  
2. Open the folder in **Visual Studio Code**.  
3. Open `genetic_knapsack.py`.  
4. Press **Run** (or run in terminal with):  
   ```bash
    python ga_knapsack.py
5.  Default parameters are set inside the function:\
    ` run_all_instances(...)`
    
## What Happens During Execution
When you run the program, it performs these steps automatically:

1.Reads the input dataset (e.g., 100, 200, 500 items).

2.Runs the Simulated Annealing algorithm with pre-defined parameters.

3.Displays in the terminal: Number of items ,Best profit ,Best weight ,Execution time.

Example output: 

Items: 100

Best Profit: 9049

Best Weight: 999

Time: 0.060 seconds

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

## Simulated Annealing SA
A main part of the project implements the Simulated Annealing (SA) algorithm to solve the same Knapsack datasets.

SA uses:

  - Single-solution local search
  - Temperature-based exploration
  - Cooling schedule (decreasing temperature)
  - Random neighbor generation
  - Probabilistic acceptance of worse solutions (to escape local minima)

After running SA, the program prints:

  - Best profit
  - Best weight
  - Average runtime
  - Number of improvements
  - Iterations performed
  - The script also generates:
  - A progress plot showing how the solution improved over time
  - A CSV file containing improvement history
  - A binary best-vector file (0/1) indicating selected items

These results can be directly compared with GA to evaluate performance, convergence behavior, and runtime efficiency. 

## Recommended SA Parameters
Each dataset has its own initial temperature and cooling rate tuned for best performance.

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
A second part of the project implements the Genetic Algorithm (GA) to
solve the same datasets.

GA uses:
- Population-based search
- Selection
- Crossover
- Mutation
- Multi-generation evolution

After running GA, the program prints:
- Best profit
- Best weight
- Average runtime

These results can be directly compared with SA.
## Recommended GA Parameters

| Dataset Size | Population Size | Crossover Rate | Mutation Rate | Generations | Elitism | Selection Method |
|--------------|----------------|----------------|---------------|-------------|---------|-----------------|
| 100 items    | 100            | 0.7            | 0.01          | 500         | yes     | tournament      |
| 200 items    | 200            | 0.7            | 0.01          | 1000        | yes     | tournament      |
| 500 items    | 500            | 0.7            | 0.01          | 2500        | yes     | tournament      |

## GA Fitness Function

The fitness function used in this Genetic Algorithm is an **objective-based maximization fitness**.  
It evaluates each candidate solution (chromosome) based on the **total profit** of the selected items.  

- If the total weight of selected items does **not exceed the knapsack capacity**, the fitness equals the total profit.  
- If the solution exceeds the capacity, it is considered **invalid** and assigned a fitness of `0`.  

This approach ensures that the algorithm favors solutions that maximize profit while strictly respecting the knapsack's weight constraint.


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
