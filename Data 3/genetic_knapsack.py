# Genetic Algorithm for 0/1 Knapsack Problem
# Main strategies used:
# - Generational replacement
# - Tournament selection for parent choice
# - Single-point crossover with CROSSOVER_RATE 0.7
# - Per-gene mutation with MUTATION_RATE 0.01
# - Repair infeasible solutions by removing low value/weight items

import os
import random
import time
import matplotlib.pyplot as plt
import pandas as pd


FOLDER_PATH = '.'
MAX_TRIES = 10
CROSSOVER_RATE =0.7
MUTATION_RATE = 0.01

# Knapsack file reading functions

def load_knapsack_file(filepath):
    """Reads a knapsack file with 'Profit Weight' header and 'Capacity' footer."""
    profits = []
    weights = []
    capacity = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("profit"):
                continue
            if line.lower().startswith("capacity"):
                capacity = int(line.split(":")[1].strip())
            else:
                parts = line.split()
                if len(parts) >= 2:
                    p, w = int(parts[0]), int(parts[1])
                    profits.append(p)
                    weights.append(w)

    if capacity is None:
        raise ValueError(f"Capacity value not found in file: {filepath}")

    return profits, weights, capacity

def load_all_instances(folder="."):
    """Loads all '_clean.txt' knapsack instances in the given folder."""
    instances = {}

    for filename in os.listdir(folder):
        if filename.endswith("_clean.txt"):
            filepath = os.path.join(folder, filename)
            try:
                profits, weights, capacity = load_knapsack_file(filepath)
                instances[filename] = {
                    "profits": profits,
                    "weights": weights,
                    "capacity": capacity,
                    "n_items": len(profits),
                }
                print(f"Loaded: {filename} ({len(profits)} items, Capacity={capacity})")
            except Exception as e:
                print(f"Error in {filename}: {e}")
    return instances

# Genetic Algorithm Components

class Item:
    def __init__(self, id, weight, value):
        self.id = id
        self.weight = weight
        self.value = value

def create_random_solution(i_list): #Creates a random binary solution (chromosome)
   return [random.randint(0, 1) for _ in i_list]

def calculate_value(i_list, chromosome): #Calculates total value and weight for a chromosome
    total_value = 0
    total_weight = 0
    for i, gene in enumerate(chromosome):
        if gene == 1:
            total_value += i_list[i].value
            total_weight += i_list[i].weight
    return total_value, total_weight

def valid_solution(i_list, chromosome, capacity): #Checks if total weight is within the capacity
    total_weight = 0
    for i in range(0, len(chromosome)):
        if chromosome[i] == 1:
            total_weight += i_list[i].weight
        if total_weight > capacity:
            return False
    return True

def repair_solution(i_list, chromosome, capacity): #Repairs infeasible solutions by removing low-value-to-weight items
    while not valid_solution(i_list, chromosome, capacity):
        ratios = [(i, i_list[i].value / i_list[i].weight) for i, g in enumerate(chromosome) if g == 1]
        if not ratios:
            break
        worst_item = min(ratios, key=lambda x: x[1])[0]
        chromosome[worst_item] = 0 #Remove item with least value
    return chromosome

def fitness(i_list, chromosome, capacity): #Calculates fitness and applies penalty if solution exceeds capacity
    total_value, total_weight = calculate_value(i_list, chromosome)
    if total_weight <= capacity:
        return total_value
    return 0 #total_value - (total_weight - capacity) * PENALTY_FACTOR

def tournament_selection(pop, i_list, capacity): #Selects best individual parents using tournament Selection
    if len(pop) <= 100:
        tournament_size = 2
    elif len(pop) <= 200:
        tournament_size = 3
    else:
        tournament_size = 4
    competitors = [pop[random.randint(0, len(pop) - 1)] for _ in range(tournament_size)] #Randomly select solutions 
    fitnesses = [fitness(i_list, c, capacity) for c in competitors]
    best_index = fitnesses.index(max(fitnesses)) #Select the best solution
    return competitors[best_index]

def crossover(p_1, p_2, i_list, capacity): #Single-point crossover to generate offspring
    if random.random() < CROSSOVER_RATE:
        for _ in range(MAX_TRIES):
            break_point = random.randint(0, len(p_1))
            child = p_1[:break_point] + p_2[break_point:]
            if valid_solution(i_list, child, capacity):
                return child
        return p_1  # fallback
    else:
        # No crossover, return parent1
        return p_1

def mutation(chromosome, i_list, capacity): #Mutates a chromosome by flipping one gene
    for _ in range(MAX_TRIES):
        temp = chromosome.copy()
        for i in range(len(temp)):
            if random.random() < MUTATION_RATE:
                temp[i] = 1 - temp[i]  # flip gene
        if valid_solution(i_list, temp, capacity):
            return temp
    return chromosome  # fallback

def initial_population(pop_size, i_list, capacity): #Generates initial population and repairs invalid individuals
    population = []
    while len(population) < pop_size:
        new_solution = create_random_solution(i_list)
        new_solution = repair_solution(i_list, new_solution, capacity)
        population.append(new_solution)
    return population

def create_generation(pop, i_list, capacity): #Creates next generation using selection, crossover, mutation
    new_gen = []
    for _ in range(len(pop)):
        #step 1: select parents
        parent_1 = tournament_selection(pop, i_list, capacity)
        parent_2 = tournament_selection(pop, i_list, capacity)

        #step 2: apply crossover
        child1 = crossover(parent_1, parent_2, i_list, capacity)
        child2 = crossover(parent_2, parent_1, i_list, capacity)
        
        #step 3: apply mutation
        child1 = mutation(child1, i_list, capacity)
        child2 = mutation(child2, i_list, capacity)

        new_gen.extend([child1, child2])
    return new_gen[:len(pop)]

def best_solution(generation, i_list): #Returns best value solution in population
    best_value, best_weight = 0, 0
    for g in generation:
        value, weight = calculate_value(i_list, g)
        if value > best_value:
            best_value, best_weight = value, weight
    return best_value, best_weight

def genetic_algorithm(capacity, pop_size, gen_size, i_list): #Executes the genetic algorithm

    population = initial_population(pop_size, i_list, capacity) #Initiate first population
    value_list = []

    for gen in range(gen_size):
        population = create_generation(population, i_list, capacity) #Create new generation

        #Save best solution of the generation
        best_val, best_weight = best_solution(population, i_list)
        value_list.append((best_val, best_weight))

    return population, value_list


def run_all_instances(folder="."):
    """Runs the GA on all knapsack instances and plots performance."""
    all_data = load_all_instances(folder)
    results = []
    
    os.makedirs("ga_plots", exist_ok=True)
    os.makedirs("ga_solutions", exist_ok=True)

    for filename, info in all_data.items():
        print(f"\nRunning GA on {filename}...")

        n_items = info["n_items"]

        #Parameter adjustment based on problem size
        if n_items <= 100:
            population_size = 100
            generation_size = 500
        elif n_items <= 200:
            population_size = 200
            generation_size = 1000
        else:  # 500
            population_size = 500
            generation_size = 2500

        i_list = [Item(i+1, info["weights"][i], info["profits"][i]) for i in range(info["n_items"])]
        capacity = info["capacity"]

        # Time tracking
        start_time = time.time()
        pop, v_list = genetic_algorithm(
            capacity=capacity,
            pop_size=population_size,
            gen_size=generation_size,
            i_list=i_list,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Best solution

        best_chrom = None
        best_value = -1
        best_weight = 0

        for chrom in pop:
            v, w = calculate_value(i_list, chrom)
            if v > best_value:
                best_value = v
                best_weight = w
                best_chrom = chrom[:]   

  
        fill_ratio = best_weight / capacity if capacity > 0 else 0
        improvements = sum(1 for i in range(1, len(v_list)) if v_list[i] > v_list[i-1])
        

        print(f"{filename}: Best = {best_value}, Weight = {best_weight}, Fill = {fill_ratio:.3f}, Time = {elapsed_time:.2f}s, Improvements = {improvements}")


        best_values_over_time = [v[0] for v in v_list]

        plt.figure(figsize=(8, 5))
        plt.plot(range(len(best_values_over_time)), best_values_over_time)
        plt.xlabel("Iteration")
        plt.ylabel("Best Profit Found")
        plt.title(f"{filename} - Progress of Best Value")
        plt.grid(True)

        plot_path = f"ga_plots/{filename.replace('.txt','')}_history.png"
        plt.savefig(plot_path)
        plt.close()

        # Save binary solution
        clean_name = filename.replace(".txt", "")
        txt_path = f"ga_solutions/{clean_name}_bestValue.txt"
        with open(txt_path, "w") as f:
            f.write("".join(str(bit) for bit in best_chrom))


        results.append({
            "file": filename,
            "items": n_items,
            "capacity": capacity,
            "best_value": best_value,
            "best_weight": best_weight,
            "fill_ratio": round(fill_ratio, 3),
            "avg_time_sec": round(elapsed_time, 4),
            "generations": generation_size,
            "num_improvements": improvements,
            "plot_path": plot_path,
            "solution_path": txt_path
        })

        # Save summary CSV
        df = pd.DataFrame(results)
        df.to_csv("ga_results.csv", index=False)
        print("\nSaved results to ga_results.csv")

    return df

# Main Execution

if __name__ == "__main__":
    print("files in the folder:")
    print(os.listdir(FOLDER_PATH))
    run_all_instances(FOLDER_PATH)
