# ===================================================================
# Simulated Annealing for 0/1 Knapsack Problem (Batch Processing)
# Clean version with only the features actually used
# Fully commented for project submission
# ===================================================================

import os
import math
import time
import random
import csv
import matplotlib.pyplot as plt


# ============================================
# 1) READ KNAPSACK DATA FILES
# ============================================

def load_knapsack_file(file_path):
    """
    Reads a knapsack dataset file.
    Each file starts with a header "Profit Weight"
    and ends with "Capacity: <value>".
    Returns: item profits list, item weights list, knapsack capacity.
    """
    item_profits = []
    item_weights = []
    knapsack_capacity = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            text = line.strip()

            # Skip empty lines or header
            if not text or text.lower().startswith("profit"):
                continue

            # Extract capacity
            if text.lower().startswith("capacity"):
                knapsack_capacity = int(text.split(":")[1].strip())
            
            # Extract profit/weight values
            else:
                profit, weight = map(int, text.split())
                item_profits.append(profit)
                item_weights.append(weight)

    # Error if capacity not found
    if knapsack_capacity is None:
        raise ValueError(f"Missing capacity value in file: {file_path}")

    return item_profits, item_weights, knapsack_capacity


def load_all_instances(folder_path=".", file_pattern_suffix="_clean.txt"):
    """
    Loads all dataset files ending with *_clean.txt.
    Returns dictionary with all problem instances.
    """

    instances = {}

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(file_pattern_suffix):
            file_path = os.path.join(folder_path, filename)

            try:
                profits, weights, capacity = load_knapsack_file(file_path)

                instances[filename] = {
                    "item_profits": profits,
                    "item_weights": weights,
                    "knapsack_capacity": capacity,
                    "number_of_items": len(profits),
                    "file_path": file_path
                }

                print(f"Loaded: {filename} ({len(profits)} items, Capacity={capacity})")

            except Exception as error:
                print(f"Error in {filename}: {error}")

    return instances


# ============================================
# 2) SA HELPERS
# ============================================

def generate_random_solution(number_of_items):
    """Generates a random 0/1 solution for the knapsack."""
    return [random.randint(0, 1) for _ in range(number_of_items)]


def penalized_value(profit, weight, capacity, penalty_factor):
    """Penalty function to discourage overweight solutions."""
    return profit - penalty_factor * max(0, weight - capacity)


# ============================================
# 3) CORE SIMULATED ANNEALING ALGORITHM
# ============================================

def simulated_annealing_knapsack(
    item_profits,
    item_weights,
    knapsack_capacity,
    initial_temperature=220.0,
    minimum_temperature=1e-3,
    cooling_rate=0.985,
    iterations_per_item_factor=35,
    random_seed=None
):
    """
    Simulated Annealing for 0/1 Knapsack.
    - Uses O(1) profit/weight update on bit flip.
    - Uses standard geometric cooling.
    - Tracks improvement history for plotting.
    """

    # Fix random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    number_of_items = len(item_profits)

    # Penalty factor to keep overweight solutions temporary only
    penalty_factor = 10 * max(item_profits)

    # Generate initial random solution
    current_solution = generate_random_solution(number_of_items)

    # Compute initial total profit and weight
    current_profit = sum(p for x, p in zip(current_solution, item_profits) if x)
    current_weight = sum(w for x, w in zip(current_solution, item_weights) if x)

    # Evaluate initial objective value
    current_objective = penalized_value(current_profit, current_weight, knapsack_capacity, penalty_factor)

    # Track the best FEASIBLE solution found
    best_feasible_solution = current_solution[:]
    best_feasible_profit = -1
    best_feasible_weight = 0

    if current_weight <= knapsack_capacity:
        best_feasible_profit = current_profit
        best_feasible_weight = current_weight

    temperature = initial_temperature
    total_iterations = 0
    start_time = time.time()

    # History of improvements (iteration, best_profit)
    improvement_history = []

    # ----------------------
    # Annealing Loop
    # ----------------------
    while temperature > minimum_temperature:

        # Number of moves per temperature level
        iteration_limit = number_of_items * iterations_per_item_factor

        for _ in range(iteration_limit):
            total_iterations += 1

            # Pick one random bit to flip
            i = random.randrange(number_of_items)

            # O(1) update of profit & weight
            if current_solution[i] == 1:
                new_profit = current_profit - item_profits[i]
                new_weight = current_weight - item_weights[i]
            else:
                new_profit = current_profit + item_profits[i]
                new_weight = current_weight + item_weights[i]

            # Compute new penalized objective
            new_objective = penalized_value(new_profit, new_weight, knapsack_capacity, penalty_factor)
            delta = new_objective - current_objective

            # Metropolis acceptance rule
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-12)):
                # Accept the new solution
                current_solution[i] = 1 - current_solution[i]
                current_profit = new_profit
                current_weight = new_weight
                current_objective = new_objective

                # Update best FEASIBLE solution
                if new_weight <= knapsack_capacity and new_profit > best_feasible_profit:
                    best_feasible_profit = new_profit
                    best_feasible_weight = new_weight
                    best_feasible_solution = current_solution[:]
                    improvement_history.append((total_iterations, best_feasible_profit))

        # Geometric cooling
        temperature *= cooling_rate

    # Total time
    elapsed = round(time.time() - start_time, 4)

    return {
        "best_profit": best_feasible_profit,
        "best_weight": best_feasible_weight,
        "capacity": knapsack_capacity,
        "time_seconds": elapsed,
        "iterations": total_iterations,
        "penalty_factor": penalty_factor,
        "best_vector": best_feasible_solution,
        "profit_improvement_log": improvement_history
    }


# ============================================
# 4) PLOTTING IMPROVEMENT CURVES
# ============================================

def plot_improvement_curve(history_data, output_path, chart_title):
    """
    Plots best-profit vs iterations.
    If no improvements occurred, saves an empty placeholder plot.
    """

    if not history_data:
        plt.figure(figsize=(8, 5))
        plt.title(chart_title + " (no improvements tracked)")
        plt.xlabel("Iterations")
        plt.ylabel("Best Profit Found")
        plt.grid(True)
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close()
        return

    iterations = [entry[0] for entry in history_data]
    profits = [entry[1] for entry in history_data]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, profits)
    plt.xlabel("Iterations")
    plt.ylabel("Best Profit Found")
    plt.title(chart_title)
    plt.grid(True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


# ============================================
# 5) RUN SA FOR ALL DATASETS
# ============================================

def run_batch_simulated_annealing(
    folder_path=".",
    file_pattern_suffix="_clean.txt",
    initial_temperature=220.0,
    minimum_temperature=1e-3,
    cooling_rate=0.985,
    iterations_per_item_factor=35,
    runs_per_file=1,
    base_random_seed=42,
    save_csv=True,
    csv_output_path="sa_results.csv",
    plots_folder="sa_plots",
    save_vectors=True,
    save_history_csv=True
):
    """
    Runs SA on all *_clean.txt files in the folder.
    Saves:
    - Plots
    - CSV summary
    - Best 0/1 vector
    - Improvement history CSV
    """

    os.makedirs(plots_folder, exist_ok=True)
    instances = load_all_instances(folder_path, file_pattern_suffix)

    if not instances:
        print("No *_clean.txt files found.")
        return []

    results_summary = []

    print(f"\nRunning SA on {len(instances)} dataset(s)...\n")

    for dataset_file, info in instances.items():

        profits = info["item_profits"]
        weights = info["item_weights"]
        capacity = info["knapsack_capacity"]
        n_items = info["number_of_items"]

        best_result = None
        execution_times = []
        last_history = []

        # Run SA multiple times and pick the best
        for r in range(runs_per_file):
            seed = base_random_seed + r

            result = simulated_annealing_knapsack(
                item_profits=profits,
                item_weights=weights,
                knapsack_capacity=capacity,
                initial_temperature=initial_temperature,
                minimum_temperature=minimum_temperature,
                cooling_rate=cooling_rate,
                iterations_per_item_factor=iterations_per_item_factor,
                random_seed=seed
            )

            execution_times.append(result["time_seconds"])
            last_history = result["profit_improvement_log"]
            if best_result is None or result["best_profit"] > best_result["best_profit"]:
                best_result = result

        # PLOT
        plot_title = f"{dataset_file} | n={n_items} | Cap={capacity}"
        png_path = os.path.join(plots_folder, dataset_file.replace(".txt", "_history.png"))
        plot_improvement_curve(last_history, png_path, plot_title)

        # Save improvement history
        if save_history_csv and last_history:
            hist_csv = os.path.join(plots_folder, dataset_file.replace(".txt", "_history.csv"))
            with open(hist_csv, "w", newline="", encoding="utf-8") as hf:
                writer = csv.writer(hf)
                writer.writerow(["iteration", "best_profit"])
                writer.writerows(last_history)

        # Save best vector
        if save_vectors:
            vec_path = os.path.join(plots_folder, dataset_file.replace(".txt", "_best_vector.txt"))
            with open(vec_path, "w", encoding="utf-8") as vf:
                vf.write(" ".join(map(str, best_result["best_vector"])))

        avg_time = sum(execution_times) / len(execution_times)
        num_improvements = len(last_history)
        fill_ratio = best_result["best_weight"] / capacity if capacity else 0.0

        results_summary.append({
            "file": dataset_file,
            "items": n_items,
            "capacity": capacity,
            "best_profit": best_result["best_profit"],
            "best_weight": best_result["best_weight"],
            "fill_ratio": round(fill_ratio, 4),
            "avg_time_sec": round(avg_time, 4),
            "iterations": best_result["iterations"],
            "num_improvements": num_improvements,
            "plot_path": png_path
        })

        print(f"{dataset_file:30s} | items={n_items:4d} | cap={capacity:4d} | "
              f"best_profit={best_result['best_profit']:6d} | best_weight={best_result['best_weight']:6d} | "
              f"avg_time={avg_time:.3f}s")

    # Save summary CSV
    if save_csv and results_summary:
        with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(results_summary[0].keys()))
            writer.writeheader()
            writer.writerows(results_summary)

        print(f"\nResults saved to: {csv_output_path}")

    return results_summary


# ============================================
# 6) MAIN (READY TO RUN)
# ============================================

if __name__ == "__main__":
    # Set your parameters here (no command-line needed)
    run_batch_simulated_annealing(
        folder_path=".",                  # Folder containing *_clean.txt files
        file_pattern_suffix="_clean.txt",
        
        # High-quality settings (modify as desired):
        initial_temperature=240.0,
        minimum_temperature=1e-3,
        cooling_rate=0.987,              # Relatively slow cooling for better quality
        iterations_per_item_factor=30,   # Attempts per temperature = n * this value
        
        runs_per_file=1,
        base_random_seed=42,

        # Saving options:
        save_csv=True,
        csv_output_path="sa_results.csv",
        plots_folder="sa_plots",
        save_vectors=True,               # Save the best 0/1 solution vector
        save_history_csv=True            # Save (iteration, best_profit) history
    )
