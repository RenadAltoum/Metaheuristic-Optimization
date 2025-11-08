# sa_knapsack_batch.py
# Simulated Annealing for 0/1 Knapsack (batch over all *_clean.txt files)
# - Clear variable names (snake_case)
# - O(1) profit/weight updates on flips
# - Penalty on overweight
# - Tracks improvement history per file
# - Saves plots + CSV summary
# - Optional: save best 0/1 vector + history CSV
# - Runs directly from VS Code (no CLI needed)

import os
import math
import time
import random
import csv
import matplotlib.pyplot as plt


# ===================================
# 1) READ KNAPSACK DATA FILES
# ===================================

def load_knapsack_file(file_path):
    """
    Reads a knapsack file that starts with 'Profit Weight' and ends with 'Capacity: <value>'.
    Returns: item_profits, item_weights, knapsack_capacity
    """
    item_profits = []
    item_weights = []
    knapsack_capacity = None
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            text = line.strip()
            if not text or text.lower().startswith("profit"):
                continue
            if text.lower().startswith("capacity"):
                knapsack_capacity = int(text.split(":")[1].strip())
            else:
                profit, weight = map(int, text.split())
                item_profits.append(profit)
                item_weights.append(weight)
    
    if knapsack_capacity is None:
        raise ValueError(f"Missing capacity value in file: {file_path}")
    
    return item_profits, item_weights, knapsack_capacity


def load_all_instances(folder_path=".", file_pattern_suffix="_clean.txt"):
    """
    Loads all files that end with file_pattern_suffix from the given folder.
    Returns a dict: name -> {item_profits, item_weights, knapsack_capacity, number_of_items}
    """
    instances = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(file_pattern_suffix):
            file_path = os.path.join(folder_path, filename)
            try:
                item_profits, item_weights, knapsack_capacity = load_knapsack_file(file_path)
                instances[filename] = {
                    "item_profits": item_profits,
                    "item_weights": item_weights,
                    "knapsack_capacity": knapsack_capacity,
                    "number_of_items": len(item_profits),
                    "file_path": file_path
                }
                print(f"Loaded: {filename} ({len(item_profits)} items, Capacity={knapsack_capacity})")
            except Exception as error:
                print(f"Error in {filename}: {error}")
    return instances


# ===================================
# 2) SA HELPERS
# ===================================

def generate_random_solution(number_of_items):
    return [random.randint(0, 1) for _ in range(number_of_items)]


def calculate_solution_value_and_weight(solution, item_weights, item_profits):
    total_profit = 0
    total_weight = 0
    for selected, weight, profit in zip(solution, item_weights, item_profits):
        if selected:
            total_weight += weight
            total_profit += profit
    return total_profit, total_weight


def penalized_value(profit, weight, capacity, penalty_factor):
    return profit - penalty_factor * max(0, weight - capacity)


# ===================================
# 3) SA CORE (O(1) UPDATES + ADAPTIVE COOLING)
# ===================================

def simulated_annealing_knapsack(
    item_profits, item_weights, knapsack_capacity,
    initial_temperature=220.0,
    minimum_temperature=1e-3,
    cooling_rate=0.985,
    iterations_per_item_factor=35,
    random_seed=None,
    track_history=True,
    neighbor_k=1,
    cooling="geometric",          # "geometric" or "linear"
    early_stop_no_improve=5,      # temperature levels with no improvement
    adaptive_cooling=True,        # acceptance-rate-based adjustment (course-aligned)
    target_acceptance=0.25,       # watch acceptance probability each temperature level
    warmup_factor=1.03,           # slightly re-heat if too cold
    cooldown_factor=0.98          # slightly cool faster if too hot (optional)
):
    """
    Simulated Annealing with:
      - Penalty for capacity overflow
      - O(1) updates when flipping bits (we update profit/weight directly)
      - Optional adaptive cooling using acceptance rate
    Returns dict with best stats (+ history and best_vector if enabled).
    """
    if random_seed is not None:
        random.seed(random_seed)

    number_of_items = len(item_profits)
    penalty_factor = 10 * max(item_profits)

    # Initial random solution
    current_solution = generate_random_solution(number_of_items)

    # Keep current profit/weight in O(1)
    current_profit = sum(p for x, p in zip(current_solution, item_profits) if x)
    current_weight = sum(w for x, w in zip(current_solution, item_weights) if x)
    current_objective = penalized_value(current_profit, current_weight, knapsack_capacity, penalty_factor)

    # Best feasible
    best_feasible_solution = current_solution[:]
    best_feasible_profit = -1
    best_feasible_weight = 0

    if current_weight <= knapsack_capacity:
        best_feasible_profit = current_profit
        best_feasible_weight = current_weight

    temperature = initial_temperature
    total_iterations = 0
    levels_without_improve = 0
    start_time = time.time()

    improvement_history = []  # (iteration, best_profit)

    while temperature > minimum_temperature:
        iteration_limit = number_of_items * iterations_per_item_factor
        improved_this_level = False
        accepted_moves = 0

        for _ in range(iteration_limit):
            total_iterations += 1

            # Flip k distinct bits
            flip_indices = random.sample(range(number_of_items), k=min(neighbor_k, number_of_items))

            # Incremental update for profit/weight
            new_profit = current_profit
            new_weight = current_weight
            for i in flip_indices:
                if current_solution[i] == 1:
                    new_profit -= item_profits[i]
                    new_weight -= item_weights[i]
                else:
                    new_profit += item_profits[i]
                    new_weight += item_weights[i]

            new_objective = penalized_value(new_profit, new_weight, knapsack_capacity, penalty_factor)
            delta = new_objective - current_objective

            # Metropolis acceptance
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-12)):
                # Accept neighbor
                for i in flip_indices:
                    current_solution[i] = 1 - current_solution[i]
                current_profit = new_profit
                current_weight = new_weight
                current_objective = new_objective
                accepted_moves += 1

                # Update best feasible
                if new_weight <= knapsack_capacity and new_profit > best_feasible_profit:
                    best_feasible_profit = new_profit
                    best_feasible_weight = new_weight
                    best_feasible_solution = current_solution[:]
                    improved_this_level = True
                    if track_history:
                        improvement_history.append((total_iterations, best_feasible_profit))

        # Early stopping by temperature levels with no improvement
        if early_stop_no_improve > 0:
            levels_without_improve = 0 if improved_this_level else (levels_without_improve + 1)
            if levels_without_improve >= early_stop_no_improve:
                break

        # Acceptance-rate based adaptive tweak (course-aligned idea)
        if adaptive_cooling and iteration_limit > 0:
            acceptance_rate = accepted_moves / iteration_limit
            if acceptance_rate < (0.6 * target_acceptance):
                temperature *= warmup_factor      # slightly re-heat if too cold
            elif acceptance_rate > (1.6 * target_acceptance):
                temperature *= cooldown_factor    # slightly cool faster if too hot

        # Base cooling schedule
        if cooling == "geometric":
            temperature *= cooling_rate
        else:  # linear
            step = (initial_temperature - minimum_temperature) * (1.0 - cooling_rate)
            temperature -= step
            if temperature <= 0:
                break

    elapsed = round(time.time() - start_time, 4)

    return {
        "best_profit": best_feasible_profit,
        "best_weight": best_feasible_weight,
        "capacity": knapsack_capacity,
        "time_seconds": elapsed,
        "iterations": total_iterations,
        "penalty_factor": penalty_factor,
        "best_vector": best_feasible_solution,
        "profit_improvement_log": improvement_history if track_history else None
    }


# ===================================
# 4) PLOTTING FUNCTION
# ===================================

def plot_improvement_curve(history_data, output_path, chart_title):
    """Creates and saves a plot showing how the best solution improved over time."""
    if not history_data:
        # still save placeholder
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


# ===================================
# 5) RUN SA ON ALL FILES AND PLOT RESULTS
# ===================================

def run_batch_simulated_annealing(
    folder_path=".",
    file_pattern_suffix="_clean.txt",
    initial_temperature=220.0,
    minimum_temperature=1e-3,
    cooling_rate=0.985,
    iterations_per_item_factor=35,
    runs_per_file=1,
    base_random_seed=42,
    cooling="geometric",
    neighbor_k=1,
    adaptive_cooling=True,
    target_acceptance=0.25,
    warmup_factor=1.03,
    cooldown_factor=0.98,
    early_stop_no_improve=5,
    save_csv=True,
    csv_output_path="sa_results.csv",
    plots_folder="sa_plots",
    save_vectors=True,
    save_history_csv=True
):
    os.makedirs(plots_folder, exist_ok=True)
    instances = load_all_instances(folder_path, file_pattern_suffix)

    if not instances:
        print("No *_clean.txt files found.")
        return []

    results_summary = []

    print(f"\nRunning SA on {len(instances)} file(s)...\n")
    for dataset_file, info in instances.items():
        item_profits = info["item_profits"]
        item_weights = info["item_weights"]
        knapsack_capacity = info["knapsack_capacity"]
        number_of_items = info["number_of_items"]

        best_result = None
        execution_times = []
        last_history = []

        for r in range(runs_per_file):
            seed = base_random_seed + r
            result = simulated_annealing_knapsack(
                item_profits=item_profits,
                item_weights=item_weights,
                knapsack_capacity=knapsack_capacity,
                initial_temperature=initial_temperature,
                minimum_temperature=minimum_temperature,
                cooling_rate=cooling_rate,
                iterations_per_item_factor=iterations_per_item_factor,
                random_seed=seed,
                track_history=True,
                neighbor_k=neighbor_k,
                cooling=cooling,
                early_stop_no_improve=early_stop_no_improve,
                adaptive_cooling=adaptive_cooling,
                target_acceptance=target_acceptance,
                warmup_factor=warmup_factor,
                cooldown_factor=cooldown_factor
            )
            execution_times.append(result["time_seconds"])
            last_history = result.get("profit_improvement_log", []) or last_history
            if best_result is None or result["best_profit"] > best_result["best_profit"]:
                best_result = result

        # Plot curve
        plot_title = f"{dataset_file} | n={number_of_items} | Cap={knapsack_capacity}"
        png_name = dataset_file.replace(".txt", "_history.png")
        png_path = os.path.join(plots_folder, png_name)
        plot_improvement_curve(last_history, png_path, plot_title)

        # Save history CSV (optional)
        if save_history_csv and last_history:
            hist_csv = os.path.join(plots_folder, dataset_file.replace(".txt", "_history.csv"))
            with open(hist_csv, "w", newline="", encoding="utf-8") as hf:
                writer = csv.writer(hf)
                writer.writerow(["iteration", "best_profit"])
                writer.writerows(last_history)

        # Save best vector (optional)
        if save_vectors and "best_vector" in best_result:
            vec_path = os.path.join(plots_folder, dataset_file.replace(".txt", "_best_vector.txt"))
            with open(vec_path, "w", encoding="utf-8") as vf:
                vf.write(" ".join(map(str, best_result["best_vector"])))

        avg_time = sum(execution_times) / len(execution_times)
        num_improvements = len(last_history)
        fill_ratio = (
            best_result["best_weight"] / best_result["capacity"]
            if best_result["capacity"] else 0.0
        )

        results_summary.append({
            "file": dataset_file,
            "items": number_of_items,
            "capacity": knapsack_capacity,
            "best_profit": best_result["best_profit"],
            "best_weight": best_result["best_weight"],
            "fill_ratio": round(fill_ratio, 4),
            "avg_time_sec": round(avg_time, 4),
            "iterations": best_result["iterations"],
            "num_improvements": num_improvements,
            "plot_path": png_path
        })

        print(f"{dataset_file:30s} | items={number_of_items:4d} | cap={knapsack_capacity:4d} | "
              f"best_profit={best_result['best_profit']:6d} | best_weight={best_result['best_weight']:6d} | "
              f"avg_time={avg_time:.3f}s")

    if save_csv and results_summary:
        with open(csv_output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(results_summary[0].keys()))
            writer.writeheader()
            writer.writerows(results_summary)
        print(f"\nResults saved to: {csv_output_path}")

    return results_summary


# ===================================
# 6) MAIN (READY FOR VS CODE ▶️)
# ===================================

if __name__ == "__main__":
    # اختاري القيم هنا (بدون سطر أوامر)
    run_batch_simulated_annealing(
        folder_path=".",                # مكان ملفات *_clean.txt
        file_pattern_suffix="_clean.txt",
        # إعدادات جودة عالية (غيريها حسب رغبتك):
        initial_temperature=240.0,
        minimum_temperature=1e-3,
        cooling_rate=0.987,             # تبريد بطيء نسبيًا للجودة
        iterations_per_item_factor=30,  # محاولات لكل حرارة = n * هذا الرقم
        runs_per_file=1,
        base_random_seed=42,
        cooling="geometric",
        neighbor_k=1,
        # تفعيل تبريد تكيّفي مبني على معدل القبول (من فكرة الدرس):
        adaptive_cooling=True,
        target_acceptance=0.25,
        warmup_factor=1.03,
        cooldown_factor=0.98,
        early_stop_no_improve=5,        # وقفي لو 5 مستويات حرارة بدون تحسّن
        # الحفظ:
        save_csv=True,
        csv_output_path="sa_results.csv",
        plots_folder="sa_plots",
        save_vectors=True,              # يحفظ أفضل متجه 0/1
        save_history_csv=True           # يحفظ (iteration,best_profit)
    )