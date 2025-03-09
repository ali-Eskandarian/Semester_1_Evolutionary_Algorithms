import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt


def plot_survival_methods_histograms(opt_gen_dict, num_q):
    plt.figure(figsize=(10, 6))

    # Define colors for each survival method
    colors = plt.cm.viridis(np.linspace(0, 1, len(opt_gen_dict)))

    for (survive_method, opt_gen_list), color in zip(opt_gen_dict.items(), colors):
        plt.hist(opt_gen_list, bins=30, color=color, alpha=0.5, edgecolor='black', label=survive_method)

    plt.xlabel("Number of Generations")
    plt.ylabel("Frequency")
    plt.title(f"Comparison of Optimal Generations Across Survival Methods - Num Queens: {num_q}")
    plt.legend()

    # Adjust layout to prevent clipping of title and labels
    plt.tight_layout()

    plt.savefig(f"plots/comparison_histograms_num_queen({num_q}).png")
    plt.close()


def plot_nqueens_histograms(opt_gen_dict):
    plt.figure(figsize=(10, 6))

    # Define colors for each N-Queens configuration
    colors = plt.cm.viridis(np.linspace(0, 1, len(opt_gen_dict)))

    for (num_q, opt_gen_list), color in zip(opt_gen_dict.items(), colors):
        plt.hist(opt_gen_list, bins=30, color=color, alpha=0.5, edgecolor='black', label=f'N={num_q}')

    plt.xlabel("Number of Generations")
    plt.ylabel("Frequency")
    plt.title("Comparison of Optimal Generations Across N-Queens Configurations")
    plt.legend()

    # Adjust layout to prevent clipping of title and labels
    plt.tight_layout()

    plt.savefig(f"plots/comparison_histograms_nqueens.png")
    plt.close()
def plot_multiple_histograms(opt_gen_lists, mutation_probs, survive_method, num_q):
    plt.figure(figsize=(10, 6))

    # Define colors for each mutation probability
    colors = plt.cm.viridis(np.linspace(0, 1, len(mutation_probs)))

    for opt_gen_list, p_m, color in zip(opt_gen_lists, mutation_probs, colors):
        plt.hist(opt_gen_list, bins=30, color=color, alpha=0.5, edgecolor='black', label=f'p_m={p_m}')

    plt.xlabel("Number of Generations")
    plt.ylabel("Frequency")
    plt.title(f"Comparison of Optimal Generations - {survive_method} Selection - Num Queens: {num_q}")
    plt.legend()

    # Adjust layout to prevent clipping of title and labels
    plt.tight_layout()

    plt.savefig(f"plots/comparison_histograms_{survive_method}_num_queen({num_q}).png")
    plt.close()

def plot_average_and_variance(results_list, max_len, runs, method):
    extended_results = [
        results + [results[-1]] * (max_len - len(results))
        for results in results_list
    ]
    extended_results = [
        results + [results[-1]] * (max_len - len(results))
        for results in extended_results
    ]
    extended_results = [res[:max_len] for res in extended_results]
    avg_fitness      = np.mean(extended_results, axis=0)
    var_fitness      = np.var(extended_results, axis=0)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(avg_fitness, label="Average Fitness")
    plt.fill_between(
        range(len(avg_fitness)),
        avg_fitness - var_fitness,
        avg_fitness + var_fitness,
        alpha=0.2,
        label="Variance",
    )
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Average and Variance of Fitness (8-Queens) - {method} Selection - {runs} runs")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/results_avg_var_{method}.png")

def plot_all_results(avg_fit_list, runs, method):
    fig = plt.figure(figsize=(12, 6))
    for results in avg_fit_list:
        plt.plot(results)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.xlim([0, 1000])
    plt.title(f"Genetic Algorithm Fitness Progress (8-Queens) - {method} Selection - {runs} Runs - all")
    plt.tight_layout()
    fig.savefig(f"plots/results_all_{method}.png")
    plt.close()

def plot_histogram(opt_gen_list, runs , method):
    fig = plt.figure(figsize=(12, 6))
    plt.hist(opt_gen_list, bins=30, color='blue', edgecolor='black')
    plt.xlabel("Number of Generation")
    plt.ylabel("Frequency")
    plt.title(f"Genetic Algorithm Fitness Progress (8-Queens) - {method} Selection - {runs} Runs - all")
    plt.tight_layout()
    fig.savefig(f"plots/hist_max_generation_{method}.png")
    plt.close()
