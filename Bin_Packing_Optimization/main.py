import random
from typing import List, Tuple
import numpy as np
from solver import GeneticAlgorithm
from utils import *


def main():
    # Test different configurations for each instance
    instances = ['1', '2', '3', '4', '5']
    configurations = {
        'insert_edge': {
            'mutation_type': 'insert',
            'crossover_method': 'edge'
        },
        'scramble_ox': {
            'mutation_type': 'scramble',
            'crossover_method': 'ox'
        },
        'swap_pmx': {
            'mutation_type': 'swap',
            'crossover_method': 'pmx'
        },
        'inversion_cycle': {
            'mutation_type': 'inversion',
            'crossover_method': 'cycle'
        },
    }
    
    # for instance in instances:
    #     items, bin_capacity = read_instance(f"BPP/instance_sets/{instance}.txt")
        
    #     # Calculate and print the theoretical optimal number of bins
    #     optimal_bins = calculate_optimal_bins(items, bin_capacity)
    #     print(f"Instance {instance}: Theoretical optimal number of bins = {optimal_bins}")
        
    #     results = {}
    #     os.makedirs(f'solutions/instance_{instance}', exist_ok=True)
        
    #     for config_name, params in configurations.items():
    #         ga = GeneticAlgorithm(
    #             items=items,
    #             bin_capacity=bin_capacity,
    #             population_size=100,
    #             generations=200,
    #             tournament_size=5,
    #             mutation_rate=0.05,
    #             mutation_type=params['mutation_type'],
    #             initialization_method='random',
    #             crossover_method=params['crossover_method'],
    #             save_path=f'plots/instance_{instance}',
    #             use_local_search=True
    #         )
            
    #         best_solution = ga.evolve()
            
    #         # Store results for plotting
    #         results[config_name] = {
    #             'fitness_history': ga.fitness_history,
    #             'initial_fitness': ga.initial_fitness,
    #             'final_fitness': ga.final_fitness
    #         }
            
    #         write_solution(f"solutions/instance_{instance}/{config_name}.txt", best_solution)
        
    #     # Create comparison plots for this instance
    #     create_comparison_plots(results, instance)

    for instance in instances:
        all_items, bin_capacity = read_instance(f"BPP/instance_sets/{instance}.txt")
        
        bins = read_solution(f"solutions/instance_{instance}/swap_pmx.txt", bin_capacity, all_items)
        
        plot_solution(bins, instance)
        
        print(f"Instance {instance}: Visualization complete.")

if __name__ == "__main__":
    main()
