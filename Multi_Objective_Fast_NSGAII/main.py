import numpy as np
import os
import matplotlib.pyplot as plt
from src.wfg_evaluator import WFGEvaluator
from src.fast_nsga_ii import FastNSGAII
from src.adaptive_crossover import create_adaptive_crossover
from src.crossover_operators import CrossoverOperators
from src.metrics import PerformanceMetrics
from src.visualization import ParetoFrontVisualization

def run_experiment(problem_name, n_var, M):
    """
    Run experiment for a specific WFG problem
    
    Args:
        problem_name (str): WFG problem name
        n_var (int): Number of decision variables
        M (int): Number of objectives
    """
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    wfg_evaluator = WFGEvaluator(problem_name, n_var, M)
    
    adaptive_crossover = create_adaptive_crossover()
    
    crossover_method = adaptive_crossover.select_crossover_operator()
    
    nsga_ii = FastNSGAII(
        problem_evaluator=wfg_evaluator,
        population_size=150,
        max_generations=50,
        crossover_operator=crossover_method,
        mutation_rate=0.1,
        track_metrics=True  # Enable metric tracking
    )
    
    population, objectives, generation_metrics = nsga_ii.run()
    
    true_front = wfg_evaluator.get_true_pareto_front()
    
    hv = PerformanceMetrics.hypervolume(objectives)
    igd = PerformanceMetrics.inverted_generational_distance(objectives, true_front)
    
    print(f"Problem: {problem_name}")
    print(f"Hypervolume: {hv}")
    print(f"Inverted Generational Distance: {igd}")
    
    # Plot Pareto Front
    ParetoFrontVisualization.plot_pareto_fronts(
        objectives, 
        true_front, 
        problem_name, 
        save_path=f'results/{problem_name}_pareto_front.png'
    )
    
    # Plot Generation Metrics
    plt.figure(figsize=(12, 5))
    
    # Hypervolume subplot
    plt.subplot(1, 2, 1)
    plt.plot(generation_metrics['hypervolume'], label='Hypervolume')
    plt.title(f'{problem_name.upper()} - Hypervolume per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume')
    plt.legend()
    
    # IGD subplot
    plt.subplot(1, 2, 2)
    plt.plot(generation_metrics['igd'], label='IGD', color='orange')
    plt.title(f'{problem_name.upper()} - IGD per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Inverted Generational Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{problem_name}_generation_metrics.png')
    plt.close()
    
    adaptive_crossover.update_performance(0, hv)

def main():
    # WFG problems to test
    wfg_problems = [
        ('wfg1', 10, 2),
        ('wfg2', 10, 2),
        ('wfg3', 10, 2),
        ('wfg4', 10, 2),
        ('wfg5', 10, 2),
        ('wfg6', 10, 2),
        ('wfg7', 10, 2),
        ('wfg8', 10, 2),
        ('wfg9', 10, 2),
    ]
    
    for problem_name, n_var, M in wfg_problems:
        run_experiment(problem_name, n_var, M)

if __name__ == "__main__":
    main() 