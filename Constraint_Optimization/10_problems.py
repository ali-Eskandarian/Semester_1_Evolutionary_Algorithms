from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple, Callable
import random
from GATDX_Optimizer import GATDX
from cops import *
from config import csop

if __name__ == "__main__":
    # Parameters
    population_size = 100
    max_iterations = 1000
    errors = []
    for problem in csop:
        
        print(f"Problem {problem['num']}:")
        # Initialize and run GA-TDX
        ga = GATDX(population_size, problem['vars'], problem['bounds'], max_iterations, problem['function'])
        best_solution, best_fitness, best_fs = ga.run()
        error = 100*abs(best_fs - problem['f_x'])/problem['f_x']
        print(f"Best Solution: {best_solution}")
        print(f"Best Function Value: {best_fs}")
        print(f"Best Fitness: {best_fitness}")
        print(f"Optimal Solution: {problem['best_solution']}")
        print(f"Optimal Function Value: {problem['f_x']}")
        print(f"Error: {error}%")
        print("--------------------------------")
        errors.append(error)
        
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(errors)), errors)
    plt.ylabel('Error (%)')
    plt.xlabel('Problem Number')
    plt.title('Error of GA-TDX for each problem')
    plt.savefig('errors.png')
    plt.close()
    