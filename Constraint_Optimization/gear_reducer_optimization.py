import numpy as np
from GATDX_Optimizer import GATDXOptimizer
from config import GEAR_REDUCER_CONFIG

class GearReducerOptimization:
    def __init__(self):
        self.config = GEAR_REDUCER_CONFIG
    
    def objective_function(self, x):
        """
        Objective function to minimize the volume of the reducer
        x = [M, Z1, B, L, Dz1, Dz2]
        """
        M, Z1, B, L, Dz1, Dz2 = x
        
        V = 0.25 * np.pi * (
            B * (M**2 * Z1**2) + 
            (L-B) * (Dz1**2 + Dz2**2) 
        )
        return V
    def penalty_function(self, x, m=10**200):
        g, h = self.constraints(x)
        f = self.objective_function(x)
        penalty = m * (np.sum(np.maximum(0, g)**2) + np.sum(h**2))
        return f + penalty, f

    def constraints(self, x):
        """
        Implement all constraint functions g1 to g15
        """
        M, Z1, B, L, Dz1, Dz2 = x
        
        # Implement constraints based on the problem description
        constraints = [
            17 - x[1],  # g1
            0.9 - x[2] / (x[1] * x[0]),  # g2
            x[2] / (x[1] * x[0]) - 1.4,  # g3
            2 - x[0],  # g4
            x[1] * x[0] - 300,  # g5
            100 - x[3],  # g6
            x[3] - 150,  # g7
            x[5] - 200,  # g8
            x[2] + 0.5 * x[5] + 40 - x[4],  # g9
            1486250 / (x[0] * x[1] * np.sqrt(x[2])) - 550,  # g10
            9064860 * 1 / (x[0]**2 * x[1] * x[2] * x[3]) - 400,  # g12
            9064860 * 1 / (x[0]**2 * x[1] * x[2] * x[3]) - 400,  # g13
            (1 / x[4]**3) * np.sqrt((2.85 * 10**6 * x[3] / (x[0] * x[1]))**2 + 2.4 * 10**12) - 5.5,  # g14
            (1 / x[5]**3) * np.sqrt((2.85 * 10**6 * x[3] / (x[0] * x[1]))**2 + 6 * 10**13) - 5.5,  # g15
        ]
        return np.array(constraints), np.array([])
    
    def optimize(self):
        """
        Run the optimization using GATDX
        """
        optimizer = GATDXOptimizer(
            fitness_function=self.penalty_function,
            bounds=(-200,200),
            dimension=6,
            max_iterations=10000,
            population_size=10,
            beta=0.2,
            gamma=5.0
        )
        
        best_solution, best_fitness, best_fs = optimizer.optimize()
        return best_solution, best_fitness, best_fs, optimizer

def main():
    gear_reducer_opt = GearReducerOptimization()
    best_solution, best_fitness, best_fs, optimizer = gear_reducer_opt.optimize()
    optimizer.save_fitness_plot("gear_reducer_fitness.png")
    
    print("Best Solution:", best_solution)
    print("Best Fitness (Volume):", best_fitness)
    print("Best FS:", best_fs)
if __name__ == "__main__":
    main() 