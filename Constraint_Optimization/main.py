import numpy as np
from typing import List, Tuple, Callable
import random

class Individual:
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]]):
        self.dimensions = dimensions
        self.bounds = bounds
        self.genes = self._initialize_genes()
        self.fitness = float('inf')
        self.constraint_violation = 0.0
        
    def _initialize_genes(self) -> np.ndarray:
        """Initialize genes randomly within bounds"""
        return np.array([random.uniform(self.bounds[i][0], self.bounds[i][1]) 
                        for i in range(self.dimensions)])
    
    def __lt__(self, other):
        """Compare individuals based on constraint violation and fitness"""
        if self.constraint_violation == other.constraint_violation:
            return self.fitness < other.fitness
        return self.constraint_violation < other.constraint_violation

class GeneticAlgorithm:
    def __init__(self, 
                 population_size: int,
                 dimensions: int,
                 bounds: List[Tuple[float, float]],
                 objective_function: Callable,
                 constraint_functions: List[Callable],
                 max_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1):
        
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
        self.objective_function = objective_function
        self.constraint_functions = constraint_functions
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        self.population = self._initialize_population()
        self.best_solution = None
        
    def _initialize_population(self) -> List[Individual]:
        """Initialize population with random individuals"""
        return [Individual(self.dimensions, self.bounds) 
                for _ in range(self.population_size)]
    
    def evaluate_individual(self, individual: Individual):
        """Evaluate fitness and constraint violation for an individual"""
        individual.fitness = self.objective_function(individual.genes)
        individual.constraint_violation = sum(max(0, constraint(individual.genes)) 
                                           for constraint in self.constraint_functions)
    
    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection"""
        tournament = random.sample(self.population, tournament_size)
        return min(tournament)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform SBX (Simulated Binary Crossover)"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        child1, child2 = Individual(self.dimensions, self.bounds), Individual(self.dimensions, self.bounds)
        eta = 20  # Distribution index
        
        for i in range(self.dimensions):
            if random.random() <= 0.5:
                if abs(parent1.genes[i] - parent2.genes[i]) > 1e-14:
                    x1, x2 = parent1.genes[i], parent2.genes[i]
                    x1, x2 = min(x1, x2), max(x1, x2)
                    
                    beta = 1.0 + (2.0 * (x1 - self.bounds[i][0]) / (x2 - x1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    rand = random.random()
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    
                    child1.genes[i] = 0.5 * ((1 + beta_q) * x1 + (1 - beta_q) * x2)
                    child2.genes[i] = 0.5 * ((1 - beta_q) * x1 + (1 + beta_q) * x2)
                    
                    # Ensure bounds
                    child1.genes[i] = np.clip(child1.genes[i], self.bounds[i][0], self.bounds[i][1])
                    child2.genes[i] = np.clip(child2.genes[i], self.bounds[i][0], self.bounds[i][1])
                else:
                    child1.genes[i] = parent1.genes[i]
                    child2.genes[i] = parent2.genes[i]
            else:
                child1.genes[i] = parent1.genes[i]
                child2.genes[i] = parent2.genes[i]
                
        return child1, child2
    
    def mutation(self, individual: Individual):
        """Perform polynomial mutation"""
        eta = 20  # Distribution index
        
        for i in range(self.dimensions):
            if random.random() <= self.mutation_rate:
                y = individual.genes[i]
                lb, ub = self.bounds[i]
                
                delta1 = (y - lb) / (ub - lb)
                delta2 = (ub - y) / (ub - lb)
                rand = random.random()
                
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (ub - lb)
                y = np.clip(y, lb, ub)
                
                individual.genes[i] = y
    
    def evolve(self):
        """Main evolution loop"""
        # Evaluate initial population
        for individual in self.population:
            self.evaluate_individual(individual)
        
        for generation in range(self.max_generations):
            offspring = []
            
            # Create offspring
            while len(offspring) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutation(child1)
                self.mutation(child2)
                
                offspring.extend([child1, child2])
            
            # Evaluate offspring
            for individual in offspring:
                self.evaluate_individual(individual)
            
            # Combine populations and select the best individuals
            combined = self.population + offspring
            combined.sort()  # Sort based on constraint violation and fitness
            self.population = combined[:self.population_size]
            
            # Update best solution
            feasible_solutions = [ind for ind in self.population if ind.constraint_violation == 0]
            if feasible_solutions:
                current_best = min(feasible_solutions, key=lambda x: x.fitness)
                if self.best_solution is None or current_best.fitness < self.best_solution.fitness:
                    self.best_solution = current_best
            
            if (generation + 1) % 10 == 0:
                best_feasible = self.best_solution.fitness if self.best_solution else "No feasible solution"
                print(f"Generation {generation + 1}: Best feasible fitness = {best_feasible}")
    
    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Return the best solution found"""
        if self.best_solution is None:
            raise ValueError("No feasible solution found")
        return self.best_solution.genes, self.best_solution.fitness

# Example usage
def example_objective_function(x):
    """Example objective function (minimize): f(x) = x1^2 + x2^2"""
    return np.sum(x**2)

def example_constraint_function1(x):
    """Example constraint: g1(x) = x1 + x2 - 1 <= 0"""
    return x[0] + x[1] - 1

def example_constraint_function2(x):
    """Example constraint: g2(x) = x1 - x2 - 2 <= 0"""
    return x[0] - x[1] - 2

if __name__ == "__main__":
    # Problem setup
    dimensions = 2
    bounds = [(-5, 5)] * dimensions
    population_size = 100
    max_generations = 100
    
    # Create and run GA
    ga = GeneticAlgorithm(
        population_size=population_size,
        dimensions=dimensions,
        bounds=bounds,
        objective_function=example_objective_function,
        constraint_functions=[example_constraint_function1, example_constraint_function2],
        max_generations=max_generations
    )
    
    ga.evolve()
    
    # Get results
    best_solution, best_fitness = ga.get_best_solution()
    print("\nOptimization completed")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
