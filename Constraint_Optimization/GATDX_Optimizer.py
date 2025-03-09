import numpy as np
from typing import List, Tuple, Callable
import random
import matplotlib.pyplot as plt  # Import matplotlib for plotting

class Individual:
    """Represents an individual in the population with its solution vector and fitness."""
    def __init__(self, solution: np.ndarray, fitness: float = None, fs: float = None):
        self.solution = solution
        self.fitness = fitness
        self.fs = fs

    def __lt__(self, other):
        return self.fitness < other.fitness

class GATDX:
    """Main Genetic Algorithm with Two-Direction Crossover and Grouped Mutation."""
    def __init__(self, 
                 population_size: int, 
                 dimension: int, 
                 bounds: Tuple[float, float], 
                 max_iterations: int, 
                 fitness_function: Callable[[np.ndarray], Tuple[float, float]],
                 beta: float = 0.5,  # Proportion of best group for grouped mutation
                 gamma: float = 5.0):  # Shape parameter for non-uniform mutation
        self.NP = population_size
        self.dim = dimension
        self.bounds = bounds
        self.T = max_iterations
        self.fitness_function = fitness_function
        self.beta = beta
        self.gamma = gamma
        self.population = self._initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_fs = float('inf')
        self.best_fitness_history = []  # List to store best fitness per generation

    def _initialize_population(self) -> List[Individual]:
        """Initialize the population with random solutions within bounds."""
        population = []
        for _ in range(self.NP):
            solution = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            fitness, fs = self.fitness_function(solution)
            population.append(Individual(solution, fitness, fs))
        return population

    def _sort_population(self) -> List[Individual]:
        """Sort population by fitness in ascending order."""
        return sorted(self.population, key=lambda x: x.fitness)

    def _grouping_selection(self) -> Tuple[List[Individual], List[Individual]]:
        """Divide sorted population into two groups."""
        sorted_pop = self._sort_population()
        half = self.NP // 2
        return sorted_pop[:half], sorted_pop[half:]

    def _update_population(self, new_individuals: List[Individual]):
        """Update population with the best individuals from old and new."""
        for i in range(self.NP):
            if new_individuals[i].fitness < self.population[i].fitness:
                self.population[i] = new_individuals[i]
            if self.population[i].fitness < self.best_fitness:
                self.best_fitness = self.population[i].fitness
                self.best_fs = self.population[i].fs
                self.best_solution = self.population[i].solution.copy()

    def run(self) -> Tuple[np.ndarray, float]:
        """Run the GA-TDX algorithm."""
        k = 0
        while k < self.T:
            # Step 1: Sorting and Grouping Selection
            group1, group2 = self._grouping_selection()

            # Step 2: Two-Direction Crossover
            crossover_pop = []
            for x1, x2 in zip(group1, group2):
                o5, o6 = TwoDirectionCrossover.crossover(x1, x2, self.fitness_function)
                crossover_pop.extend([Individual(x1.solution, x1.fitness, x1.fs), Individual(x2.solution, x2.fitness, x2.fs), o5, o6])
            crossover_best = sorted(crossover_pop, key=lambda x: x.fitness)[:self.NP]
            self._update_population(crossover_best)

            # Step 3: Grouped Mutation
            mutated_pop = GroupedMutation.mutate(self.population, self.fitness_function, k, self.T, self.beta, self.gamma, self.bounds)
            self._update_population(mutated_pop)

            k += 1
            self.best_fitness_history.append(self.best_fitness)  # Record best fitness of this generation

        return self.best_solution, self.best_fitness, self.best_fs

    def save_fitness_plot(self, path: str):
        """Save a plot of the best fitness per generation to the specified path."""
        plt.figure()
        plt.plot(self.best_fitness_history, label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Log Fitness')
        plt.yscale('log')
        plt.title('Log Fitness per Generation')
        plt.legend()
        plt.savefig(path)
        plt.close()

class TwoDirectionCrossover:
    """Implementation of Two-Direction Crossover (TDX)."""
    @staticmethod
    def _find_perpendicular_vector(d1: np.ndarray) -> np.ndarray:
        """Find a vector perpendicular to d1 with adjustments for zero components."""
        n = len(d1)
        dp = np.zeros(n)
        for j in range(n - 1):
            dp[j] = random.uniform(0, 1)
        
        mean_d1 = np.mean(np.abs(d1))
        if abs(d1[-1]) > 1e-10:  # Avoid division by zero
            dp[-1] = -np.sum(d1[:-1] * dp[:-1]) / d1[-1]
        else:
            dp[-1] = random.uniform(-mean_d1, mean_d1)
        
        # Scale dp to have the same length as d1
        dp = dp * np.linalg.norm(d1) / np.linalg.norm(dp)
        return dp

    @staticmethod
    def crossover(x1: Individual, x2: Individual, fitness_function: Callable[[np.ndarray], Tuple[float, float]]) -> Tuple[Individual, Individual]:
        """Perform Two-Direction Crossover between two parents."""
        d1 = x1.solution - x2.solution  # First direction: from worse to better
        dp = TwoDirectionCrossover._find_perpendicular_vector(d1)  # Perpendicular direction
        d2 = (d1 + dp) / 2  # Second direction (45 degrees)

        alpha = random.uniform(0, 1)
        
        # Generate four offspring
        o1 = x1.solution + alpha * d1
        o2 = x1.solution + alpha * d2
        o3 = x2.solution + alpha * d1
        o4 = x2.solution + alpha * d2

        # Evaluate fitness
        candidates = [
            Individual(o1, fitness_function(o1)[0], fitness_function(o1)[1]),
            Individual(o2, fitness_function(o2)[0], fitness_function(o2)[1]),
            Individual(o3, fitness_function(o3)[0], fitness_function(o3)[1]),
            Individual(o4, fitness_function(o4)[0], fitness_function(o4)[1])
        ]
        
        # Select the best from each pair
        o5 = min(candidates[:2], key=lambda x: x.fitness)
        o6 = min(candidates[2:], key=lambda x: x.fitness)
        
        return o5, o6

class GroupedMutation:
    """Implementation of Grouped Mutation (GM)."""
    @staticmethod
    def _normal_mutation(x1: np.ndarray, x_beta: np.ndarray) -> np.ndarray:
        """Normal mutation for local search."""
        variance = np.abs(x1 - x_beta) / 6
        return x1 + np.random.normal(0, variance)

    @staticmethod
    def _non_uniform_mutation(x: np.ndarray, t: int, T: int, gamma: float, bounds: Tuple[float, float]) -> np.ndarray:
        """Non-uniform mutation for global search."""
        delta = lambda t: (1 - random.random() ** (1 - t / T) ** gamma)
        result = x.copy()
        for j in range(len(x)):
            rand = random.random()
            if rand <= 0.5:
                result[j] += (bounds[1] - x[j]) * delta(t)
            else:
                result[j] += (bounds[0] - x[j]) * delta(t)
        return np.clip(result, bounds[0], bounds[1])

    @staticmethod
    def mutate(population: List[Individual], 
              fitness_function: Callable[[np.ndarray], Tuple[float, float]], 
              t: int, T: int, beta: float, gamma: float, 
              bounds: Tuple[float, float]) -> List[Individual]:
        """Perform Grouped Mutation on the population."""
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        split_idx = max(1, int(beta * len(population)))  # Ensure split_idx is at least 1
        
        # First group: Local search with normal mutation
        group1 = sorted_pop[:split_idx]
        o1 = []
        for i, ind in enumerate(group1):
            mutated = GroupedMutation._normal_mutation(ind.solution, sorted_pop[split_idx - 1].solution)
            o1.append(Individual(mutated, fitness_function(mutated)[0], fitness_function(mutated)[1]))

        # Second group: Global search with non-uniform mutation
        group2 = sorted_pop[split_idx:]
        o2 = [Individual(GroupedMutation._non_uniform_mutation(ind.solution, t, T, gamma, bounds), 
                        fitness_function(GroupedMutation._non_uniform_mutation(ind.solution, t, T, gamma, bounds))[0], 
                        fitness_function(GroupedMutation._non_uniform_mutation(ind.solution, t, T, gamma, bounds))[1]) 
              for ind in group2]

        return o1 + o2

class GATDXOptimizer:
    def __init__(self, 
                 population_size: int, 
                 dimension: int, 
                 bounds: Tuple[float, float], 
                 max_iterations: int, 
                 beta: float,
                 gamma: float,
                 fitness_function: Callable[[np.ndarray], Tuple[float, float]]):
        self.ga = GATDX(population_size, dimension, bounds, max_iterations, fitness_function, beta, gamma)

    def optimize(self):
        return self.ga.run()
    
    def save_fitness_plot(self, path: str):
        return self.ga.save_fitness_plot(path)
