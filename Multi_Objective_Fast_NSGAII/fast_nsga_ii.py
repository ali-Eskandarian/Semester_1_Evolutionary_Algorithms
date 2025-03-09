import numpy as np
import copy
from typing import List, Tuple
from tqdm import tqdm
from src.metrics import PerformanceMetrics

class FastNSGAII:
    def __init__(
        self, 
        problem_evaluator, 
        population_size: int = 100, 
        max_generations: int = 200, 
        crossover_operator=None, 
        mutation_rate: float = 0.1,
        track_metrics: bool = False
    ):
        """
        Initialize Fast NSGA-II algorithm
        
        Args:
            problem_evaluator: Evaluator for the problem
            population_size (int): Size of population
            max_generations (int): Maximum number of generations
            crossover_operator (callable): Crossover method
            mutation_rate (float): Probability of mutation
            track_metrics (bool): Whether to track generation metrics
        """
        self.problem_evaluator = problem_evaluator
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_operator = crossover_operator
        self.mutation_rate = mutation_rate
        self.track_metrics = track_metrics
        self.generation_metrics = {
            'hypervolume': [],
            'igd': []
        } if track_metrics else None
        
        # Problem-specific parameters
        self.n_var, self.n_obj = problem_evaluator.n_var, problem_evaluator.M
        self.lower_bounds, self.upper_bounds = problem_evaluator.get_bounds()
    
    def initialize_population(self) -> np.ndarray:
        """
        Initialize random population within problem bounds
        
        Returns:
            np.ndarray: Initial population
        """
        return np.random.uniform(
            low=self.lower_bounds, 
            high=self.upper_bounds, 
            size=(self.population_size, self.n_var)
        )
    
    def fast_non_dominated_sort(self, population_objectives: np.ndarray) -> List[List[int]]:
        """
        Fast non-dominated sorting algorithm
        
        Args:
            population_objectives (np.ndarray): Objectives of population
        
        Returns:
            List of Pareto front ranks
        """
        n = len(population_objectives)
        fronts = [[]]
        
        # Domination counting and dominated sets
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                
                # Check domination
                if self._is_dominated(population_objectives[p], population_objectives[q]):
                    dominated_solutions[p].append(q)
                    domination_count[q] += 1
                elif self._is_dominated(population_objectives[q], population_objectives[p]):
                    domination_count[p] += 1
            
            # First front
            if domination_count[p] == 0:
                fronts[0].append(p)
        
        # Subsequent fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            for p in fronts[front_index]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            
            front_index += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts
    
    def _is_dominated(self, solution1: np.ndarray, solution2: np.ndarray) -> bool:
        """
        Check if solution1 is dominated by solution2
        
        Args:
            solution1 (np.ndarray): First solution objectives
            solution2 (np.ndarray): Second solution objectives
        
        Returns:
            bool: True if solution1 is dominated
        """
        return np.all(solution1 >= solution2) and np.any(solution1 > solution2)
    
    def crowding_distance_assignment(self, front_objectives: np.ndarray) -> np.ndarray:
        """
        Calculate crowding distance for solutions in a front
        
        Args:
            front_objectives (np.ndarray): Objectives of solutions in a front
        
        Returns:
            np.ndarray: Crowding distances
        """
        n_solutions, n_objectives = front_objectives.shape
        crowding_distances = np.zeros(n_solutions)
        
        for m in range(n_objectives):
            # Sort solutions by m-th objective
            sorted_indices = np.argsort(front_objectives[:, m])
            sorted_objectives = front_objectives[sorted_indices, m]
            
            # Boundary solutions get infinite distance
            crowding_distances[sorted_indices[0]] = np.inf
            crowding_distances[sorted_indices[-1]] = np.inf
            
            # Normalize objective range
            obj_range = sorted_objectives[-1] - sorted_objectives[0]
            if obj_range == 0:
                continue
            
            # Calculate crowding distance
            for i in range(1, n_solutions - 1):
                crowding_distances[sorted_indices[i]] += (
                    sorted_objectives[i+1] - sorted_objectives[i-1]
                ) / obj_range
        
        return crowding_distances
    
    def selection(self, population, objectives):
        """
        Select parent using binary tournament selection
        
        Args:
            population (np.ndarray): Current population
            objectives (np.ndarray): Population objectives
        
        Returns:
            np.ndarray: Selected parent
        """
        # Get non-dominated fronts
        fronts = self.fast_non_dominated_sort(objectives)
        
        # Calculate crowding distance for each front
        crowding_distances = np.zeros(len(population))
        for front in fronts:
            front_objectives = objectives[front]
            front_crowding = self.crowding_distance_assignment(front_objectives)
            crowding_distances[front] = front_crowding
        
        # Tournament selection
        tournament_size = 2
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        
        # Get ranks for tournament individuals
        ranks = np.zeros(tournament_size, dtype=int)
        for i, idx in enumerate(tournament_indices):
            for rank, front in enumerate(fronts):
                if idx in front:
                    ranks[i] = rank
                    break
        
        # Select based on rank and crowding distance
        best_idx = tournament_indices[0]
        best_rank = ranks[0]
        best_crowd = crowding_distances[tournament_indices[0]]
        
        for i in range(1, tournament_size):
            current_idx = tournament_indices[i]
            current_rank = ranks[i]
            current_crowd = crowding_distances[current_idx]
            
            # Select based on better rank or equal rank but better crowding distance
            if (current_rank < best_rank) or \
               (current_rank == best_rank and current_crowd > best_crowd):
                best_idx = current_idx
                best_rank = current_rank
                best_crowd = current_crowd
        return population[best_idx]

    def mutation(self, population):
        """
        Polynomial mutation
        
        Args:
            population (np.ndarray): Population to mutate
        
        Returns:
            np.ndarray: Mutated population
        """
        mutated_population = population.copy()
        
        for i in range(len(mutated_population)):
            if np.random.random() < self.mutation_rate:
                # Polynomial mutation
                for j in range(self.n_var):
                    r = np.random.random()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (1 + 20)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (1 + 20))
                    
                    mutated_population[i, j] += delta * (
                        self.upper_bounds[j] - self.lower_bounds[j]
                    )
                    
                    # Ensure bounds
                    mutated_population[i, j] = np.clip(
                        mutated_population[i, j], 
                        self.lower_bounds[j], 
                        self.upper_bounds[j]
                    )
        
        return mutated_population
    
    def run(self):
        """
        Run Fast NSGA-II algorithm
        
        Returns:
            Tuple of final population and their objectives
        """
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        objectives = np.array([
            self.problem_evaluator.evaluate(ind) for ind in population
        ])
        
        true_front = self.problem_evaluator.get_true_pareto_front()
        
        for generation in tqdm(range(self.max_generations)):
            # Crossover and mutation
            offspring = []
            for _ in range(self.population_size // 2):
                # Select parents
                parent1, parent2 = self.selection(population, objectives), self.selection(population, objectives)
                
                # Crossover
                if self.crossover_operator:
                    child1, child2 = self.crossover_operator(parent1, parent2)
                    offspring.extend([child1, child2])
            
            # Mutate offspring
            offspring = self.mutation(np.array(offspring))
            
            # Evaluate offspring
            offspring_objectives = np.array([
                self.problem_evaluator.evaluate(ind) for ind in offspring
            ])
            
            # Combine population and offspring
            combined_population = np.vstack([population, offspring])
            combined_objectives = np.vstack([objectives, offspring_objectives])
            
            # Non-dominated sorting and selection
            fronts = self.fast_non_dominated_sort(combined_objectives)
            
            # Select next generation
            next_population = []
            next_objectives = []
            
            for front in fronts:
                front_population = combined_population[front]
                front_objectives = combined_objectives[front]
                
                # Crowding distance selection
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend(front_population)
                    next_objectives.extend(front_objectives)
                else:
                    # Select based on crowding distance
                    remaining = self.population_size - len(next_population)
                    crowding_distances = self.crowding_distance_assignment(front_objectives)
                    sorted_indices = np.argsort(crowding_distances)[::-1]
                    
                    next_population.extend(front_population[sorted_indices[:remaining]])
                    next_objectives.extend(front_objectives[sorted_indices[:remaining]])
                    break
            
            population = np.array(next_population)
            objectives = np.array(next_objectives)
            
            if self.track_metrics:
                hv = PerformanceMetrics.hypervolume(objectives)
                self.generation_metrics['hypervolume'].append(hv)
                
                igd = PerformanceMetrics.inverted_generational_distance(objectives, true_front)
                self.generation_metrics['igd'].append(igd)
        
        # Return population, objectives, and optional generation metrics
        return (population, objectives, self.generation_metrics) if self.track_metrics else (population, objectives) 