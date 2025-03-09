from typing import List, Tuple
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from copy import deepcopy

class Item:
    def __init__(self, id: int, size: int):
        self.id = id
        self.size = size

    def __str__(self):
        return f"Item {self.id} (size: {self.size})"

class Bin:
    def __init__(self, capacity: int):  
        self.capacity = capacity
        self.items: List[Item] = []
        self.remaining_space = capacity

    def can_add_item(self, item: Item) -> bool:
        return self.remaining_space >= item.size

    def add_item(self, item: Item) -> bool:
        if self.can_add_item(item):
            self.items.append(item)
            self.remaining_space -= item.size
            return True
        return False
    
    def __str__(self):
        return f"Bin (used: {self.capacity - self.remaining_space}/{self.capacity})"

class Individual:
    def __init__(self, chromosome: List[int], items: List[Item], bin_capacity: int):
        self.chromosome = chromosome
        self.items = items
        self.bin_capacity = bin_capacity
        self.bin_count = self.count_bins()
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self) -> float:
        bins: List[Bin] = []
        for gene in self.chromosome:
            item = self.items[gene - 1]
            packed = False
            
            # Try to pack in existing bins
            for bin in bins:
                if bin.add_item(item):
                    packed = True
                    break
            
            # Create new bin if needed
            if not packed:
                new_bin = Bin(self.bin_capacity)
                new_bin.add_item(item)
                bins.append(new_bin)

        fitness = 0
        for bin in bins:
            fitness += np.exp(-1 * (bin.remaining_space / self.bin_capacity)**2)
        fitness = fitness/len(bins)

        return fitness

    def count_bins(self) -> int:
        bins: List[Bin] = []
        for gene in self.chromosome:
            item = self.items[gene - 1]
            packed = False
            
            # Try to pack in existing bins
            for bin in bins:
                if bin.add_item(item):
                    packed = True
                    break
            
            # Create new bin if needed
            if not packed:
                new_bin = Bin(self.bin_capacity)
                new_bin.add_item(item)
                bins.append(new_bin)

        return len(bins)

class GeneticAlgorithm:
    
    def __init__(self, 
                 items: List[Item], 
                 bin_capacity: int, 
                 population_size: int = 100,
                 generations: int = 10000,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.05,
                 mutation_type: str = 'swap',  # 'swap', 'insert', 'scramble', 'inversion'
                 initialization_method: str = 'random',  # 'random' or 'ffd'
                 crossover_method: str = 'ox',  # 'ox' or 'hybrid'
                 save_path: str = 'plots/instance_1',  # 'ox' or 'hybrid'
                 use_local_search: bool = False):
        self.items = items
        self.bin_capacity = bin_capacity
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_type = mutation_type
        self.initialization_method = initialization_method
        self.crossover_method = crossover_method
        self.use_local_search = use_local_search
        self.population: List[Individual] = []  
        self.save_path = save_path
        self.fitness_history = []
        self.initial_fitness = []
        self.final_fitness = []
        self.initial_bin_counts = []

    def ffd_initialization(self):
        # Sort items by size in descending order
        sorted_items = sorted(self.items, key=lambda x: x.size, reverse=True)
        sorted_indices = [item.id for item in sorted_items]
        return sorted_indices

    def initialize_population(self):
        for _ in range(self.population_size):
            if self.initialization_method == 'ffd':
                # Create FFD-based chromosome
                chromosome = self.ffd_initialization()
                if random.random() < 0.5:  # Add some randomness
                    random.shuffle(chromosome)
            else:
                # Original random initialization
                chromosome = list(range(1, len(self.items) + 1))
                random.shuffle(chromosome)
            
            self.population.append(Individual(chromosome, self.items, self.bin_capacity))
        
        for ind in self.population:
            self.initial_bin_counts.append(ind.bin_count)

    def tournament_selection(self) -> Individual:
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        size = len(self.items)
        
        if self.crossover_method == 'pmx':
        
            # PMX crossover
            point1, point2 = sorted(random.sample(range(size), 2))
            
            def create_pmx_child(p1, p2):
                child = [-1] * size
                child[point1:point2] = p1.chromosome[point1:point2]

                for i in range(point1, point2):
                    element = p2.chromosome[i]
                    if element not in child[point1:point2]:
                        current_pos = i
                        while point1 <= current_pos < point2:
                            copied_element = p1.chromosome[current_pos]
                            current_pos = p2.chromosome.index(copied_element)

                        child[current_pos] = p2.chromosome[i]

                for i in range(size):
                    if child[i] == -1:
                        child[i] = p2.chromosome[i]
                
                return Individual(child, self.items, self.bin_capacity)
            
            child1 = create_pmx_child(parent1, parent2)
            child2 = create_pmx_child(parent2, parent1)
            
        elif self.crossover_method == 'edge':
            # Edge crossover
            def build_edge_table(p1, p2):
                table = {i: set() for i in range(1, size + 1)}
                
                def add_edges(chromosome):
                    for i in range(size):
                        current = chromosome[i]
                        prev = chromosome[i-1]
                        next = chromosome[(i+1) % size]
                        table[current].add(prev)
                        table[current].add(next)
                
                add_edges(p1.chromosome)
                add_edges(p2.chromosome)
                return table
            
            def create_edge_child():
                edge_table = build_edge_table(parent1, parent2)
                child = []
                current = random.choice(list(edge_table.keys()))
                
                while len(child) < size:
                    child.append(current)
                    for neighbors in edge_table.values():
                        neighbors.discard(current)
                    
                    if current in edge_table:
                        neighbors = edge_table[current]
                        del edge_table[current]
                        
                        if neighbors:
                            next_element = min(neighbors, 
                                            key=lambda x: len(edge_table.get(x, set())) if x in edge_table else float('inf'))
                        else:
                            next_element = random.choice(list(edge_table.keys())) if edge_table else child[0]
                        current = next_element
                        
                return Individual(child, self.items, self.bin_capacity)
            
            child1 = create_edge_child()
            child2 = create_edge_child()
            
        elif self.crossover_method == 'cycle':
            # Cycle crossover
            def find_cycle(p1, p2, start_pos):
                cycle = {start_pos}
                pos = start_pos
                while True:
                    value = p2.chromosome[pos]
                    pos = p1.chromosome.index(value)
                    if pos == start_pos:
                        break
                    cycle.add(pos)
                return cycle
            
            def create_cycle_child(p1, p2):
                child = [-1] * size
                unused_positions = set(range(size))
                
                while unused_positions:
                    start_pos = min(unused_positions)
                    cycle = find_cycle(p1, p2, start_pos)
                    
                    parent = p1 if len(unused_positions) % 2 == 0 else p2
                    
                    for pos in cycle:
                        child[pos] = parent.chromosome[pos]
                        unused_positions.remove(pos)
                        
                return Individual(child, self.items, self.bin_capacity)
            
            child1 = create_cycle_child(parent1, parent2)
            child2 = create_cycle_child(parent2, parent1)
            
        else:  # Default OX crossover
            # Order crossover (OX)
            point1, point2 = sorted(random.sample(range(size), 2))
            
            def create_ox_child(p1, p2):
                segment = p1.chromosome[point1:point2]
                remaining = [gene for gene in p2.chromosome if gene not in segment]
                child = remaining[:point1] + segment + remaining[point1:]
                return Individual(child, self.items, self.bin_capacity)

            child1 = create_ox_child(parent1, parent2)
            child2 = create_ox_child(parent2, parent1)
            
        return child1, child2

    def mutate(self, individual: Individual):
        # Mutation

        if self.mutation_type == 'swap':
            # Swap mutation - exchange two random positions
            idx1, idx2 = random.randint(0, len(self.items)-1), (random.randint(0, len(self.items)-1) + 1) % len(self.items)
            individual.chromosome[idx1], individual.chromosome[idx2] = \
                individual.chromosome[idx2], individual.chromosome[idx1]
            
        elif self.mutation_type == 'insert':
            # Insert mutation - move second position next to first
            idx1, idx2 = sorted(random.sample(range(len(self.items)), 2))
            value = individual.chromosome.pop(idx2)
            individual.chromosome.insert(idx1 + 1, value)

        elif self.mutation_type == 'scramble':
            # Scramble mutation - randomize order of a subsequence
            idx1, idx2 = sorted(random.sample(range(len(self.items)), 2))
            subsequence = individual.chromosome[idx1:idx2+1]
            random.shuffle(subsequence)
            individual.chromosome[idx1:idx2+1] = subsequence

        elif self.mutation_type == 'inversion':
            # Inversion mutation - reverse a subsequence
            idx1, idx2 = sorted(random.sample(range(len(self.items)), 2))
            individual.chromosome[idx1:idx2+1] = \
                individual.chromosome[idx1:idx2+1][::-1]    
            

        individual.fitness = individual.calculate_fitness()
        return individual
    
    def evolve(self):
        self.initialize_population()
        best_solution = max(self.population, key=lambda ind: ind.fitness)
        
        # Store initial generation fitness
        self.initial_fitness = [ind.fitness for ind in self.population]
        
        generation_stats = []
        for generation in range(self.generations):
            current_fitness = [ind.fitness for ind in self.population]
            generation_stats.append({
                'generation': generation,
                'avg_fitness': sum(current_fitness) / len(current_fitness),
                'max_fitness': max(current_fitness)
            })
            
            new_population = []
            # Elitism: keep the best individual
            new_population.append(best_solution)
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2)

                if random.random()<self.mutation_rate:
                    child1 = self.mutate(deepcopy(child1))
                else:
                    child1 = deepcopy(child1)   
                if random.random()<self.mutation_rate:
                    child2 = self.mutate(deepcopy(child2))
                else:
                    child2 = deepcopy(child2)
                
                new_population.extend([child1, child2])
            self.population = new_population[:self.population_size]
            current_best = max(self.population, key=lambda ind: ind.fitness)
            
            if current_best.fitness > best_solution.fitness:
                best_solution = current_best

        # Store final generation fitness
        self.final_fitness = [ind.fitness for ind in self.population]
        self.fitness_history = generation_stats
        
        self.plot_fitness_progress()
        self.plot_generation_comparison()
        self.plot_bin_count_comparison()
        
        return best_solution

    def plot_fitness_progress(self):
        """Plot average and max fitness over generations using Plotly"""
        generations = [stat['generation'] for stat in self.fitness_history]
        avg_fitness = [stat['avg_fitness'] for stat in self.fitness_history]
        max_fitness = [stat['max_fitness'] for stat in self.fitness_history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=avg_fitness,
            name='Average Fitness',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=max_fitness,
            name='Max Fitness',
            mode='lines'
        ))
        
        fig.update_layout(
            title='Fitness Progress Over Generations',
            xaxis_title='Generation',
            yaxis_title='Fitness',
            template='plotly_white'
        )
        fig.write_html(f'{self.save_path}/fitness_progress_{self.mutation_type}_crossover_{self.crossover_method}.html')

    def plot_generation_comparison(self):
        """Plot histogram comparison of initial and final generation fitness"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=self.initial_fitness,
                name='Initial Generation',
                opacity=0.75,
                marker_color='blue',
                nbinsx=5
            )
        )
        
        fig.add_trace(
            go.Histogram(
                x=self.final_fitness,
                name='Final Generation',
                opacity=0.75,
                marker_color='red',
                nbinsx=5
            )
        )
        
        fig.update_layout(
            title_text='Fitness Distribution Comparison',
            xaxis_title='Fitness',
            yaxis_title='Count',
            barmode='overlay',
            template='plotly_white'
        )
        fig.write_html(f'{self.save_path}/generation_comparison_{self.mutation_type}_crossover_{self.crossover_method}.html')

    def plot_bin_count_comparison(self):
        """Plot histogram comparison of bin counts for initial and final generations"""
        # Calculate bin counts for initial generation
       
        
        # Calculate bin counts for final generation
        final_bin_counts = []
        for ind in self.population:
            final_bin_counts.append(ind.bin_count)

        fig = go.Figure()
        
        fig.add_trace(
            go.Histogram(
                x=self.initial_bin_counts,
                name='Initial Generation',
                opacity=0.75,
                marker_color='blue',
                nbinsx=5
            )
        )
        
        fig.add_trace(
            go.Histogram(
                x=final_bin_counts,
                name='Final Generation',
                opacity=0.75,
                marker_color='red',
                nbinsx=5
            )
        )
        
        fig.update_layout(
            title_text='Bin Count Distribution Comparison',
            xaxis_title='Number of Bins',
            yaxis_title='Count',
            barmode='overlay',
            template='plotly_white'
        )
        
        fig.write_html(f'{self.save_path}/bin_count_comparison_mutation_{self.mutation_type}_crossover_{self.crossover_method}.html')

