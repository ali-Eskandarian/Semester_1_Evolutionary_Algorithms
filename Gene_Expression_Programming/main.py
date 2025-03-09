from utils                   import *
from models                  import *
from operators               import *
from config                  import GEPConfig as cfg
from config                  import DataClass as data


# Main GEP algorithm
def run_gep():
    population = GEPPopulation(cfg.POPULATION_SIZE, cfg.HEAD_LENGTH)
    best_fitness_history = []
    avg_fitness_history = []  # New list to track average fitness
    
    for generation in range(cfg.MAX_GENERATIONS):
        population.evaluate_population(data.X_train, data.y_train)
        
        # Track best and average fitness
        best_fitness_history.append(population.chromosomes[0].fitness)
        avg_fitness = np.mean([chrom.fitness for chrom in population.chromosomes])
        avg_fitness_history.append(avg_fitness)
        
        if generation % 10 == 0:
            print(f"Generation {generation}, Best Fitness: {population.chromosomes[0].fitness:.6f}, Avg Fitness: {avg_fitness:.6f}")
        
        new_population = []
        new_population.append(population.chromosomes[0])
        
        while len(new_population) < cfg.POPULATION_SIZE:
            parent1 = population.selection()
            parent2 = population.selection()

            #Mutation
            if np.random.random() < cfg.MUTATION_RATE:
                parent1 = mutation(parent1)
            if np.random.random() < cfg.MUTATION_RATE:
                parent2 = mutation(parent2)

            #Transposition
            if np.random.random() < cfg.TRANSPOSITION_IS_RATE:
                parent1 = is_transposition(parent1)
            if np.random.random() < cfg.TRANSPOSITION_IS_RATE:
                parent2 = is_transposition(parent2)    

            #Crossover
            if np.random.random() < cfg.CROSSOVER_ONE_RATE:
                parent1, parent2 = recombination(parent1, parent2, 'one-point')
            if np.random.random() < cfg.CROSSOVER_TWO_RATE:
                parent1, parent2 = recombination(parent1, parent2, 'two-point')
            if np.random.random() < cfg.CROSSOVER_GENE_RATE:
                parent1, parent2 = recombination(parent1, parent2, 'gene')


            
            new_population.extend([parent1, parent2])
        
        population.chromosomes = new_population[:cfg.POPULATION_SIZE]
    
    # Updated plotting code to show both metrics
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.title('Fitness Evolution Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (MSE)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('fitness_history.png')
    plt.close()
    
    return population.chromosomes[0]

def run_multiple_gep(n_runs=10):
    all_best_r2 = np.zeros((n_runs, cfg.MAX_GENERATIONS))
    all_avg_r2 = np.zeros((n_runs, cfg.MAX_GENERATIONS))
    all_best_val_r2 = np.zeros((n_runs, cfg.MAX_GENERATIONS))
    all_avg_val_r2 = np.zeros((n_runs, cfg.MAX_GENERATIONS))
    
    best_overall_chromosome = None
    best_overall_val_r2 = -float('inf')
    
    for run in range(n_runs):
        print(f"Starting run {run + 1}/{n_runs}")
        population = GEPPopulation(cfg.POPULATION_SIZE, cfg.HEAD_LENGTH)
        best_r2_history = []
        avg_r2_history = []
        best_val_r2_history = []
        avg_val_r2_history = []
        
        for generation in range(cfg.MAX_GENERATIONS):
            population.evaluate_population(data.X_train, data.y_train, data.X_val, data.y_val)
            
            # Track metrics
            best_r2_history.append(population.chromosomes[0].r2_score)
            best_val_r2_history.append(population.chromosomes[0].val_r2_score)
            valid_r2_scores = [chrom.r2_score for chrom in population.chromosomes if 0 <= chrom.r2_score <= 1]
            valid_val_r2_scores = [chrom.val_r2_score for chrom in population.chromosomes if 0 <= chrom.val_r2_score <= 1]
            avg_r2 = np.mean(valid_r2_scores) if valid_r2_scores else 0
            avg_val_r2 = np.mean(valid_val_r2_scores) if valid_val_r2_scores else 0
            avg_r2_history.append(avg_r2)
            avg_val_r2_history.append(avg_val_r2)
            # Update best overall chromosome based on validation R²
            if population.chromosomes[0].val_r2_score > best_overall_val_r2:
                best_overall_val_r2 = population.chromosomes[0].val_r2_score
                best_overall_chromosome = population.chromosomes[0]
            
            if generation % 10 == 0:
                print(f"Run {run + 1}, Gen {generation}")
                print(f"Train - Best R²: {population.chromosomes[0].r2_score:.4f}, Avg R²: {avg_r2:.4f}")
                print(f"Val   - Best R²: {population.chromosomes[0].val_r2_score:.4f}, Avg R²: {avg_val_r2:.4f}")
            
            new_population = []
            new_population.append(population.chromosomes[0])
            
            while len(new_population) < cfg.POPULATION_SIZE:
                parent1 = population.selection()
                parent2 = population.selection()

                if np.random.random() < cfg.CROSSOVER_ONE_RATE:
                    parent1, parent2 = recombination(parent1, parent2, 'one-point')
                if np.random.random() < cfg.CROSSOVER_TWO_RATE:
                    parent1, parent2 = recombination(parent1, parent2, 'two-point')
                if np.random.random() < cfg.CROSSOVER_GENE_RATE:
                    parent1, parent2 = recombination(parent1, parent2, 'gene')

                if np.random.random() < cfg.MUTATION_RATE:
                    parent1 = mutation(parent1)
                if np.random.random() < cfg.MUTATION_RATE:
                    parent2 = mutation(parent2)

                if np.random.random() < cfg.TRANSPOSITION_IS_RATE:
                    parent1 = is_transposition(parent1)
                if np.random.random() < cfg.TRANSPOSITION_IS_RATE:
                    parent2 = is_transposition(parent2)
                
                new_population.extend([parent1, parent2])
            
            population.chromosomes = new_population[:cfg.POPULATION_SIZE]
        
        all_best_r2[run] = best_r2_history
        all_avg_r2[run] = avg_r2_history
        all_best_val_r2[run] = best_val_r2_history
        all_avg_val_r2[run] = avg_val_r2_history
    
    # Plot training and validation results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    generations = np.arange(cfg.MAX_GENERATIONS)
    
    # Training plot
    mean_best_r2 = np.mean(all_best_r2, axis=0)
    mean_avg_r2 = np.mean(all_avg_r2, axis=0)
    std_best_r2 = np.std(all_best_r2, axis=0)
    std_avg_r2 = np.std(all_avg_r2, axis=0)
    
    ax1.plot(generations, mean_best_r2, 'b-', label='Best R² (mean)')
    ax1.fill_between(generations, 
                    mean_best_r2 - std_best_r2, 
                    mean_best_r2 + std_best_r2, 
                    alpha=0.2, color='b')
    ax1.plot(generations, mean_avg_r2, 'r-', label='Average R² (mean)')
    ax1.fill_between(generations, 
                    mean_avg_r2 - std_avg_r2, 
                    mean_avg_r2 + std_avg_r2, 
                    alpha=0.2, color='r')
    ax1.set_title('Training R² Score Evolution')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('R² Score')
    ax1.grid(True)
    ax1.legend()
    
    # Validation plot
    mean_best_val_r2 = np.mean(all_best_val_r2, axis=0)
    mean_avg_val_r2 = np.mean(all_avg_val_r2, axis=0)
    std_best_val_r2 = np.std(all_best_val_r2, axis=0)
    std_avg_val_r2 = np.std(all_avg_val_r2, axis=0)
    
    ax2.plot(generations, mean_best_val_r2, 'b-', label='Best Val R² (mean)')
    ax2.fill_between(generations, 
                    mean_best_val_r2 - std_best_val_r2, 
                    mean_best_val_r2 + std_best_val_r2, 
                    alpha=0.2, color='b')
    ax2.plot(generations, mean_avg_val_r2, 'r-', label='Average Val R² (mean)')
    ax2.fill_between(generations, 
                    mean_avg_val_r2 - std_avg_val_r2, 
                    mean_avg_val_r2 + std_avg_val_r2, 
                    alpha=0.2, color='r')
    ax2.set_title('Validation R² Score Evolution')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('R² Score')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('r2_history_train_val.png')
    plt.close()
    
    return (mean_best_r2[-1], mean_avg_r2[-1], 
            mean_best_val_r2[-1], mean_avg_val_r2[-1], 
            best_overall_chromosome)


# Run the algorithm
best_solution = run_gep()
print(f"Final best fitness: {best_solution.fitness}")

# To run the multiple trials:
best_train_r2, avg_train_r2, best_val_r2, avg_val_r2, best_model = run_multiple_gep(n_runs=100)
print(f"Final average best train R²: {best_train_r2:.4f}")
print(f"Final average mean train R²: {avg_train_r2:.4f}")
print(f"Final average best val R²: {best_val_r2:.4f}")
print(f"Final average mean val R²: {avg_val_r2:.4f}")
print(f"Best model validation R²: {best_model.val_r2_score:.4f}")

# After running multiple GEP trials, evaluate the best model
evaluate_and_plot_best_model(best_model, data.test_data, data.X_train, data.y_train,data.X_val, data.y_val)