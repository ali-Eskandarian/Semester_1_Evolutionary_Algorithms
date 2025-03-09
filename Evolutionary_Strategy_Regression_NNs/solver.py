from typing import List, Tuple, Dict
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from copy import deepcopy

class Individual:
    def __init__(self, architecture: List[int], X: np.ndarray, Y: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, activation: str = 'relu', type_: str = 'accuracy'):
        """
        Initialize an individual representing MLP parameters
        
        Args:
            architecture: List of integers representing layer sizes [input_dim, hidden1, ..., output_dim]
            activation: Activation function to use ('relu' or 'tanh' or 'sigmoid')
            X: Input data for training
            Y: Target data for training
        """
        self.architecture = architecture
        self.activation = activation
        self.X = X
        self.X_val = X_val
        self.Y = Y
        self.Y_val = Y_val
        self.type = type_
        
        # Initialize weights and biases for each layer
        self.params = {}
        for i in range(len(architecture) - 1):

            # std = np.sqrt(2.0 / architecture[i])
            self.params[f'W{i}'] = np.random.randn(architecture[i], architecture[i+1])
            self.params[f'b{i}'] = np.ones(architecture[i+1])
        
        # Calculate total number of parameters
        self.n_params = sum(w.size + b.size for w, b in 
                          zip(self.get_weights(), self.get_biases()))
        
        # Initial sigma & alpha
        self.sigma, self.alpha = {}, {}
        for i in range(self.n_params):
            self.sigma[f's{i}'] = np.random.randn()
        
        for i in range(self.n_params):
            for j in range(i+1, self.n_params):
                self.alpha[f'a{i}{j}'] = np.random.randn()

        self.fitness = self._fitness()
    
    def get_weights(self) -> List[np.ndarray]:
        """Get all weight matrices"""
        return [self.params[f'W{i}'] for i in range(len(self.architecture)-1)]
    
    def get_biases(self) -> List[np.ndarray]:
        """Get all bias vectors"""
        return [self.params[f'b{i}'] for i in range(len(self.architecture)-1)]
    
    def get_alpha(self) -> List[np.ndarray]:
        """Get alpha vectors"""
        return [self.alpha[f'a{i}'] for i in range(self.n_params)]
    
    def get_sigma(self) -> List[np.ndarray]:
        """Get sigma vectors"""
        return [self.sigma[f's{i}'] for i in range(self.n_params)]
    
    def to_vector(self) -> np.ndarray:
        """Convert all parameters to a single vector"""
        return np.concatenate([p.flatten() for p in self.params.values()])
    
    def from_vector(self, vector: np.ndarray):
        """Update parameters from a vector"""
        start = 0
        for key in self.params:
            shape = self.params[key].shape
            size = self.params[key].size
            self.params[key] = vector[start:start+size].reshape(shape)
            start += size
    
    def activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        current = X
        
        # Hidden layers with activation
        for i in range(len(self.architecture) - 2):
            current = np.dot(current, self.params[f'W{i}']) + self.params[f'b{i}']
            current = self.activate(current)
        
        # Output layer with softmax
        logits = np.dot(current, self.params[f'W{len(self.architecture)-2}']) + \
                self.params[f'b{len(self.architecture)-2}']
        
        # Softmax function from scratch
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax = exps / np.sum(exps, axis=1, keepdims=True)

        return softmax
    
    def _fitness(self) -> float:
        """
        Evaluate individual's fitness using cross-entropy loss or 1-accuracy on X_train
        """
        if self.type == 'loss':
            predictions = self.forward(self.X)
            
            # Convert y to one-hot if needed
            if len(self.Y.shape) == 1:
                y_one_hot = np.zeros((len(self.Y), predictions.shape[1]))
                y_one_hot[np.arange(len(self.Y)), self.Y.astype(int)] = 1
                y = y_one_hot
            class_weights = np.array([1.0 / len(np.where(self.Y == i)[0]) for i in np.unique(self.Y)])
            # Calculate cross-entropy loss
            epsilon = 1e-15  # Small constant to avoid log(0)
            predictions = np.clip(class_weights * predictions, epsilon, 1 - epsilon)
            loss =  -np.mean(np.sum(y * np.log(predictions), axis=1))
            
            self.fitness = -loss

        elif self.type == 'accuracy':
            predictions_train = self.forward(self.X)
            y_pred_train = np.argmax(predictions_train, axis=1)
            accuracy_train = np.mean(y_pred_train == self.Y)
            self.fitness = accuracy_train
        else:
            raise ValueError(f"Invalid fitness type: {type}. Expected 'loss' or 'accuracy'.")
        
        return self.fitness
    

class EvolutionStrategy:
    def __init__(self, 
                 mu: int,                        # Parent population size 
                 lambda_: int,                   # Offspring population size
                 architecture: List[int],        # Neural network architecture
                 X: np.ndarray,                                     
                 Y: np.ndarray,
                 Correlated:bool = False,
                 select_plus:bool = False,
                 class_weight:bool = False,
                 p_c: float = 0.1,               # probability of recombination
                 activation: str = 'relu',
                 type_: str = 'accuracy'
                 ):      # Activation function
        
        self.mu = mu
        self.lambda_ = lambda_
        self.architecture = architecture
        self.p_c = p_c
        self.activation = activation
        self.X = X
        self.Y = Y
        self.corr = Correlated
        self.select_plus = select_plus
        self.type = type_
        self.class_weight = class_weight

        # Calculate class weights for balanced sampling
        self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
        sample_weights = self.class_weights[Y.astype(int)]

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.population = []

        # Store dimensionality of search space
        self.n_dims = 10
    
    def evolve(self, n_generations: int) -> Tuple[List[float], List[float], List[float], Individual]:
        """
        Main evolution loop with accuracy tracking
        
        Returns:
            Tuple containing:
            - fitness_history: List of best fitness values per generation
            - train_acc_history: List of training accuracies
            - val_acc_history: List of validation accuracies
            - best_individual: Best individual found across all generations
        """
        fitness_history = []
        avg_fitness_history = []
        train_acc_history = []
        val_acc_history = []
        
        
        # Track best solution across all generations
        best_individual_overall = Individual(self.architecture, self.X_train, self.y_train, self.X_val, self.y_val, self.activation, self.type)
        best_fitness_overall    = float('-inf')
        
        self.population = [
            Individual(self.architecture, self.X_train, self.y_train, self.X_val, self.y_val, self.activation, self.type) 
            for _ in range(self.mu)
        ]

        self.n_dims = self.population[0].n_params
        for gen in range(n_generations):
            print(f"=== Generation {gen} ===")
            offspring       = self._generate_offspring(self.X_train, self.y_train)
            self.population = self._select(offspring)
            
            # Get best individual and calculate average fitness
            best_individual = max(self.population, key=lambda x: x.fitness)
            best_fitness    = best_individual.fitness
            avg_fitness     = np.mean([ind.fitness for ind in self.population])
            avg_sigma = np.mean([np.mean(list(ind.sigma.values())) for ind in self.population])
            print(f"Average Sigma: {avg_sigma:.4f}")
            
            # Update best overall individual if necessary 
            if best_fitness > best_fitness_overall:
                best_fitness_overall            = best_fitness
                best_individual_overall         = Individual(self.architecture, self.X_train, self.y_train, self.X_val, self.y_val, self.activation, self.type) 
                best_individual_overall.params  = {k: v.copy() for k, v in best_individual.params.items()}
                best_individual_overall.sigma   = best_individual.sigma.copy()
                best_individual_overall.alpha   = best_individual.alpha.copy()
                best_individual_overall.fitness = best_individual.fitness
            
            # Calculate accuracies
            train_pred = best_individual.forward(self.X_train)
            train_labels = np.argmax(train_pred, axis=1)
            train_acc = np.mean(train_labels == self.y_train)
            
            val_pred = best_individual.forward(self.X_val)
            val_labels = np.argmax(val_pred, axis=1)
            val_acc = np.mean(val_labels == self.y_val)
            
            # Store histories
            fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)  # Store average fitness
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            print(f"=== Best Fitness: {best_fitness:.4f}, Avg Fitness: {avg_fitness:.4f}")#, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} ===")

         # Evaluate on test set
        print(f"Test set fitness: {best_individual.fitness}")
        
        # Make predictions
        val_predictions = best_individual.forward(self.X_val)
        predicted_labels = np.argmax(val_predictions, axis=1)
        test_accuracy = np.mean(predicted_labels == self.y_val)
        print(f"val final accuracy: {test_accuracy:.4f}")
        
        # Calculate other metrics
        confusion = confusion_matrix(self.y_val, predicted_labels)
        print("Confusion Matrix:\n", confusion)
        print("Classification Report:\n", classification_report(self.y_val, predicted_labels))
        precision = precision_score(self.y_val, predicted_labels, average='weighted')
        recall = recall_score(self.y_val, predicted_labels, average='weighted')
        f1 = f1_score(self.y_val, predicted_labels, average='weighted')
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        
        # Plot fitness history
        
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, label='Best Fitness')
        plt.plot(avg_fitness_history, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution Progress')
        plt.legend()
        plt.savefig('evolution_progress.png')
        plt.show()
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion, interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.savefig('confusion_matrix.png')  # Save the figure
        plt.show()

        return fitness_history, train_acc_history, val_acc_history, best_individual_overall
    
    def _generate_offspring(self,X_train, y_train) -> List[Individual]:
        """
        Generate lambda offspring using recombination and correlated Gaussian mutation
        """

        offspring = []
        for _ in range(self.lambda_):

            # Select parent
            parent = np.random.choice(self.population)
            child = Individual(self.architecture, X_train, y_train,  self.X_val, self.y_val, self.activation, self.type)
            
            # Perform arithmetic recombination with probability p_c
            if np.random.random() < self.p_c:
                # Select second parent
                parent2 = np.random.choice(self.population)
                # Perform whole arithmetic recombination with α = 0.5
                parent_params = 0.5 * (parent.to_vector() + parent2.to_vector())
                
                # Recombine strategy parameters (sigmas and alphas)
                for i in range(self.n_dims):
                    child.sigma[f's{i}'] = 0.5 * (parent.sigma[f's{i}'] + parent2.sigma[f's{i}'])
                
                for i in range(self.n_dims):
                    for j in range(i+1, self.n_dims):
                        key = f'a{i}{j}'
                        # Special handling for angles to avoid discontinuity at ±π
                        angle1, angle2 = parent.alpha[key], parent2.alpha[key]
                        # Ensure angles are in [-π, π]
                        if abs(angle1 - angle2) > np.pi:
                            if angle1 > angle2:
                                angle2 += 2 * np.pi
                            else:
                                angle1 += 2 * np.pi
                        child.alpha[key] = 0.5 * (angle1 + angle2)
                        if child.alpha[key] > np.pi:
                            child.alpha[key] -= 2 * np.pi
            else:
                parent_params = parent.to_vector()
                # Copy strategy parameters from parent
                child.sigma = parent.sigma.copy()
                child.alpha = parent.alpha.copy()
            
            # Update strategy parameters (sigmas and alphas)
            tau = 1 / np.sqrt(2.0 * self.n_dims)  # Global learning rate
            tau_prime = 1 / np.sqrt(2.0 * np.sqrt(self.n_dims))  # Individual learning rate
            beta = np.deg2rad(5)  # ~5 degrees for angle updates
            
            # Update sigmas using log-normal distribution
            global_factor = np.exp(tau_prime * np.random.randn())
            eps = 1e-3
            eps_up = 2
            Ni = []
            for i in range(self.n_dims):
                ni = np.random.randn()
                Ni.append(ni)
                child.sigma[f's{i}'] = parent.sigma[f's{i}'] * global_factor * \
                                    np.exp(tau *ni )
                if child.sigma[f's{i}'] < eps:
                    child.sigma[f's{i}'] = eps
                if child.sigma[f's{i}'] > eps_up:
                    child.sigma[f's{i}'] = eps_up
            if self.corr:
                # Update rotation angles (vectorized form)
                i, j = np.triu_indices(self.n_dims, k=1)  # Get upper triangular indices
                keys = [f'a{i_val}{j_val}' for i_val, j_val in zip(i, j)]
                angles = np.array([parent.alpha[key] for key in keys])

                # Generate random perturbations for all angles at once
                perturbations = beta   * np.random.randn(len(keys))
                new_angles    = angles + perturbations

                # Keep angles in [-π, π] (vectorized)
                new_angles = new_angles - 2 * np.pi * np.sign(new_angles) * (np.abs(new_angles) > np.pi)

                # Update child's alpha dictionary
                for key, angle in zip(keys, new_angles):
                    child.alpha[key] = angle
                
                # Construct covariance matrix (vectorized form)
                # Initialize covariance matrix
                C = np.zeros((self.n_dims, self.n_dims))

                # Set diagonal elements (variances)
                sigma_values = np.array([child.sigma[f's{i}'] for i in range(self.n_dims)])
                np.fill_diagonal(C, sigma_values**2)

                # Get upper triangular indices
                i, j = np.triu_indices(self.n_dims, k=1)

                # Get sigma values for all pairs
                sigma_i = sigma_values[i]
                sigma_j = sigma_values[j]

                # Get alpha values for all pairs
                alpha_ij = np.array([child.alpha[f'a{i_val}{j_val}'] for i_val, j_val in zip(i, j)])

                # Calculate covariances for all pairs at once
                cij = 0.5 * (sigma_i**2 - sigma_j**2) * np.tan(2 * alpha_ij)

                # Fill upper and lower triangular parts simultaneously
                C[i, j] = cij
                C[j, i] = cij
                
                # Generate correlated mutation
                mutation     = np.random.multivariate_normal(np.zeros(self.n_dims), C)
                child_params = parent_params + mutation
            else:
                child_params = []
                for i, param in enumerate(parent_params):
                    mutation     =  child.sigma[f's{i}'] * Ni[i]
                    child_params.append(param + mutation)
            # Update child parameters
            child.from_vector(np.array(child_params))
            offspring.append(child)
        
            
        return offspring
    
    def _select(self, offspring: List[Individual]) -> List[Individual]:
        """
        (μ,λ) selection - select best μ individuals from offspring
        (μ+λ) selection - select best μ individuals from offspring + parents
        """
        if self.select_plus:
            offspring.extend(deepcopy(self.population))
        sorted_offspring = sorted(offspring, key=lambda x: x.fitness, reverse=True)
        return sorted_offspring[:self.mu]
    



