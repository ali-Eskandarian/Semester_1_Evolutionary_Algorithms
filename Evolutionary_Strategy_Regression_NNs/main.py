import random
from typing import List, Tuple
import numpy as np
from solver import EvolutionStrategy
from utils import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data('Codes/processed_heart.csv')
    # Define network architecture
    input_dim = X.shape[1]
    n_classes = len(np.unique(y))
    architecture = [input_dim, 8,  n_classes]  

    # Initialize ES
    es = EvolutionStrategy(
        mu=10,           # 10 parents
        lambda_=70,      # 70 offspring
        architecture=architecture,
        p_c =0.2,
        X=X,
        Y=y,
        Correlated=False,
        select_plus=True,
        class_weight=False,
        activation='relu',
        type_= 'loss',
        )
    
    # Run evolution
    fitness_history, train_acc_history, val_acc_history, best_solution = es.evolve(n_generations=200)
    
    # Use best_solution instead of getting it from population
    print(f"Best solution fitness: {best_solution.fitness}")
    
    # Make predictions with best solution
    test_predictions = best_solution.forward(X)
    test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == y)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Fitness Progress
    plt.subplot(1, 2, 1)
    plt.plot(fitness_history, 'b-', label='Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution Progress')
    plt.legend()
    
    # Plot 2: Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, 'g-', label='Training Accuracy')
    plt.plot(val_acc_history, 'r-', label='Validation Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Generations')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('evolution_and_accuracy.png')
    plt.show()    

    
   

if __name__ == "__main__":
    main()