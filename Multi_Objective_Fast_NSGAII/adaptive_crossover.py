import numpy as np
from .crossover_operators import CrossoverOperators

class AdaptiveCrossover:
    def __init__(self, crossover_operators):
        """
        Initialize adaptive crossover with multiple operators
        
        Args:
            crossover_operators (list): List of crossover methods
        """
        self.operators = crossover_operators
        self.operator_performance = np.ones(len(crossover_operators))
        self.operator_usage = np.zeros(len(crossover_operators))
    
    def select_crossover_operator(self):
        """
        Adaptively select a crossover operator based on past performance
        
        Returns:
            callable: Selected crossover operator
        """
        # Normalize performance scores
        probabilities = self.operator_performance / np.sum(self.operator_performance)
        
        # Select operator probabilistically
        selected_index = np.random.choice(len(self.operators), p=probabilities)
        self.operator_usage[selected_index] += 1
        
        return self.operators[selected_index]
    
    def update_performance(self, operator_index, performance_score):
        """
        Update performance of a crossover operator
        
        Args:
            operator_index (int): Index of the operator
            performance_score (float): Performance metric
        """
        # Exponential moving average for performance tracking
        self.operator_performance[operator_index] = (
            0.7 * self.operator_performance[operator_index] + 
            0.3 * performance_score
        )
    
    def get_operator_statistics(self):
        """
        Get statistics of crossover operator performance
        
        Returns:
            dict: Performance and usage statistics
        """
        return {
            'performance': self.operator_performance,
            'usage': self.operator_usage
        }

# Example usage
def create_adaptive_crossover():
    crossover_operators = [
        CrossoverOperators.single_point_crossover,
        CrossoverOperators.two_point_crossover,
        CrossoverOperators.uniform_crossover,
        CrossoverOperators.blend_crossover
    ]
    return AdaptiveCrossover(crossover_operators) 