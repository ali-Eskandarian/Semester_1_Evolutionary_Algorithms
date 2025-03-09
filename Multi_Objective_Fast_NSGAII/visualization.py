import matplotlib.pyplot as plt
import numpy as np

class ParetoFrontVisualization:
    @staticmethod
    def plot_pareto_fronts(obtained_front, true_front, problem_name, save_path=None):
        """
        Plot obtained and true Pareto fronts
        
        Args:
            obtained_front (np.ndarray): Obtained Pareto front
            true_front (np.ndarray): True Pareto front
            problem_name (str): Name of the problem
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot true Pareto front
        plt.scatter(
            true_front[:, 0], 
            true_front[:, 1] if true_front.shape[1] > 1 else np.zeros_like(true_front[:, 0]), 
            label='True Pareto Front', 
            color='red', 
            marker='x'
        )
        
        # Plot obtained Pareto front
        plt.scatter(
            obtained_front[:, 0], 
            obtained_front[:, 1] if obtained_front.shape[1] > 1 else np.zeros_like(obtained_front[:, 0]), 
            label='Obtained Pareto Front', 
            color='blue', 
            alpha=0.7
        )
        
        plt.title(f'Pareto Front Comparison - {problem_name.upper()}')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2' if obtained_front.shape[1] > 1 else 'Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close() 