import numpy as np
from pymoo.problems import get_problem

class WFGEvaluator:
    def __init__(self, problem_name, n_var, M):
        """
        Initialize WFG problem evaluator
        
        Args:
            problem_name (str): Name of WFG problem (e.g., 'wfg1', 'wfg2', ..., 'wfg9')
            n_var (int): Number of decision variables
            M (int): Number of objectives
        """
        self.problem_name = problem_name
        self.n_var = n_var
        self.M = M
        
        # Load the specific WFG problem from pymoo
        self.problem = get_problem(problem_name, n_var=n_var, n_obj=M)
    
    def evaluate(self, x):
        """
        Evaluate the solution using the WFG problem
        
        Args:
            x (np.ndarray): Decision variables
        
        Returns:
            np.ndarray: Objective function values
        """
        return self.problem.evaluate(x)
    
    def get_true_pareto_front(self):
        """
        Get the true Pareto front for the problem
        
        Returns:
            np.ndarray: True Pareto front points
        """
        return self.problem.pareto_front()
    
    def get_bounds(self):
        """
        Get the lower and upper bounds of decision variables
        
        Returns:
            tuple: (lower bounds, upper bounds)
        """
        return self.problem.xl, self.problem.xu 