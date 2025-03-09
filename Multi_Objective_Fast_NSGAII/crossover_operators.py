import numpy as np

class CrossoverOperators:
    @staticmethod
    def single_point_crossover(parent1, parent2):
        """
        Single point crossover
        
        Args:
            parent1 (np.ndarray): First parent
            parent2 (np.ndarray): Second parent
        
        Returns:
            tuple: Two offspring
        """
        crossover_point = np.random.randint(len(parent1))
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2
    
    @staticmethod
    def two_point_crossover(parent1, parent2):
        """
        Two point crossover
        
        Args:
            parent1 (np.ndarray): First parent
            parent2 (np.ndarray): Second parent
        
        Returns:
            tuple: Two offspring
        """
        points = np.sort(np.random.choice(len(parent1), 2, replace=False))
        offspring1 = np.copy(parent1)
        offspring2 = np.copy(parent2)
        offspring1[points[0]:points[1]] = parent2[points[0]:points[1]]
        offspring2[points[0]:points[1]] = parent1[points[0]:points[1]]
        return offspring1, offspring2
    
    @staticmethod
    def uniform_crossover(parent1, parent2, prob=0.5):
        """
        Uniform crossover
        
        Args:
            parent1 (np.ndarray): First parent
            parent2 (np.ndarray): Second parent
            prob (float): Probability of swapping genes
        
        Returns:
            tuple: Two offspring
        """
        mask = np.random.random(len(parent1)) < prob
        offspring1 = np.where(mask, parent2, parent1)
        offspring2 = np.where(mask, parent1, parent2)
        return offspring1, offspring2
    
    @staticmethod
    def blend_crossover(parent1, parent2, alpha=0.5):
        """
        Blend crossover
        
        Args:
            parent1 (np.ndarray): First parent
            parent2 (np.ndarray): Second parent
            alpha (float): Blend factor
        
        Returns:
            tuple: Two offspring
        """
        gamma = (1 + 2 * alpha) * np.random.random(len(parent1)) - alpha
        offspring1 = parent1 + gamma * (parent2 - parent1)
        offspring2 = parent2 + gamma * (parent1 - parent2)
        return offspring1, offspring2 