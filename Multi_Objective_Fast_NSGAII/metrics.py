import numpy as np
from pymoo.indicators.hv import HV as Hypervolume
from pymoo.indicators.igd import IGD as InvertedGenerationalDistance

class PerformanceMetrics:
    @staticmethod
    def hypervolume(obtained_front, reference_point=None):
        """
        Calculate Hypervolume metric
        
        Args:
            obtained_front (np.ndarray): Obtained Pareto front
            reference_point (np.ndarray, optional): Reference point
        
        Returns:
            float: Hypervolume value
        """
        if reference_point is None:
            # If no reference point, use max values of obtained front
            reference_point = np.max(obtained_front, axis=0) * 1.1
        
        hv = Hypervolume(ref_point=reference_point)
        return hv(obtained_front)
    
    @staticmethod
    def inverted_generational_distance(obtained_front, true_front):
        """
        Calculate Inverted Generational Distance (IGD)
        
        Args:
            obtained_front (np.ndarray): Obtained Pareto front
            true_front (np.ndarray): True Pareto front
        
        Returns:
            float: IGD value
        """
        igd = InvertedGenerationalDistance(true_front)
        return igd(obtained_front) 