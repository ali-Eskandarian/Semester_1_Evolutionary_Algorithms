from models import *
import numpy as np
from config import GEPConfig as cfg

def mutation(chromosome):
    new_chromosome = GEPChromosome(chromosome.head_length)
    new_chromosome.genes = chromosome.genes.copy()
    
    mutation_points = np.random.choice(len(new_chromosome.genes), size=2, replace=False)
    
    for point in mutation_points:
        if point < chromosome.head_length:
            new_chromosome.genes[point] = np.random.choice(cfg.FUNCTIONS + cfg.TERMINALS)
        else:
            new_chromosome.genes[point] = np.random.choice(cfg.TERMINALS)
    
    return new_chromosome

def recombination(parent1, parent2, recomb_type = 'gene'):
    
    if recomb_type == 'one-point':
        point = np.random.randint(0, len(parent1.genes))
        child1_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        child2_genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
    
    elif recomb_type == 'two-point':
        points = sorted(np.random.choice(len(parent1.genes), size=2, replace=False))
        child1_genes = np.concatenate([
            parent1.genes[:points[0]],
            parent2.genes[points[0]:points[1]],
            parent1.genes[points[1]:]
        ])
        child2_genes = np.concatenate([
            parent2.genes[:points[0]],
            parent1.genes[points[0]:points[1]],
            parent2.genes[points[1]:]
        ])
    
    else:  
        gene_length = parent1.head_length + parent1.tail_length
        gene_to_exchange = np.random.randint(0, len(parent1.genes) // gene_length)
        start = gene_to_exchange * gene_length
        end = start + gene_length
        
        child1_genes = parent1.genes.copy()
        child2_genes = parent2.genes.copy()
        child1_genes[start:end] = parent2.genes[start:end]
        child2_genes[start:end] = parent1.genes[start:end]
    
    child1 = GEPChromosome(parent1.head_length)
    child2 = GEPChromosome(parent1.head_length)
    child1.genes = child1_genes
    child2.genes = child2_genes
    return child1, child2

def is_transposition(chromosome):
    """Implements insertion sequence (IS) elements transposition"""
    new_chromosome = GEPChromosome(chromosome.head_length)
    new_chromosome.genes = chromosome.genes.copy()
    
    is_length = np.random.randint(1, 4)
    
    is_start = np.random.randint(0, len(chromosome.genes) - is_length)
    
    is_element = chromosome.genes[is_start:is_start + is_length]
    
    gene_length = chromosome.head_length + chromosome.tail_length
    target_gene = np.random.randint(0, len(chromosome.genes) // gene_length)
    gene_start = target_gene * gene_length
    
    target_pos = gene_start + np.random.randint(1, chromosome.head_length)
    
    head_end = gene_start + chromosome.head_length
    
    new_sequence = np.concatenate([
        new_chromosome.genes[:target_pos],
        is_element,
        new_chromosome.genes[target_pos:head_end-is_length]
    ])
    
    if head_end < len(new_chromosome.genes):
        new_sequence = np.concatenate([
            new_sequence,
            new_chromosome.genes[head_end:]
        ])
    
    new_chromosome.genes = new_sequence
    return new_chromosome
