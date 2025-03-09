from config import DataClass as data
from config import GEPConfig as cfg
import numpy as np

class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        if self.value in cfg.TERMINALS:
            return x if self.value == 'x' else float(self.value)
        elif self.value in cfg.FUNCTIONS:
            left_val = self.left.evaluate(x) if self.left else 0
            right_val = self.right.evaluate(x) if self.right else 0
            if   self.value == '+':
                return left_val + right_val
            elif self.value == '-':
                return left_val - right_val
            elif self.value == '*':
                return left_val * right_val
            elif self.value == '/':
                return left_val / right_val if right_val != 0 else np.inf
            elif self.value == 'sqrt':
                return np.sqrt(abs(left_val))
            elif self.value == 'sin':
                return np.sin(left_val)
            elif self.value == 'cos':
                return np.cos(left_val)
            elif self.value == 'exp':
                return np.exp(left_val) if left_val < 100 else np.inf  # Prevent overflow
        return 0

class GEPChromosome:
    def __init__(self, head_length):
        self.head_length = head_length
        self.tail_length = head_length * (cfg.MAX_ARITY - 1) + 1
        self.length = self.head_length + self.tail_length
        
        
        max_attempts = 25  
        self.fitness = float('inf')
        attempt = 0
        
        while self.fitness > 25 and attempt < max_attempts:
            self.genes = self._initialize_genes()
            self.expr_tree = self._build_expression_tree()
            
            try:
                predictions = np.array([self.evaluate(x) for x in data.X_train])
                self.fitness = np.mean((predictions - data.y_train) ** 2)
            except:
                self.fitness = float('inf')
            attempt += 1
        
        
        self.val_fitness = float('inf')
        self.r2_score = -float('inf')
        self.val_r2_score = -float('inf')

    def _initialize_genes(self):
        head = np.random.choice(cfg.FUNCTIONS + cfg.TERMINALS, size=self.head_length)
        tail = np.random.choice(cfg.TERMINALS, size=self.tail_length)
        return np.concatenate([head, tail])

    def _build_expression_tree(self):
        stack = []
        for gene in reversed(self.genes):
            if gene in cfg.TERMINALS:
                stack.append(TreeNode(gene))
            elif gene in cfg.FUNCTIONS:
                if len(stack) < 2:
                    continue
                right = stack.pop()
                left = stack.pop()
                stack.append(TreeNode(gene, left, right))
        return stack[-1] if stack else TreeNode('0')

    def evaluate(self, x):
        try:
            return self.expr_tree.evaluate(x)
        except:
            return np.inf

    def calculate_fitness(self, X, y, X_val=None, y_val=None):
        
        train_predictions = np.array([self.evaluate(x) for x in X])
        self.fitness = np.mean((train_predictions - y) ** 2)
        
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - train_predictions) ** 2)
        self.r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else -float('inf')
        
        
        if X_val is not None and y_val is not None:
            val_predictions = np.array([self.evaluate(x) for x in X_val])
            self.val_fitness = np.mean((val_predictions - y_val) ** 2)
            
            val_y_mean = np.mean(y_val)
            val_ss_tot = np.sum((y_val - val_y_mean) ** 2)
            val_ss_res = np.sum((y_val - val_predictions) ** 2)
            self.val_r2_score = 1 - (val_ss_res / val_ss_tot) if val_ss_tot != 0 else -float('inf')
        
        return self.fitness

    def _parse_expression(self):
        """Returns a string representation of the expression tree"""
        def _recursive_parse(node):
            if node is None:
                return ""
            if node.value in cfg.TERMINALS:
                return str(node.value)
            return f"({node.value} {_recursive_parse(node.left)} {_recursive_parse(node.right)})"
        
        return _recursive_parse(self.expr_tree)

class GEPPopulation:
    def __init__(self, size, head_length):
        self.size = size
        self.chromosomes = [GEPChromosome(head_length) for _ in range(size)]

    def evaluate_population(self, X, y, X_val=None, y_val=None):
        for chromosome in self.chromosomes:
            chromosome.calculate_fitness(X, y, X_val, y_val)
        self.chromosomes.sort(key=lambda x: x.fitness)

    def selection(self):
        tournament_size = 5
        tournament = np.random.choice(self.chromosomes, tournament_size)
        return min(tournament, key=lambda x: x.fitness)
