import pandas as pd
from sklearn.model_selection import train_test_split

class GEPConfig:
    # Population parameters
    POPULATION_SIZE = 30
    HEAD_LENGTH = 8
    MAX_GENERATIONS = 50
    
    # Genetic operator rates
    MUTATION_RATE = 0.051
    TRANSPOSITION_IS_RATE = 0.1
    TRANSPOSITION_RIS_RATE = 0.1
    TRANSPOSITION_GENE_RATE = 0.1
    
    CROSSOVER_ONE_RATE = 0.2
    CROSSOVER_TWO_RATE = 0.5
    CROSSOVER_GENE_RATE = 0.1
    
    # Function and terminal sets
    FUNCTIONS = ['+', '-', '*', '/', 'sqrt'] + 2* ['cos']+ 2* ['sin']+ 2* ['exp']
    TERMINALS = ['x'] * 8 + ['3', '2', '0.25', '0.1']
    MAX_ARITY = 2 
    
    # Initialization parameters
    MAX_FITNESS_INIT = 50
    MAX_INIT_ATTEMPTS = 10
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Data split ratio
    TEST_SIZE = 0.2

class DataClass:
    train_data = pd.read_csv("./training_data.csv")
    test_data  = pd.read_csv("./test_data.csv")

    X_full = train_data["x"].to_numpy()
    y_full = train_data["y"].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

