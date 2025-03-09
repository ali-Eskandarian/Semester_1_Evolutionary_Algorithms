import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_expression_tree(node, ax=None, x=0, y=0, dx=1.0, level=0):
    """Plot the expression tree using matplotlib"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_axis_off()
        plot_expression_tree(node, ax, 0, 0, 1.0, 0)
        ax.set_aspect('equal')
        plt.savefig('best_model_tree.png', bbox_inches='tight')
        plt.close()
        return

    circle_radius = 0.3 
    circle = plt.Circle((x, -y), circle_radius, fill=False)
    ax.add_patch(circle)
    ax.text(x, -y, str(node.value), ha='center', va='center', fontsize=20)  
    
    if node.left:
        left_x = x - dx
        left_y = y + 1.5  

        ax.plot([x, left_x], [-y, -left_y], 'k-', linewidth=2)  
        plot_expression_tree(node.left, ax, left_x, left_y, dx/1.5, level+1) 
    
    if node.right:
        right_x = x + dx
        right_y = y + 1.5 

        ax.plot([x, right_x], [-y, -right_y], 'k-', linewidth=2)  
        plot_expression_tree(node.right, ax, right_x, right_y, dx/1.5, level+1)  

def evaluate_and_plot_best_model(best_model, test_data, X_train, y_train,X_val, y_val):
    """Evaluate best model on test data and create visualizations"""
    plot_expression_tree(best_model.expr_tree)
    
    X_test = test_data["x"].to_numpy()
    test_predictions = np.array([best_model.evaluate(x) for x in X_test])
    
    test_results = pd.DataFrame({
        'x': X_test,
        'y_predicted': test_predictions
    })
    test_results.to_csv('test_predicted.csv', index=False)
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
    plt.scatter(X_val, y_val, color='green', alpha=0.5, label='Validation Data')
    
    sort_idx = np.argsort(X_test)
    plt.plot(X_test[sort_idx], test_predictions[sort_idx], 'r-', label='Model Predictions', linewidth=2)
    plt.title('Best Model Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('best_model_predictions.png')
    plt.close()
    
    print("\nBest Model Performance:")
    print(f"Training R²: {best_model.r2_score:.4f}")
    print(f"Validation R²: {best_model.val_r2_score:.4f}")
    print(f"Expression: {best_model._parse_expression()}")
