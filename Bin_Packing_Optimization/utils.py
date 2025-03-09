from solver import Item, Bin, Individual
from typing import List, Tuple, Dict, Any
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def read_solution(filename: str, bin_capacity: int, all_items: List[Item]) -> List[Bin]:
    with open(filename, 'r') as f:
        num_bins = int(f.readline().strip())
        bins = []
        for _ in range(num_bins):
            items = list(map(int, f.readline().strip().split()))
            bin = Bin(bin_capacity)
            for item_id in items:
                item = next(item for item in all_items if item.id == item_id)
                bin.add_item(item)
            bins.append(bin)
    return bins

def plot_solution(bins: List[Bin], instance_name: str):
    
    # Calculate number of rows and columns needed
    if instance_name=='3':
        num_cols = 20
    else:
        num_cols = 5
    num_rows = (len(bins) + num_cols - 1) // num_cols

    fig = make_subplots(rows=num_rows, cols=num_cols, 
                       subplot_titles=[f'Bin {i}' for i in range(len(bins))])
    
    for i, bin in enumerate(bins):
        row = (i // num_cols) + 1
        col = (i % num_cols) + 1
        y_offset = 0
        
        for item in bin.items:
            fig.add_trace(
                go.Bar(
                    x=[item.size],
                    y=[0],
                    base=[y_offset],
                    orientation='h',
                    text=f'<b>id: {item.id} - size: {item.size}</b>',
                    textposition='inside',
                    textfont=dict(size=14),
                    marker=dict(color='skyblue')
                ),
                row=row, col=col
            )
            y_offset += item.size
        
        fig.update_xaxes(range=[0, bin.capacity], row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)
    
    fig.update_layout(
        height=250 * num_rows,  # Fixed height per row
        width=1250,  # Fixed total width (250px * 5 columns)
        title_text=f'Solution Visualization - Instance {instance_name}',
        title_font=dict(size=20, family="Arial Black"),
        showlegend=False
    )
    
    fig.write_html(f'plots/{instance_name}_solution.html')

def calculate_optimal_bins(weights: List[Item], max_capacity: int) -> int:
    total_weight = sum(item.size for item in weights)
    return (total_weight + max_capacity - 1) // max_capacity  # Ceiling division

def read_instance(filename: str) -> Tuple[List[Item], int]:
    with open(filename, 'r') as f:
        num_items = int(f.readline().strip())
        bin_capacity = int(f.readline().strip())
        items = []
        for i in range(num_items):
            size = int(f.readline().strip())
            items.append(Item(i + 1, size))
    return items, bin_capacity

def write_solution(filename: str, solution: Individual):
    bins: List[Bin] = []
    for gene in solution.chromosome:
        item = solution.items[gene - 1]
        packed = False
        for bin in bins:
            if bin.add_item(item):
                packed = True
                break
        if not packed:
            new_bin = Bin(solution.bin_capacity)
            new_bin.add_item(item)
            bins.append(new_bin)
            
    with open(filename, 'w') as f:
        f.write(f"{len(bins)}\n")
        for bin in bins:
            bin_items = [str(item.id) for item in bin.items]
            f.write(" ".join(bin_items) + "\n")

def create_comparison_plots(results: Dict[str, Dict[str, Any]], instance_name: str):
    os.makedirs(f'plots/instance_{instance_name}', exist_ok=True)
    
    fig_fitness = make_subplots(rows=2, cols=2, 
                               subplot_titles=list(results.keys()))
    
    fig_dist = make_subplots(rows=2, cols=2, 
                            subplot_titles=list(results.keys()))
    
    for idx, (config_name, data) in enumerate(results.items()):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        fig_fitness.add_trace(
            go.Scatter(
                x=[stat['generation'] for stat in data['fitness_history']],
                y=[stat['avg_fitness'] for stat in data['fitness_history']],
                name=f'{config_name} - Avg',
                mode='lines'
            ),
            row=row, col=col
        )
        
        fig_fitness.add_trace(
            go.Scatter(
                x=[stat['generation'] for stat in data['fitness_history']],
                y=[stat['max_fitness'] for stat in data['fitness_history']],
                name=f'{config_name} - Max',
                mode='lines'
            ),
            row=row, col=col
        )
        
        fig_dist.add_trace(
            go.Histogram(
                x=data['initial_fitness'],
                name=f'{config_name} - Initial',
                opacity=0.75,
                nbinsx=5
            ),
            row=row, col=col
        )
        
        fig_dist.add_trace(
            go.Histogram(
                x=data['final_fitness'],
                name=f'{config_name} - Final',
                opacity=0.75,
                nbinsx=5
            ),
            row=row, col=col
        )
    
    fig_fitness.update_layout(
        height=800,
        title_text=f'Fitness Progress Comparison - Instance {instance_name}',
        template='plotly_white'
    )
    
    fig_dist.update_layout(
        height=800,
        title_text=f'Fitness Distribution Comparison - Instance {instance_name}',
        template='plotly_white',
        barmode='overlay'
    )
    
    # Save plots
    fig_fitness.write_html(f'plots/instance_{instance_name}/fitness_comparison.html')
    fig_dist.write_html(f'plots/instance_{instance_name}/distribution_comparison.html')