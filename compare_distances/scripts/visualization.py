"""
Visualization module for P-median analysis.

This module contains functions for visualizing graphs and analysis results.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def create_node_labels(graph: nx.Graph, layout: Dict, centers: Dict[str, List[str]]) -> List[Dict]:
    """
    Create node labels for visualization, showing only center nodes.
    
    Args:
        graph: NetworkX graph object
        layout: Node layout dictionary
        centers: Dictionary with center nodes and their assigned clients
        
    Returns:
        List of label parameter dictionaries
    """
    new_labels = {}
    for node in graph.nodes():
        if node in centers:
            new_labels[node] = node
        else:
            new_labels[node] = ''
    return [{'G': graph, 'pos': layout, 'labels': new_labels}]


def get_visualization_params(graph: nx.Graph, layout: Dict, node_size_multiplier: int, 
                           centers: Dict[str, List[str]]) -> Dict:
    """
    Get all parameters needed for graph visualization.
    
    Args:
        graph: NetworkX graph object
        layout: Node layout dictionary
        node_size_multiplier: Multiplier for node sizes
        centers: Dictionary with center nodes and their assigned clients
        
    Returns:
        Dictionary with all visualization parameters
    """
    # Use rainbow color map
    rainbow = plt.cm.rainbow
    
    # Assign colors for each center and its served vertices
    color_map = {}
    num_centers = len(centers)
    for idx, center in enumerate(centers):
        color = rainbow(idx / num_centers)  # Color from rainbow based on center index
        for node in centers[center]:
            color_map[node] = color
        color_map[center] = color

    # Create two sets of nodes: centers and non-centers (for different shapes)
    non_center_nodes = [node for node in graph.nodes() if node not in centers]
    center_nodes = [node for node in centers]

    # Parameters for non-central vertices (circles)
    non_center_params = {
        'G': graph,
        'pos': layout,
        'nodelist': non_center_nodes,
        'node_size': [node_size_multiplier for _ in non_center_nodes],
        'node_color': [color_map.get(i, 'grey') for i in non_center_nodes],
        'node_shape': 'o'  # Circles for non-central vertices
    }

    # Parameters for central vertices (squares)
    center_params = {
        'G': graph,
        'pos': layout,
        'nodelist': center_nodes,
        'node_size': [node_size_multiplier * 2 for _ in center_nodes],
        'node_color': [color_map.get(i, 'grey') for i in center_nodes],
        'node_shape': 's'  # Squares for central vertices
    }

    edges_params = {
        'G': graph,
        'pos': layout,
        'width': 0.5,
        'alpha': 1
    }
    
    return {
        'non_center_nodes_params': non_center_params, 
        'center_nodes_params': center_params,
        'edges_params': edges_params, 
        'labels_param_list': create_node_labels(graph, layout, centers)
    }


def draw_graph_with_centers(graph: nx.Graph, centers_dict: Dict[str, List[str]],
                          node_size_multiplier: int = 100, title: str = '',
                          ax: Optional[plt.Axes] = None) -> None:
    """
    Draw graph with highlighted center nodes and color-coded assignments.
    
    Args:
        graph: NetworkX graph object
        centers_dict: Dictionary with center nodes and their assigned clients
        node_size_multiplier: Multiplier for node sizes
        title: Title for the plot
        ax: Optional matplotlib Axes object. If None, creates new figure
    """
    layout = nx.kamada_kawai_layout(graph)
    all_params = get_visualization_params(graph, layout, node_size_multiplier, centers_dict)
    
    # Create figure if ax is not provided
    if ax is None:
        plt.figure(figsize=(8, 5))
        current_ax = plt.gca()
    else:
        current_ax = ax
    
    # Add ax parameter to all drawing functions
    all_params['non_center_nodes_params']['ax'] = current_ax
    all_params['center_nodes_params']['ax'] = current_ax
    all_params['edges_params']['ax'] = current_ax
    
    # Draw non-central nodes (circles)
    nx.draw_networkx_nodes(**all_params['non_center_nodes_params'])
    
    # Draw central nodes (squares)
    nx.draw_networkx_nodes(**all_params['center_nodes_params'])
    
    # Draw edges
    nx.draw_networkx_edges(**all_params['edges_params'])
    
    # Draw labels
    for params in all_params['labels_param_list']:
        params['ax'] = current_ax
        nx.draw_networkx_labels(**params)
    
    # Set title
    current_ax.set_title(title)
    
    # Only show if we created the figure ourselves
    if ax is None:
        plt.show()


def plot_multiple_quantile_distributions(distances_list: List[List[int]], labels: List[str], save_path: Optional[str] = None) -> None:
    """
    Plot quantile distributions for multiple distance datasets.
    
    Args:
        distances_list: List of distance lists to plot
        labels: Labels for each distance dataset
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 5))
    
    # Plot quantile distribution for each dataset
    for distances, label in zip(distances_list, labels):
        quantiles = np.percentile(distances, np.linspace(0, 100, 101))
        plt.plot(np.linspace(0, 100, 101), quantiles, label=label)
    
    plt.xlabel('Percentile')
    plt.ylabel('GUF Distance')
    plt.title('Quantile Distribution of Distances')
    plt.legend(title='Distance Types')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()