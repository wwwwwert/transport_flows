"""
Scripts package for P-median distance analysis.

This package contains modular functions for analyzing P-median problems
with different distance metrics on graphs.
"""

# Import all main functions for easy access
from .graph_operations import (
    load_graph_from_csv,
    create_graphs,
    create_graphs_from_edgelist,
    graph_distances_dict,
    protected_distance,
    protected_distance_speed_graph,
    extract_speed_attributes,
    has_speed_attributes,
    prune_leaf_nodes,
    calculate_distance_matrices
)

from .optimization import (
    create_pmedian_model,
    solve_pmedian_problem,
    solve_pmedian_teitz_bart,
    extract_solution_results
)

from .analysis import (
    extract_distances_from_solution,
    create_results_dataframe
)

from .visualization import (
    create_node_labels,
    get_visualization_params,
    draw_graph_with_centers,
    draw_networkx_auto,
    plot_multiple_quantile_distributions
)

__all__ = [
    # Graph operations
    'load_graph_from_csv',
    'create_graphs',
    'create_graphs_from_edgelist',
    'graph_distances_dict',
    'protected_distance',
    'protected_distance_speed_graph',
    'extract_speed_attributes',
    'has_speed_attributes',
    'prune_leaf_nodes',
    'calculate_distance_matrices',
    
    # Optimization
    'create_pmedian_model',
    'solve_pmedian_problem',
    'solve_pmedian_teitz_bart',
    'extract_solution_results',
    
    # Analysis
    'extract_distances_from_solution',
    'create_results_dataframe',
    
    # Visualization
    'create_node_labels',
    'get_visualization_params',
    'draw_graph_with_centers',
    'draw_networkx_auto',
    'plot_multiple_quantile_distributions'
]