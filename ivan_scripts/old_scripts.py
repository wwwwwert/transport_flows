"""
Utility functions for P-median distance analysis.

This module contains refactored functions extracted from the original Jupyter notebook
for analyzing P-median problems with different distance metrics on graphs.
"""

import pyomo.environ as pyo
import numpy as np
import networkx as nx
import pandas as pd
import igraph as ig
from igraph import Graph
from pyomo.solvers.plugins.solvers.GLPK import GLPKSHELL
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from scipy import stats
from copy import deepcopy
import random
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union


# Graph Operations
def load_graph_from_csv(csv_path: str, weight_value: float = 1.0) -> pd.DataFrame:
    """
    Load graph edges from CSV file and preprocess the data.
    
    Args:
        csv_path: Path to the CSV file containing edge data
        weight_value: Default weight value for edges
        
    Returns:
        DataFrame with preprocessed edge data including weights
    """
    # Load data and add weights
    edges_df = pd.read_csv(csv_path)
    edges_df['weight'] = weight_value
    
    # Create pairs and remove duplicates
    df = edges_df.copy()
    df['pair'] = df.apply(lambda row: tuple(sorted([row['source'], row['target']])), axis=1)
    df = df.drop_duplicates(subset='pair')
    
    # Clean up and return
    unique_pairs_df = df.drop(columns=['pair']).reset_index(drop=True)
    return unique_pairs_df


def create_graphs(edges_df: pd.DataFrame) -> Tuple[ig.Graph, nx.Graph]:
    """
    Create both igraph and networkx graph representations from edge data.
    
    Args:
        edges_df: DataFrame containing edge data with 'source', 'target', and 'weight' columns
        
    Returns:
        Tuple of (igraph.Graph, networkx.Graph) objects
    """
    # Create igraph
    ig_graph = ig.Graph.TupleList(
        edges_df.itertuples(index=False), 
        directed=False, 
        edge_attrs=list(edges_df.columns[2:])
    )
    
    # Ensure all edges have proper weights in igraph
    for edge in ig_graph.es:
        if not isinstance(edge['weight'], (int, float)) or pd.isnull(edge['weight']):
            edge['weight'] = 1
    
    # Create networkx graph
    nx_graph = nx.from_pandas_edgelist(edges_df)
    for (a, b) in nx_graph.edges():
        nx_graph[a][b]['weight'] = 1
        
    return ig_graph, nx_graph


def graph_distances_dict(graph: ig.Graph, weight: str = 'weight') -> Dict[str, Dict[str, float]]:
    """
    Calculate shortest path matrix and return as dictionary with vertex names.
    
    Args:
        graph: igraph Graph object
        weight: Name of the edge attribute to use as weight
        
    Returns:
        Dictionary where keys are vertex names, values are dictionaries of distances to other vertices
    """
    # Get vertex names
    vertex_names = graph.vs["name"] if "name" in graph.vs.attributes() else list(range(graph.vcount()))
    
    # Calculate distance matrix
    distances = graph.distances(weights=weight)
    
    # Convert to dictionary
    distances_dict = {
        vertex_names[i]: {
            vertex_names[j]: distances[i][j]
            for j in range(len(vertex_names))
        }
        for i in range(len(vertex_names))
    }
    
    return distances_dict


def protected_distance(graph: ig.Graph, weight: str = 'weight', inf_replacement: float = 10**6) -> Tuple[pd.DataFrame, bool]:
    """
    Calculate protected distances (maximum shortest path after removing any single edge).
    
    Args:
        graph: igraph Graph object
        weight: Name of the edge attribute to use as weight
        inf_replacement: Value to use for infinite distances
        
    Returns:
        Tuple of (DataFrame with protected distances, boolean indicating if graph is biconnected)
    """
    # Check if graph is biconnected
    biconnected = graph.is_biconnected()

    # Pre-calculate all shortest paths for original graph
    all_shortest_paths = graph_distances_dict(graph, weight=weight)
    
    # Create copy for storing results
    res_dict = deepcopy(all_shortest_paths)

    # Iterate through all edges
    for edge_id in tqdm(range(graph.ecount())):
        # Create temporary graph copy
        temp_graph = deepcopy(graph)
        temp_graph.delete_edges(edge_id)

        # Ensure all edges have numeric weights
        for edge in temp_graph.es:
            if not isinstance(edge['weight'], (int, float)) or pd.isnull(edge['weight']):
                edge['weight'] = 1

        # Recalculate shortest paths after edge removal
        cur_shortest_paths = graph_distances_dict(temp_graph, weight=weight)

        # Compare and update results
        for node_1 in graph.vs:
            u = node_1['name']
            for node_2 in graph.vs:
                v = node_2['name']
                if cur_shortest_paths[u][v] > res_dict[u][v]:
                    if cur_shortest_paths[u][v] == float('inf'):
                        res_dict[u][v] = inf_replacement
                    else:
                        res_dict[u][v] = cur_shortest_paths[u][v]

    return pd.DataFrame(res_dict), biconnected


def prune_leaf_nodes(graph: nx.Graph) -> nx.Graph:
    """
    Remove leaf nodes (degree 1) iteratively until no more leaf nodes exist.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        Graph with leaf nodes removed
    """
    # Create copy to avoid modifying original graph
    G = graph.copy()

    # Continue until no leaf nodes remain
    while True:
        # Find all nodes with degree 1
        leaf_nodes = [node for node in G.nodes() if G.degree(node) == 1]

        # If no leaf nodes, break
        if len(leaf_nodes) == 0:
            break

        # Remove found leaf nodes
        G.remove_nodes_from(leaf_nodes)

    return G


def calculate_distance_matrices(ig_graph: ig.Graph, nx_graph: nx.Graph) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate all three types of distance matrices: protected, resistance, and geodesic.
    
    Args:
        ig_graph: igraph Graph object
        nx_graph: NetworkX Graph object
        
    Returns:
        Tuple of (protected_data, resistance_data, distance_data) DataFrames
    """
    # Calculate protected distances
    protected_data = pd.DataFrame(protected_distance(ig_graph, weight='weight')[0])
    protected_data.replace(np.inf, 10**9, inplace=True)
    
    # Calculate resistance distances
    resistance_data = pd.DataFrame(nx.resistance_distance(nx_graph, weight='weight'))
    
    # Calculate geodesic distances
    distance_data = pd.DataFrame(dict(nx.all_pairs_shortest_path_length(nx_graph)))
    distance_data = distance_data.sort_index(axis=0).sort_index(axis=1)
    
    return protected_data, resistance_data, distance_data


# Optimization Functions
def create_pmedian_model(distance_matrix: pd.DataFrame, p: int) -> pyo.ConcreteModel:
    """
    Create a P-median optimization model using Pyomo.
    
    Args:
        distance_matrix: DataFrame containing distance matrix
        p: Number of facilities to locate
        
    Returns:
        Pyomo ConcreteModel ready for solving
    """
    # Convert to numpy array
    c = distance_matrix.to_numpy()
    
    # Create model
    model = pyo.ConcreteModel()
    
    # Create sets
    model.M = pyo.Set(initialize=list(distance_matrix.index))  # Clients
    model.N = pyo.Set(initialize=list(distance_matrix.columns))  # Potential facility locations
    
    # Parameters (cost matrix)
    model.c = pyo.Param(
        model.M, model.N, 
        initialize=lambda model, i, j: c[list(distance_matrix.index).index(i)][list(distance_matrix.columns).index(j)], 
        within=pyo.NonNegativeReals
    )
    
    # Variables
    model.x = pyo.Var(model.M, model.N, within=pyo.Binary)  # Assignment variables
    model.y = pyo.Var(model.N, within=pyo.Binary)  # Facility location variables
    
    # Objective function
    def obj_rule(model):
        return sum(model.c[i, j] * model.x[i, j] for i in model.M for j in model.N)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Constraints
    def single_assignment_rule(model, i):
        return sum(model.x[i, j] for j in model.N) == 1
    model.single_assignment = pyo.Constraint(model.M, rule=single_assignment_rule)
    
    def facility_open_rule(model, i, j):
        return model.x[i, j] <= model.y[j]
    model.facility_open = pyo.Constraint(model.M, model.N, rule=facility_open_rule)
    
    model.facility_count = pyo.Constraint(expr=sum(model.y[j] for j in model.N) == p)
    
    return model


def solve_pmedian_problem(
        distance_matrix: pd.DataFrame, 
        p: int, 
        solver_path: str = '/usr/local/bin/glpsol',
        verbose: bool = False
    ) -> Dict[str, List[str]]:
    """
    Solve P-median problem for given distance matrix.
    
    Args:
        distance_matrix: DataFrame containing distance matrix
        p: Number of facilities to locate
        solver_path: Path to GLPK solver executable
        
    Returns:
        Dictionary with facility locations as keys and assigned clients as values
    """
    # Create model
    model = create_pmedian_model(distance_matrix, p)
    
    # Solve
    solver: GLPKSHELL = pyo.SolverFactory('glpk', executable=solver_path)
    solver.solve(model, tee=verbose)
    
    # Extract results
    return extract_solution_results(model)


def extract_solution_results(model: pyo.ConcreteModel) -> Dict[str, List[str]]:
    """
    Extract solution results from solved P-median model.
    
    Args:
        model: Solved Pyomo ConcreteModel
        
    Returns:
        Dictionary with facility locations as keys and assigned clients as values
    """
    result_dict = {}
    for j in model.N:
        if pyo.value(model.y[j]) > 0.5:
            result_dict[j] = []
            for i in model.M:
                if pyo.value(model.x[i, j]) > 0.5:
                    result_dict[j].append(i)
    return result_dict


# Analysis Functions
def extract_distances_from_solution(result_dict: Dict[str, List[str]], 
                                   prot_dict: Dict[Tuple[str, str], float]) -> List[int]:
    """
    Extract protected distances for solution analysis.
    
    Args:
        result_dict: Dictionary with facility locations and assigned clients
        prot_dict: Dictionary with protected distances between vertex pairs
        
    Returns:
        List of distances for analysis
    """
    result_list = []
    for facility in result_dict.keys():
        for client in result_dict[facility]:
            try:
                # Try to get value from prot_dict[(facility, client)]
                value = prot_dict[(facility, client)]
            except KeyError:
                try:
                    # If not found, try prot_dict[(client, facility)]
                    value = prot_dict[(client, facility)]
                except KeyError:
                    # If neither key found, set default value
                    value = None
            if value is not None:
                result_list.append(int(value))
    return result_list


# Visualization Functions
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


def plot_multiple_quantile_distributions(distances_list: List[List[int]], labels: List[str]) -> None:
    """
    Plot quantile distributions for multiple distance datasets.
    
    Args:
        distances_list: List of distance lists to plot
        labels: Labels for each distance dataset
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
    plt.show()