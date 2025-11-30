"""
Graph operations module for P-median distance analysis.

This module contains functions for loading, creating, and analyzing graphs.
"""

import numpy as np
import networkx as nx
import pandas as pd
import igraph as ig
from tqdm import tqdm
from copy import deepcopy
import re
import ast
from typing import Dict, List, Tuple, Optional, Union


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