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
    Automatically detects speed attributes and uses speed-based protected distances if available.
    
    Args:
        ig_graph: igraph Graph object
        nx_graph: NetworkX Graph object
        
    Returns:
        Tuple of (protected_data, resistance_data, distance_data) DataFrames
    """
    # Check if speed attributes are available
    if has_speed_attributes(ig_graph):
        print("Speed attributes detected. Using speed-based protected distances...")
        # Calculate speed-based protected distances
        protected_data = pd.DataFrame(protected_distance_speed_graph(ig_graph)[0])
        protected_data.replace(np.inf, 10**9, inplace=True)
    else:
        print("No speed attributes found. Using standard protected distances...")
        # Calculate standard protected distances
        protected_data = pd.DataFrame(protected_distance(ig_graph, weight='weight')[0])
        protected_data.replace(np.inf, 10**9, inplace=True)
    
    # Calculate resistance distances (unchanged)
    resistance_data = pd.DataFrame(nx.resistance_distance(nx_graph, weight='weight'))
    
    # Calculate geodesic distances (unchanged)
    distance_data = pd.DataFrame(dict(nx.all_pairs_shortest_path_length(nx_graph)))
    distance_data = distance_data.sort_index(axis=0).sort_index(axis=1)
    
    return protected_data, resistance_data, distance_data


def extract_speed_attributes(edge_attrs: Dict) -> List[str]:
    """
    Extract speed attribute names from edge attributes dictionary.
    
    Args:
        edge_attrs: Dictionary containing edge attributes
        
    Returns:
        List of speed attribute names (e.g., ['speed_23:25:00', 'speed_05:00:00', ...])
    """
    speed_pattern = re.compile(r'^speed_\d{2}:\d{2}:\d{2}$')
    return [attr for attr in edge_attrs.keys() if speed_pattern.match(attr)]


def create_graphs_from_edgelist(edgelist_path: str) -> Tuple[ig.Graph, nx.Graph]:
    """
    Create both igraph and networkx graph representations from edgelist file.
    
    Args:
        edgelist_path: Path to the edgelist file containing edge data with speed attributes
        
    Returns:
        Tuple of (igraph.Graph, networkx.Graph) objects with speed attributes preserved
    """
    print(f"Loading graph from edgelist: {edgelist_path}")
    
    # Load NetworkX graph directly from edgelist
    nx_graph = nx.read_edgelist(edgelist_path, create_using=nx.Graph())
    
    print(f"Loaded NetworkX graph with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges")
    
    # Convert to igraph
    # First, create edge list with attributes for igraph
    edges_data = []
    for u, v, attrs in nx_graph.edges(data=True):
        edge_record = {'source': u, 'target': v}
        edge_record.update(attrs)
        edges_data.append(edge_record)
    
    if not edges_data:
        raise ValueError("No edges found in the graph")
    
    # Convert to DataFrame for processing
    edges_df = pd.DataFrame(edges_data)
    
    # Interpolate missing speed values
    speed_cols = [col for col in edges_df.columns if col.startswith('speed_')]
    print(f"Interpolating {len(speed_cols)} speed attributes...")
    # Fill missing values: first backward fill, then forward fill
    edges_df[speed_cols] = edges_df[speed_cols].replace(0, np.nan)
    edges_df[speed_cols] = edges_df[speed_cols].bfill(axis=1).ffill(axis=1).fillna(20.0)
    edges_df['weight'] = 1.0


    zero_count = (edges_df[speed_cols] == 0).sum().sum()
    print(f"Total zeros in speed columns: {zero_count}")
    nan_count = (edges_df[speed_cols].isna()).sum().sum()
    print(f"Total nan in speed columns: {nan_count}")

    print("Creating igraph")

    # Create igraph
    edge_attrs = list(edges_df.columns[2:])  # All columns except source and target
    ig_graph = ig.Graph.TupleList(
        edges_df.itertuples(index=False),
        directed=False,
        edge_attrs=edge_attrs
    )
    
    # Update NetworkX graph with interpolated values
    nx_graph_updated = nx.Graph()
    for _, row in tqdm(edges_df.iterrows(), desc='Creating nx_graph', total=edges_df.shape[0]):
        source = row['source']
        target = row['target']
        attrs = row.drop(['source', 'target']).to_dict()
        nx_graph_updated.add_edge(source, target, **attrs)
        
    return ig_graph, nx_graph_updated


def create_time_weighted_graph(graph: ig.Graph, time_attr: str) -> ig.Graph:
    """
    Create a graph with edge weights based on travel time for a specific time period.
    
    Args:
        graph: igraph Graph object with speed attributes
        time_attr: Speed attribute name (e.g., 'speed_23:25:00')
        
    Returns:
        Graph with edge weights set to travel_time = length / speed (in hours)
    """
    temp_graph = deepcopy(graph)
    
    for edge in temp_graph.es:
        length = edge['length']
        speed = edge[time_attr]
        if speed == 0:
            raise RuntimeError(f'speed id {speed} ; time_attr is {time_attr}')
        travel_time = (length / 1000.0) / speed
        if travel_time < 10**-5 or np.isnan(travel_time):
            raise RuntimeError(f'travel_time is {travel_time} ; time_attr is {time_attr}')
        edge['weight'] = travel_time

    return temp_graph


def protected_distance_speed_graph(graph: ig.Graph, inf_replacement: float = 10**6) -> Tuple[pd.DataFrame, bool]:
    """
    Calculate protected distances using time-varying speeds (maximum shortest path across all time periods).
    
    Args:
        graph: igraph Graph object with speed attributes
        inf_replacement: Value to use for infinite distances
        
    Returns:
        Tuple of (DataFrame with protected distances, boolean indicating if graph is biconnected)
    """
    # Check if graph is biconnected
    biconnected = graph.is_biconnected()
    
    # Extract all speed attributes
    if graph.ecount() == 0:
        raise ValueError("Graph has no edges")
    
    # Get speed attributes from the first edge
    first_edge_attrs = dict(graph.es[0].attributes())
    speed_attrs = extract_speed_attributes(first_edge_attrs)
    
    if not speed_attrs:
        raise ValueError("No speed attributes found in graph edges")
    
    print(f"Found {len(speed_attrs)} time intervals for speed analysis")
    
    # Get vertex names
    vertex_names = graph.vs["name"] if "name" in graph.vs.attributes() else list(range(graph.vcount()))
    
    # Initialize result matrix with zeros
    num_vertices = len(vertex_names)
    max_distances = np.zeros((num_vertices, num_vertices))

    i = 0
    
    # Iterate through all time periods
    for time_attr in tqdm(speed_attrs, desc="Processing time intervals"):
        i += 1
        if i > 5:
            break
        # Create graph with weights based on current time period
        time_graph = create_time_weighted_graph(graph, time_attr)
        
        # Calculate shortest paths for this time period
        try:
            distances = time_graph.distances(weights='weight')
            
            # Update maximum distances
            for i in range(num_vertices):
                for j in range(num_vertices):
                    current_dist = distances[i][j]
                    if current_dist == float('inf'):
                        current_dist = inf_replacement
                    
                    # Take maximum distance across all time periods
                    max_distances[i][j] = max(max_distances[i][j], current_dist)
                    
        except Exception as e:
            print(f"Warning: Error processing time {time_attr}: {e}")
            continue
    
    # Convert to DataFrame with proper index and column names
    result_df = pd.DataFrame(
        max_distances,
        index=vertex_names,
        columns=vertex_names
    )
    
    return result_df, biconnected


def has_speed_attributes(graph: ig.Graph) -> bool:
    """
    Check if the graph has speed attributes on its edges.
    
    Args:
        graph: igraph Graph object
        
    Returns:
        True if speed attributes are found, False otherwise
    """
    if graph.ecount() == 0:
        return False
    
    # Check first edge for speed attributes
    first_edge_attrs = dict(graph.es[0].attributes())
    speed_attrs = extract_speed_attributes(first_edge_attrs)
    
    return len(speed_attrs) > 0