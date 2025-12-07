"""
Visualization module for P-median analysis.

This module contains functions for visualizing graphs and analysis results.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd


def extract_node_coordinates_from_edges(graph: nx.Graph) -> Optional[Dict]:
    """
    Extract node coordinates from edge attributes.
    
    Args:
        graph: NetworkX graph object with edge coordinate attributes
        
    Returns:
        Dictionary mapping node IDs to (longitude, latitude) tuples, or None if no coordinates found
    """
    node_coords = {}
    
    # Check if any edge has coordinate attributes
    sample_edge = next(iter(graph.edges(data=True)), None)
    if sample_edge is None:
        return None
    
    edge_data = sample_edge[2]
    coord_attrs = ['x_coordinate_start', 'y_coordinate_start', 'x_coordinate_end', 'y_coordinate_end']
    
    if not all(attr in edge_data for attr in coord_attrs):
        return None
    
    # Extract coordinates from all edges
    for u, v, data in graph.edges(data=True):
        # Start node coordinates
        if u not in node_coords:
            node_coords[u] = (data['x_coordinate_start'], data['y_coordinate_start'])
        
        # End node coordinates
        if v not in node_coords:
            node_coords[v] = (data['x_coordinate_end'], data['y_coordinate_end'])
    
    return node_coords


def has_speed_attributes(graph: nx.Graph) -> bool:
    """
    Check if graph has speed attributes in edges.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        True if speed attributes are found, False otherwise
    """
    sample_edge = next(iter(graph.edges(data=True)), None)
    if sample_edge is None:
        return False
    
    edge_data = sample_edge[2]
    # Look for any speed attribute (they follow pattern speed_HH:MM:SS)
    return any(key.startswith('speed_') for key in edge_data.keys())


def create_geographic_layout(graph: nx.Graph) -> Optional[Dict]:
    """
    Create layout using geographic coordinates from edge attributes.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        Layout dictionary mapping nodes to (x, y) positions, or None if no coordinates
    """
    node_coords = extract_node_coordinates_from_edges(graph)
    if node_coords is None:
        return None
    
    # Convert to layout format (node_id -> (x, y))
    layout = {}
    for node_id, (lon, lat) in node_coords.items():
        layout[node_id] = (lon, lat)
    
    return layout


def draw_graph_on_map(graph: nx.Graph, centers_dict: Optional[Dict[str, List[str]]] = None,
                     node_size_multiplier: int = 50, title: str = '',
                     ax: Optional[plt.Axes] = None, save_path: Optional[str] = None) -> None:
    """
    Draw graph on geographic map with OpenStreetMap background.
    
    Args:
        graph: NetworkX graph object with coordinate attributes
        centers_dict: Optional dictionary with center nodes and their assigned clients
        node_size_multiplier: Multiplier for node sizes
        title: Title for the plot
        ax: Optional matplotlib Axes object. If None, creates new figure
        save_path: Optional path to save the plot
    """
    # Extract coordinates
    node_coords = extract_node_coordinates_from_edges(graph)
    if node_coords is None:
        raise ValueError("Graph does not contain coordinate attributes in edges")
    
    # Create GeoDataFrame for nodes
    nodes_data = []
    for node_id, (lon, lat) in node_coords.items():
        nodes_data.append({
            'node_id': node_id,
            'geometry': Point(lon, lat),
            'is_center': centers_dict is not None and node_id in centers_dict
        })
    
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
    
    # Create GeoDataFrame for edges
    edges_data = []
    for u, v, data in graph.edges(data=True):
        start_coords = node_coords[u]
        end_coords = node_coords[v]
        line = LineString([start_coords, end_coords])
        edges_data.append({
            'from_node': u,
            'to_node': v,
            'geometry': line
        })
    
    edges_gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
    
    # Convert to Web Mercator for contextily
    nodes_gdf = nodes_gdf.to_crs('EPSG:3857')
    edges_gdf = edges_gdf.to_crs('EPSG:3857')
    
    # Create figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot edges first (so they appear behind nodes)
    edges_gdf.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.7)
    
    if centers_dict is not None:
        # Plot non-center nodes
        non_centers = nodes_gdf[~nodes_gdf['is_center']]
        if not non_centers.empty:
            # Color nodes by their assigned center
            colors = []
            rainbow = plt.cm.rainbow
            num_centers = len(centers_dict)
            center_color_map = {}
            
            for idx, center in enumerate(centers_dict):
                center_color_map[center] = rainbow(idx / num_centers)
            
            for _, node in non_centers.iterrows():
                node_id = node['node_id']
                # Find which center this node belongs to
                node_color = 'gray'  # default
                for center, assigned_nodes in centers_dict.items():
                    if node_id in assigned_nodes:
                        node_color = center_color_map[center]
                        break
                colors.append(node_color)
            
            non_centers.plot(ax=ax, color=colors, markersize=node_size_multiplier,
                           alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Plot center nodes with different style
        centers = nodes_gdf[nodes_gdf['is_center']]
        if not centers.empty:
            center_colors = []
            for _, node in centers.iterrows():
                center_id = node['node_id']
                center_colors.append(center_color_map.get(center_id, 'red'))
            
            centers.plot(ax=ax, color=center_colors, markersize=node_size_multiplier * 2,
                        marker='s', alpha=0.9, edgecolors='black', linewidth=1)
    else:
        # Plot all nodes with same style
        nodes_gdf.plot(ax=ax, color='blue', markersize=node_size_multiplier,
                      alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add basemap
    ctx.add_basemap(ax, crs=nodes_gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
    
    # Set title and remove axes
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_axis_off()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Only show if we created the figure ourselves
    if ax is None:
        plt.show()


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
                          ax: Optional[plt.Axes] = None, save_path: Optional[str] = None) -> None:
    """
    Draw graph with highlighted center nodes and color-coded assignments.
    Automatically detects if geographic coordinates are available and uses map visualization.
    
    Args:
        graph: NetworkX graph object
        centers_dict: Dictionary with center nodes and their assigned clients
        node_size_multiplier: Multiplier for node sizes
        title: Title for the plot
        ax: Optional matplotlib Axes object. If None, creates new figure
        save_path: Optional path to save the plot
    """
    # Check if graph has geographic coordinates
    if has_speed_attributes(graph) and extract_node_coordinates_from_edges(graph) is not None:
        # Use geographic visualization with map background
        draw_graph_on_map(graph, centers_dict, node_size_multiplier, title, ax, save_path)
        return
    
    # Fall back to standard layout-based visualization
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
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Only show if we created the figure ourselves
    if ax is None:
        plt.show()


def draw_networkx_auto(graph: nx.Graph, title: str = "Graph",
                      ax: Optional[plt.Axes] = None, save_path: Optional[str] = None) -> None:
    """
    Automatically draw NetworkX graph with best available method.
    Uses geographic visualization if coordinates are available, otherwise uses standard layout.
    
    Args:
        graph: NetworkX graph object
        title: Title for the plot
        ax: Optional matplotlib Axes object. If None, creates new figure
        save_path: Optional path to save the plot
    """
    # Check if graph has geographic coordinates
    if has_speed_attributes(graph) and extract_node_coordinates_from_edges(graph) is not None:
        # Use geographic visualization with map background
        draw_graph_on_map(graph, centers_dict=None, title=title, ax=ax, save_path=save_path)
    else:
        # Use standard NetworkX drawing
        if ax is None:
            plt.figure(figsize=(8, 5))
            current_ax = plt.gca()
        else:
            current_ax = ax
        
        nx.draw_networkx(graph, ax=current_ax)
        current_ax.set_title(title)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
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