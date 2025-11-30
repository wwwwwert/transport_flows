# %%
%load_ext autoreload
%autoreload 2

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

%config InlineBackend.figure_format = 'retina'
plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.format'] = 'pdf'
sns.set_style('darkgrid')

# %%
import numpy as np
import networkx as nx
import pandas as pd

# Import our refactored functions
from scripts import (
    load_graph_from_csv,
    create_graphs,
    calculate_distance_matrices,
    solve_pmedian_problem,
    extract_distances_from_solution,
    create_results_dataframe,
    draw_graph_with_centers,
    plot_multiple_quantile_distributions,
    prune_leaf_nodes
)

# %%
CSV_FILE = 'Tata_upd.csv'
NUM_FACILITIES = 6  # Number of centers to place (p parameter)
SOLVER_PATH = '/usr/local/bin/glpsol'  # Configurable solver path

# %%
print("Loading and preprocessing graph data...")
edges_data = load_graph_from_csv(CSV_FILE, weight_value=1.0)
print(f"Loaded {len(edges_data)} unique edges")
edges_data.head()

# %%
print("Creating graph representations...")
igraph_graph, networkx_graph = create_graphs(edges_data)
print(f"Created graphs with {igraph_graph.vcount()} vertices and {igraph_graph.ecount()} edges")

# %%
print("Calculating distance matrices...")
protected_distances, resistance_distances, geodesic_distances = calculate_distance_matrices(
    igraph_graph, networkx_graph
)
print(f"- Protected distances: {protected_distances.shape}")
print(f"- Resistance distances: {resistance_distances.shape}")
print(f"- Geodesic distances: {geodesic_distances.shape}")

# %%
num_clients = len(networkx_graph.nodes)
num_locations = len(networkx_graph.nodes)
num_facilities = NUM_FACILITIES

print(f"Problem parameters:")
print(f"- Number of clients: {num_clients}")
print(f"- Number of potential locations: {num_locations}")
print(f"- Number of facilities to place: {num_facilities}")

# %%
print("\nSolving P-median problem with geodesic distances...")
geodesic_solution = solve_pmedian_problem(
    geodesic_distances, 
    num_facilities, 
    solver_path=SOLVER_PATH
)
print("Geodesic solution:")
print(geodesic_solution)
print(f"Selected vertices: {list(geodesic_solution.keys())}")

# %%
print("\nSolving P-median problem with resistance distances...")
resistance_solution = solve_pmedian_problem(
    resistance_distances, 
    num_facilities, 
    solver_path=SOLVER_PATH
)
print("Resistance solution:")
print(resistance_solution)
print(f"Selected vertices: {list(resistance_solution.keys())}")

# %%
print("\nSolving P-median problem with protected distances...")
protected_solution = solve_pmedian_problem(
    protected_distances, 
    num_facilities, 
    solver_path=SOLVER_PATH
)
print("Protected solution:")
print(protected_solution)
print(f"Selected vertices: {list(protected_solution.keys())}")

# %%
print("\nVisualizing original graph...")
nx.draw_networkx(networkx_graph)
plt.title("Original Graph")
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

draw_graph_with_centers(
    networkx_graph, 
    geodesic_solution, 
    title='Geodesic P-median Solution',
    ax = axes[0]
)

draw_graph_with_centers(
    networkx_graph, 
    resistance_solution, 
    title='Resistance P-median Solution',
    ax = axes[1]
)

draw_graph_with_centers(
    networkx_graph, 
    protected_solution, 
    title='Protected P-median Solution',
    ax = axes[2]
)

# %%
print("\nCalculating protected distances dictionary for detailed analysis...")
protected_dict = {
    (row_name, col_name): protected_distances.loc[row_name, col_name]
    for row_name in protected_distances.index
    for col_name in protected_distances.columns
}
print(f"Protected distances dictionary contains {len(protected_dict)} entries")

# %%
print("Extracting distances for geodesic solution analysis...")
geodesic_analysis_distances = extract_distances_from_solution(
    geodesic_solution, 
    protected_dict
)
print(f"Extracted {len(geodesic_analysis_distances)} distance values")

# %%
print("Extracting distances for resistance solution analysis...")
resistance_analysis_distances = extract_distances_from_solution(
    resistance_solution, 
    protected_dict
)
print(f"Extracted {len(resistance_analysis_distances)} distance values")

# %%
print("Extracting distances for protected solution analysis...")
protected_analysis_distances = extract_distances_from_solution(
    protected_solution, 
    protected_dict
)
print(f"Extracted {len(protected_analysis_distances)} distance values")

# %%
print("Plotting quantile distributions comparison...")
plot_multiple_quantile_distributions(
    [geodesic_analysis_distances, resistance_analysis_distances, protected_analysis_distances],
    ['Geodesic centers', 'Resistance centers', 'GUF centers']
)


# %%
print("Creating results DataFrame and performing statistical analysis...")
results_df = create_results_dataframe(
    geodesic_analysis_distances,
    resistance_analysis_distances, 
    protected_analysis_distances
)

# Calculate quantiles for analysis
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
statistical_summary = results_df.describe(percentiles=quantiles)

print("Statistical Summary:")
print(statistical_summary)

# %%
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print(f"\nProblem Configuration:")
print(f"- Dataset: {CSV_FILE}")
print(f"- Number of facilities: {num_facilities}")
print(f"- Graph size: {num_clients} nodes, {igraph_graph.ecount()} edges")

print(f"\nSolution Comparison:")
print(f"- Geodesic centers: {list(geodesic_solution.keys())}")
print(f"- Resistance centers: {list(resistance_solution.keys())}")
print(f"- Protected centers: {list(protected_solution.keys())}")

print(f"\nDistance Analysis (median values):")
print(f"- Geodesic solution median distance: {np.median(geodesic_analysis_distances):.2f}")
print(f"- Resistance solution median distance: {np.median(resistance_analysis_distances):.2f}")
print(f"- Protected solution median distance: {np.median(protected_analysis_distances):.2f}")

print("\nAnalysis complete!")


