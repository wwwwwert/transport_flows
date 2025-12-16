# %%
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style('darkgrid')

# %%
import numpy as np
import networkx as nx
import pandas as pd

# Import our refactored functions
from scripts import (
    load_graph_from_csv,
    create_graphs_from_edgelist,
    calculate_distance_matrices,
    solve_pmedian_problem,
    extract_distances_from_solution,
    create_results_dataframe,
    draw_graph_with_centers,
    plot_multiple_quantile_distributions,
    draw_networkx_auto,
    prune_leaf_nodes
)


for WEIGHT in ['binary', 'length', 'time']:
    # %%
    EDGELIST_FILE = 'data/graphs/yekaterinburg_small_speed_history_cleared.edgelist'
    NUM_FACILITIES = 6
    SOLVER_PATH = '/usr/local/bin/glpsol'

    # Extract graph name and create output directory structure
    GRAPH_NAME = Path(EDGELIST_FILE).stem
    OUTPUT_DIR = Path('data') / 'results' / 'solve_p_median' / GRAPH_NAME / WEIGHT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Graph: {GRAPH_NAME}")
    print(f"Weight type: {WEIGHT}")
    print(f"Output directory: {OUTPUT_DIR}")

    # %%
    print("Creating graph representations...")
    igraph_graph, networkx_graph = create_graphs_from_edgelist(EDGELIST_FILE, weight=WEIGHT)
    print(f"Created graphs with {igraph_graph.vcount()} vertices and {igraph_graph.ecount()} edges")

    # %%
    # This functionality is available but not used in the main analysis
    # pruned_graph = prune_leaf_nodes(networkx_graph)
    # print(f"Pruned graph would have {len(pruned_graph.nodes())} nodes")

    # %%
    print("Calculating distance matrices...")
    protected_distances, resistance_distances, geodesic_distances = calculate_distance_matrices(
        igraph_graph, networkx_graph
    )
    print(f"- Protected distances: {protected_distances.shape}")
    print(f"- Resistance distances: {resistance_distances.shape}")
    print(f"- Geodesic distances: {geodesic_distances.shape}")

    # %%
    # protected_distances = protected_distances.iloc[:200, :200]
    # resistance_distances = resistance_distances.iloc[:200, :200]
    # geodesic_distances = geodesic_distances.iloc[:200, :200]

    # %%
    num_clients = len(networkx_graph.nodes)
    num_locations = len(networkx_graph.nodes)
    num_facilities = NUM_FACILITIES

    print(f"Problem parameters:")
    print(f"- Number of clients: {num_clients}")
    print(f"- Number of potential locations: {num_locations}")
    print(f"- Number of facilities to place: {num_facilities}")

    # %% [markdown]
    # 300x300 работает нормально

    # %%
    print("\nSolving P-median problem with geodesic distances...")
    geodesic_solution = solve_pmedian_problem(
        geodesic_distances, 
        num_facilities, 
        solver_path=SOLVER_PATH,
        use_heuristic_solution=False,
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
    draw_networkx_auto(
        networkx_graph,
        title="Original Graph",
        save_path=OUTPUT_DIR / "original_graph.png"
    )

    # %%
    print("Creating P-median solutions comparison...")
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

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pmedian_solutions_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

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
        ['Geodesic centers', 'Resistance centers', 'GUF centers'],
        save_path=OUTPUT_DIR / "quantile_distributions.png"
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
    print(f"- Dataset: {EDGELIST_FILE}")
    print(f"- Weight type: {WEIGHT}")
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

    # Save results summary to file
    summary_text = f"""P-MEDIAN ANALYSIS RESULTS
    {'='*50}

    Problem Configuration:
    - Dataset: {EDGELIST_FILE}
    - Weight type: {WEIGHT}
    - Number of facilities: {num_facilities}
    - Graph size: {num_clients} nodes, {igraph_graph.ecount()} edges

    Solution Comparison:
    - Geodesic centers: {list(geodesic_solution.keys())}
    - Resistance centers: {list(resistance_solution.keys())}
    - Protected centers: {list(protected_solution.keys())}

    Distance Analysis (median values):
    - Geodesic solution median distance: {np.median(geodesic_analysis_distances):.2f}
    - Resistance solution median distance: {np.median(resistance_analysis_distances):.2f}
    - Protected solution median distance: {np.median(protected_analysis_distances):.2f}

    Statistical Summary:
    {statistical_summary.to_string()}
    """

    with open(OUTPUT_DIR / "analysis_summary.txt", 'w') as f:
        f.write(summary_text)

    # Save statistical summary as CSV
    statistical_summary.to_csv(OUTPUT_DIR / "statistical_summary.csv")

    # Save results dataframe
    results_df.to_csv(OUTPUT_DIR / "results_data.csv", index=False)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Files created:")
    print("- original_graph.png")
    print("- pmedian_solutions_comparison.png")
    print("- quantile_distributions.png")
    print("- analysis_summary.txt")
    print("- statistical_summary.csv")
    print("- results_data.csv")
    print("\nAnalysis complete!")


