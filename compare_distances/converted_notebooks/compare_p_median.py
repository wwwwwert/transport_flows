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
    create_graphs,
    calculate_distance_matrices,
    solve_pmedian_problem,
    extract_distances_from_solution,
    create_results_dataframe,
    draw_graph_with_centers,
    plot_multiple_quantile_distributions,
    prune_leaf_nodes,
    create_graphs_from_edgelist
)

for WEIGHT in ['binary', 'length', 'time']:

    # %%
    EDGELIST_FILE = 'data/graphs/yekaterinburg_small_speed_history_cleared.edgelist'
    NUM_FACILITIES = 6
    SOLVER_PATH = '/usr/local/bin/glpsol'

    # Extract graph name and create output directory structure
    GRAPH_NAME = Path(EDGELIST_FILE).stem
    OUTPUT_DIR = Path('data') / 'results' / 'compare_p_median' / GRAPH_NAME / WEIGHT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Graph: {GRAPH_NAME}")
    print(f"Weight type: {WEIGHT}")
    print(f"Output directory: {OUTPUT_DIR}")

    # %%
    print("Creating graph representations...")
    TT_ig, TT = create_graphs_from_edgelist(EDGELIST_FILE, weight=WEIGHT)
    print(f"Created graphs with {TT_ig.vcount()} vertices and {TT_ig.ecount()} edges")

    # %%
    print("Calculating distance matrices...")
    protected_data, resistance_data, distance_data = calculate_distance_matrices(TT_ig, TT)
    print(f"- Protected distances: {protected_data.shape}")
    print(f"- Resistance distances: {resistance_data.shape}")
    print(f"- Geodesic distances: {distance_data.shape}")

    # Create protected distance dictionary for analysis
    from scripts.graph_operations import graph_distances_dict

    # Get protected distances as dictionary for compatibility with existing analysis functions
    vertex_names = TT_ig.vs["name"] if "name" in TT_ig.vs.attributes() else list(range(TT_ig.vcount()))
    prot_dict = {}
    for i, row_name in enumerate(protected_data.index):
        for j, col_name in enumerate(protected_data.columns):
            prot_dict[(row_name, col_name)] = protected_data.loc[row_name, col_name]


    # %%
    def get_max_from_result_dict(result_dict, prot_dict):
        """
        Возвращает максимальное значение из списка, сформированного на основе result_dict и prot_dict.

        :param result_dict: Словарь с ключами (места) и значениями (список клиентов).
        :param prot_dict: Словарь с возможными парами (ключами) и значениями.
        :return: Максимальное значение из сформированного списка или None, если список пуст.
        """
        result_list = []

        for i in result_dict.keys():
            for u in result_dict[i]:
                value = prot_dict[(i, u)]
                result_list.append(value)

        # Возвращаем максимум или None, если список пуст
        return max(result_list) if result_list else None


    def optimize_and_compute_max(input_data, p_range, glpk_path, prot_dict):
        """
        Вызывает solve_pmedian_problem для каждого p из p_range, затем вычисляет максимум через get_max_from_result_dict.

        :param input_data: DataFrame с матрицей расстояний или затрат.
        :param p_range: Итерабельный объект с диапазоном значений p.
        :param glpk_path: Путь к исполняемому файлу glpsol.
        :param prot_dict: Словарь с парами (ключами) и их значениями для расчёта максимумов.
        :return: Список словарей с результатами для каждого p.
        """
        results = []

        for p in p_range:
            print(f"Solving P-median problem for p={p}...")
            
            # Шаг 1: Оптимизация для текущего значения p с использованием новой функции
            result_dict = solve_pmedian_problem(
                input_data,
                p,
                solver_path=glpk_path,
                verbose=True,
                use_heuristic_solution=False,
            )

            # Шаг 2: Вычисление максимального значения
            max_value = get_max_from_result_dict(result_dict, prot_dict)

            # Шаг 3: Сохранение результата
            results.append({
                "p": p,
                "max_value": max_value
            })

        return results

    # %%
    # Configuration
    glpk_path = '/usr/local/bin/glpsol'
    p_range = [i for i in range(1, 10)]

    print(f"Solver path: {glpk_path}")
    print(f"Testing p values: {p_range}")

    # %%
    print("Solving P-median problems with geodesic distances...")
    results_geodesic = optimize_and_compute_max(distance_data, p_range, glpk_path, prot_dict)

    # %%
    print("Solving P-median problems with resistance distances...")
    results_resistance = optimize_and_compute_max(resistance_data, p_range, glpk_path, prot_dict)

    # %%
    print("Solving P-median problems with protected distances...")
    results_guf = optimize_and_compute_max(protected_data, p_range, glpk_path, prot_dict)

    # %%
    def plot_comparison(results_list, labels, save_path=None):
        """
        Строит график для сравнения массивов max_value из нескольких результатов optimize_and_compute_max.

        :param results_list: Список результатов (массивов словарей) от optimize_and_compute_max.
        :param labels: Список подписей для каждого набора данных.
        :param save_path: Путь для сохранения графика (опционально).
        """
        if len(results_list) != len(labels):
            raise ValueError("Количество результатов должно совпадать с количеством подписей.")

        plt.figure(figsize=(10, 6))

        for results, label in zip(results_list, labels):
            # Извлекаем значения p и max_value
            p_values = [result['p'] for result in results]
            max_values = [result['max_value'] for result in results]

            # Построение графика
            plt.plot(p_values, max_values, marker='o', label=label, linewidth=2, markersize=6)

        # Настройка графика
        plt.xlabel("Number of facilities", fontsize=12)
        plt.ylabel("Worst distance GUF", fontsize=12)
        plt.title("P-median Coverage Analysis: Worst-case Distance Comparison", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Сохранение графика
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    # %%
    print("Creating comparison plot...")
    labels = ["Geodesic", "Resistance", "GUF"]
    plot_comparison([results_geodesic, results_resistance, results_guf], labels,
                save_path=OUTPUT_DIR / "coverage_comparison.png")

    print("\nAnalysis complete!")
    print("="*50)
    print("COVERAGE ANALYSIS SUMMARY")
    print("="*50)
    print(f"Dataset: {EDGELIST_FILE}")
    print(f"Weight type: {WEIGHT}")
    print(f"Graph: {TT_ig.vcount()} vertices, {TT_ig.ecount()} edges")
    print(f"P-values tested: {p_range}")
    print("\nFinal results:")
    print("- Geodesic:", [r['max_value'] for r in results_geodesic])
    print("- Resistance:", [r['max_value'] for r in results_resistance])
    print("- GUF:", [r['max_value'] for r in results_guf])

    # Save detailed results to files
    coverage_summary = f"""P-MEDIAN COVERAGE ANALYSIS RESULTS
    {'='*50}

    Problem Configuration:
    - Dataset: {EDGELIST_FILE}
    - Weight type: {WEIGHT}
    - Graph size: {TT_ig.vcount()} vertices, {TT_ig.ecount()} edges
    - P-values tested: {p_range}

    Coverage Analysis Results:
    - Geodesic worst distances: {[r['max_value'] for r in results_geodesic]}
    - Resistance worst distances: {[r['max_value'] for r in results_resistance]}
    - GUF worst distances: {[r['max_value'] for r in results_guf]}

    Detailed Results by P-value:
    """

    for p in p_range:
        geodesic_val = next(r['max_value'] for r in results_geodesic if r['p'] == p)
        resistance_val = next(r['max_value'] for r in results_resistance if r['p'] == p)
        guf_val = next(r['max_value'] for r in results_guf if r['p'] == p)
        
        coverage_summary += f"\nP={p}:\n"
        coverage_summary += f"  - Geodesic: {geodesic_val:.2f}\n"
        coverage_summary += f"  - Resistance: {resistance_val:.2f}\n"
        coverage_summary += f"  - GUF: {guf_val:.2f}\n"

    # Save summary to file
    with open(OUTPUT_DIR / "coverage_analysis_summary.txt", 'w') as f:
        f.write(coverage_summary)

    # Save results as CSV for further analysis
    results_df = pd.DataFrame({
        'p_value': p_range,
        'geodesic_worst_distance': [r['max_value'] for r in results_geodesic],
        'resistance_worst_distance': [r['max_value'] for r in results_resistance],
        'guf_worst_distance': [r['max_value'] for r in results_guf]
    })

    results_df.to_csv(OUTPUT_DIR / "coverage_results.csv", index=False)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Files created:")
    print("- coverage_comparison.png")
    print("- coverage_analysis_summary.txt")
    print("- coverage_results.csv")


