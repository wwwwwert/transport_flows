"""
Optimization module for P-median problems.

This module contains functions for creating and solving P-median optimization models.
"""

import pyomo.environ as pyo
import pandas as pd
from pyomo.opt.solver import SystemCallSolver
from typing import Dict, List
import numpy as np


def calculate_total_cost(facilities, clients_coords, dist_matrix_np):
    """Вспомогательная функция для расчета стоимости решения."""
    # Для каждого клиента находим расстояние до ближайшего из открытых объектов
    client_distances = dist_matrix_np[:, facilities]
    min_distances = np.min(client_distances, axis=1)
    return np.sum(min_distances)


def solve_pmedian_teitz_bart(
        distance_matrix: pd.DataFrame, 
        p: int
    ) -> Dict[str, any]:
    """
    Решает задачу P-median приближенно с помощью эвристики Teitz and Bart.
    """
    print("Starting Teitz and Bart heuristic...")
    nodes = np.array(distance_matrix.index)
    n = len(nodes)
    dist_matrix_np = distance_matrix.to_numpy()
    
    # 1. Инициализация: выбираем p случайных локаций
    current_facilities_indices = np.random.choice(n, p, replace=False)
    current_cost = calculate_total_cost(current_facilities_indices, np.arange(n), dist_matrix_np)
    
    print(f"Initial random solution cost: {current_cost}")

    while True:
        best_improvement = 0
        best_swap = None # (индекс убираемой локации, индекс добавляемой локации)

        non_facilities_indices = np.setdiff1d(np.arange(n), current_facilities_indices)
        
        # 2. Итеративное улучшение
        # Перебираем каждую локацию В решении
        for i_to_remove in current_facilities_indices:
            # Перебираем каждую локацию НЕ в решении
            for j_to_add in non_facilities_indices:
                
                # Создаем временный набор локаций после обмена
                prospective_facilities = np.copy(current_facilities_indices)
                prospective_facilities[prospective_facilities == i_to_remove] = j_to_add
                
                # Рассчитываем новую стоимость
                new_cost = calculate_total_cost(prospective_facilities, np.arange(n), dist_matrix_np)
                
                improvement = current_cost - new_cost
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_swap = (i_to_remove, j_to_add)

        # 3. Остановка или применение лучшего обмена
        if best_improvement > 0:
            i_rem, j_add = best_swap
            current_facilities_indices[current_facilities_indices == i_rem] = j_add
            current_cost -= best_improvement
            print(f"Found improvement: {best_improvement:.2f}. New cost: {current_cost:.2f}")
        else:
            print("No further improvements possible. Heuristic finished.")
            break
            
    # Формируем результат в том же формате, что и у MILP
    final_facility_names = nodes[current_facilities_indices]
    result_dict = {name: [] for name in final_facility_names}
    
    client_distances = dist_matrix_np[:, current_facilities_indices]
    # Находим индекс ближайшего объекта для каждого клиента
    closest_facility_map = np.argmin(client_distances, axis=1)

    for client_idx, facility_rel_idx in enumerate(closest_facility_map):
        client_name = nodes[client_idx]
        facility_name = final_facility_names[facility_rel_idx]
        result_dict[facility_name].append(client_name)

    return {
        "solution": result_dict,
        "objective_value": current_cost
    }


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
    print('Parameters')
    # Create index mappings for fast lookup
    node_to_idx = {node: idx for idx, node in enumerate(distance_matrix.index)}
    
    def cost_init(model, i, j):
        return c[node_to_idx[i]][node_to_idx[j]]
    
    model.c = pyo.Param(
        model.M, model.N,
        initialize=cost_init,
        within=pyo.NonNegativeReals
    )
    
    # Variables
    print('Variables')
    model.x = pyo.Var(model.M, model.N, within=pyo.Binary)  # Assignment variables
    model.y = pyo.Var(model.N, within=pyo.Binary)  # Facility location variables
    
    # Objective function
    print('Objective function')
    def obj_rule(model):
        return sum(model.c[i, j] * model.x[i, j] for i in model.M for j in model.N)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Constraints
    print('Constraints')
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
        verbose: bool = False,
        use_warm_start: bool = True
    ) -> Dict[str, List[str]]:
    """
    Solve P-median problem for given distance matrix.
    Install GLPK like this:
    ```
    sudo apt-get install glpk-utils
    which glpsol
    ```
    
    Args:
        distance_matrix: DataFrame containing distance matrix
        p: Number of facilities to locate
        solver_path: Path to GLPK solver executable
        verbose: Whether to show solver output
        use_warm_start: Whether to use heuristic solution as warm start
        
    Returns:
        Dictionary with facility locations as keys and assigned clients as values
    """
    # Create model
    model = create_pmedian_model(distance_matrix, p)
    
    # Get warm start solution if requested
    if use_warm_start:
        print('Computing warm start solution with Teitz-Bart heuristic...')
        heuristic_result = solve_pmedian_teitz_bart(distance_matrix, p)
        heuristic_solution = heuristic_result["solution"]
        heuristic_objective = heuristic_result["objective_value"]
        
        print(f'Heuristic solution objective: {heuristic_objective:.2f}')
        print(f'Heuristic facilities: {list(heuristic_solution.keys())}')
        
        # Set warm start values
        print('Setting warm start values...')
        
        # Initialize all variables to 0
        for j in model.N:
            model.y[j].set_value(0)
            for i in model.M:
                model.x[i, j].set_value(0)
        
        # Set facility location variables based on heuristic solution
        for facility in heuristic_solution.keys():
            model.y[facility].set_value(1)
            
            # Set assignment variables based on heuristic solution
            for client in heuristic_solution[facility]:
                model.x[client, facility].set_value(1)
        
        print('Warm start values set successfully.')
    
    # Solve
    print('Initializing solver')
    # solver = pyo.SolverFactory('glpk', executable=solver_path)
    solver: SystemCallSolver = pyo.SolverFactory('scip', executable='/usr/bin/scip')
    # solver: SystemCallSolver = pyo.SolverFactory('cbc', executable='/usr/bin/cbc')
    solver.options['lp/threads'] = 16

    print('Starting exact solving')
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