"""
Optimization module for P-median problems.

This module contains functions for creating and solving P-median optimization models.
"""

import pyomo.environ as pyo
import pandas as pd
from pyomo.solvers.plugins.solvers.GLPK import GLPKSHELL
from typing import Dict, List


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
        verbose: Whether to show solver output
        
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