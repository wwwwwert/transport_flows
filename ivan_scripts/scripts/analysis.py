"""
Analysis module for P-median solutions.

This module contains functions for analyzing and processing P-median solution results.
"""

import pandas as pd
from typing import Dict, List, Tuple


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


def create_results_dataframe(geodesic_distances: List[int], 
                           resistance_distances: List[int], 
                           protected_distances: List[int]) -> pd.DataFrame:
    """
    Create DataFrame with analysis results for different distance types.
    
    Args:
        geodesic_distances: List of distances from geodesic solution
        resistance_distances: List of distances from resistance solution
        protected_distances: List of distances from protected solution
        
    Returns:
        DataFrame with distance analysis results
    """
    df = pd.DataFrame({
        'geodesic': geodesic_distances,
        'resistance': resistance_distances,
        'protected': protected_distances
    })
    return df