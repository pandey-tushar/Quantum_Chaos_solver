"""
Utility functions for loading and saving data.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict


def load_classical_solution(data_dir: Path = None) -> Dict:
    """
    Load the classical Lorenz solution from saved file.
    
    Args:
        data_dir: Directory containing the data file. If None, uses default data directory.
    
    Returns:
        Dictionary containing:
            - 't': time array
            - 'x': x trajectory
            - 'y': y trajectory
            - 'z': z trajectory
            - 'initial_state': initial conditions [x0, y0, z0]
            - 'sigma', 'rho', 'beta': Lorenz parameters
            - 't_span': [t_start, t_end]
            - 'n_steps': number of time steps
    """
    if data_dir is None:
        # Assume we're in the project root
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
    
    solution_file = data_dir / "lorenz_classical_solution.npz"
    
    if not solution_file.exists():
        raise FileNotFoundError(
            f"Classical solution file not found: {solution_file}\n"
            "Please run scripts/generate_classical_solution.py first."
        )
    
    data = np.load(solution_file)
    
    return {
        't': data['t'],
        'x': data['x'],
        'y': data['y'],
        'z': data['z'],
        'initial_state': data['initial_state'],
        'sigma': float(data['sigma']),
        'rho': float(data['rho']),
        'beta': float(data['beta']),
        't_span': data['t_span'],
        'n_steps': int(data['n_steps'])
    }


def save_training_data(data: Dict, filename: str, results_dir: Path = None):
    """
    Save training data (e.g., quantum solution, loss history) to file.
    
    Args:
        data: Dictionary of data to save
        filename: Name of the file (without extension)
        results_dir: Directory to save to. If None, uses default results directory.
    """
    if results_dir is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results"
    
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f"{filename}.npz"
    
    np.savez(output_file, **data)
    return output_file

