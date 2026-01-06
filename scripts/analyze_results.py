#!/usr/bin/env python3
"""
Generate analysis report for trained Lorenz quantum circuit.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from analysis import generate_analysis_report


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze trained Lorenz circuit")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory with training results")
    parser.add_argument("--classical-data", type=str, default=None,
                       help="Path to classical solution data")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for analysis report")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Default classical data path
    if args.classical_data is None:
        classical_data_path = project_root / "data" / "lorenz_classical_solution.npz"
    else:
        classical_data_path = Path(args.classical_data)
    
    if not classical_data_path.exists():
        print(f"Error: Classical data not found: {classical_data_path}")
        return 1
    
    output_path = Path(args.output) if args.output else None
    
    generate_analysis_report(
        results_dir=results_dir,
        classical_data_path=classical_data_path,
        output_path=output_path
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

