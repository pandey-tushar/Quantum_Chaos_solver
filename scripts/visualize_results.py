#!/usr/bin/env python3
"""
Create visualizations for trained Lorenz quantum circuit.

This script loads training results and creates comparison plots.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from visualization import create_all_visualizations


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize trained Lorenz circuit")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Directory with training results")
    parser.add_argument("--classical-data", type=str, default=None,
                       help="Path to classical solution data")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for plots")
    
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
        print("Run: python scripts/generate_classical_solution.py")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Classical data: {classical_data_path}")
    print(f"Output directory: {output_dir or results_dir / 'plots'}")
    print("=" * 80)
    print()
    
    create_all_visualizations(
        results_dir=results_dir,
        classical_data_path=classical_data_path,
        output_dir=output_dir
    )
    
    print()
    print("=" * 80)
    print("✓ VISUALIZATION COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

