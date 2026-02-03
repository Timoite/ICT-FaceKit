#!/bin/bash
# Wrapper script to run the grid search with proper environment
# Usage: ./run_grid_search.sh

cd /home/timoite/Documents/ICT-FaceKit

# Activate environment and run
uv run python tongue_scripts/test_tongue_grid_search.py 2>&1 | tee tongue_scripts/grid_search.log

echo "Grid search complete! Check results in: tongue_scripts/tongue_param_tests/"
