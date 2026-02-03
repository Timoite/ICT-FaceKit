#!/bin/bash
# Quick test script - runs a small subset of configurations
# Usage: ./quick_test.sh

echo "========================================"
echo "QUICK TONGUE PARAMETER TEST"
echo "========================================"
echo ""
echo "This will test a small subset (27 configurations)"
echo "Estimated time: 30-45 minutes"
echo ""

cd /home/timoite/Documents/ICT-FaceKit

# Check if we should run the full or quick test
if [ "$1" == "full" ]; then
    echo "Running FULL grid search (245 configurations)..."
    echo "Estimated time: 4-6 hours"
    echo ""
    ./tongue_scripts/run_grid_search.sh
else
    echo "Running QUICK test (27 configurations)..."
    echo "To run full test: $0 full"
    echo ""
    
    # Create a quick test version with reduced parameters
    cat > /tmp/quick_test.py << 'EOFPYTHON'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/timoite/Documents/ICT-FaceKit')

# Import and modify parameters
from tongue_scripts.test_tongue_grid_search import *

# Override with smaller parameter set
ROTATION_RANGE = [0, 10, 20]
THICKNESS_RANGE = [1.0, 2.0, 3.0]
STD_SCALAR_RANGE = [0.15, 0.20, 0.30]

# Update output directory
TEST_OUTPUT_DIR = SCRIPT_DIR / "tongue_param_tests_quick"

print("="*60)
print("QUICK TEST MODE")
print("="*60)
print(f"Testing {len(ROTATION_RANGE)} × {len(THICKNESS_RANGE)} × {len(STD_SCALAR_RANGE)} = {len(ROTATION_RANGE) * len(THICKNESS_RANGE) * len(STD_SCALAR_RANGE)} configurations")
print(f"Output: {TEST_OUTPUT_DIR}")
print()

# Run the test
run_parameter_grid_test()
EOFPYTHON

    # Run the quick test
    uv run python /tmp/quick_test.py 2>&1 | tee tongue_scripts/quick_test.log
    
    echo ""
    echo "Quick test complete!"
    echo "Results saved to: tongue_scripts/tongue_param_tests_quick/"
    echo ""
    echo "To analyze results:"
    echo "  python tongue_scripts/analyze_grid_results.py"
fi
