#!/bin/bash
#
# MPI Communication Validation Launcher
#
# Usage:
#   bash run.sh                           # Basic validation with 2 processes
#   bash run.sh --np 4                   # Use 4 processes
#   bash run.sh --np 4 --performance     # Include performance tests
#   bash run.sh --np 8 --hosts n1,n2     # Multi-node
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
NP=2
HOSTS=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --np)
            NP="$2"
            shift 2
            ;;
        --hosts)
            HOSTS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "============================================================"
echo "MPI Communication Validation Recipe"
echo "============================================================"
echo "Processes: $NP"
if [ -n "$HOSTS" ]; then
    echo "Hosts: $HOSTS"
fi
echo ""

# Check MPI is available
if ! command -v mpirun &> /dev/null; then
    echo "ERROR: mpirun not found. Ensure MPI is installed and in PATH."
    echo "For MVAPICH-Plus: export PATH=\$MVAPICH_HOME/bin:\$PATH"
    exit 1
fi

# Build mpirun command
if [ -n "$HOSTS" ]; then
    MPIRUN_CMD="mpirun -np $NP -hosts $HOSTS"
else
    MPIRUN_CMD="mpirun -np $NP"
fi

# Run validation
$MPIRUN_CMD python validate.py $EXTRA_ARGS
