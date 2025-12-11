#!/bin/bash

# Script to delete directories corresponding to Daytona errors, DaytonaNotFoundError, or CancelledError in result.json
# Usage: ./delete_daytona_errors.sh jobs/2025-11-24__01-48-28

if [ $# -eq 0 ]; then
    echo "Usage: $0 <jobs_directory>"
    echo "Example: $0 jobs/2025-11-24__01-48-28"
    exit 1
fi

JOB_DIR="$1"
RESULT_FILE="$JOB_DIR/result.json"

# Check if the job directory exists
if [ ! -d "$JOB_DIR" ]; then
    echo "Error: Directory '$JOB_DIR' not found!"
    exit 1
fi

# Check if result.json exists
if [ ! -f "$RESULT_FILE" ]; then
    echo "Error: '$RESULT_FILE' not found!"
    exit 1
fi

echo "Processing $RESULT_FILE..."
echo ""

# Extract directories with DaytonaError, DaytonaNotFoundError, or CancelledError using Python
# The structure is: .stats.evals[...].exception_stats.<error_type> = [array of trial names]
ERROR_TRIALS=$(python3 << PYEOF
import json

ERROR_KEYS = ["DaytonaError", "DaytonaNotFoundError", "CancelledError"]

try:
    with open("$RESULT_FILE", 'r') as f:
        data = json.load(f)
    
    trials = []
    if 'stats' in data and 'evals' in data['stats']:
        for eval_name, eval_data in data['stats']['evals'].items():
            if 'exception_stats' in eval_data and eval_data['exception_stats']:
                for error_key in ERROR_KEYS:
                    if error_key in eval_data['exception_stats']:
                        trials.extend(eval_data['exception_stats'][error_key])
    
    for trial in trials:
        print(trial)
except Exception as e:
    import sys
    print(f"Error: {e}", file=sys.stderr)
PYEOF
)

if [ -z "$ERROR_TRIALS" ]; then
    echo "No DaytonaError, DaytonaNotFoundError, or CancelledError entries found in result.json"
    exit 0
fi

# Count the number of error trials
COUNT=$(echo "$ERROR_TRIALS" | wc -l)
echo "Found $COUNT directories with DaytonaError, DaytonaNotFoundError, or CancelledError"
echo ""

# Delete each directory
DELETED=0
NOT_FOUND=0

for TRIAL_NAME in $ERROR_TRIALS; do
    TRIAL_DIR="$JOB_DIR/$TRIAL_NAME"
    
    if [ -d "$TRIAL_DIR" ]; then
        echo "Deleting: $TRIAL_DIR"
        rm -rf "$TRIAL_DIR"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully deleted"
            ((DELETED++))
        else
            echo "  ✗ Failed to delete"
        fi
    else
        echo "Skipping: $TRIAL_NAME (directory not found)"
        ((NOT_FOUND++))
    fi
done

echo ""
echo "Summary:"
echo "  - Deleted: $DELETED directories"
echo "  - Not found: $NOT_FOUND directories"
echo "  - Total processed: $COUNT entries"