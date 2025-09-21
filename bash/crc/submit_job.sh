#!/bin/bash

# Script to submit job after cleaning up previous log files
# Usage: ./submit_job.sh [script_file]
# Default script: ./bash/crc/base.script

# Set default job script
JOB_SCRIPT="${1:-./bash/crc/base.script}"

# Check if the job script exists
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "Error: Job script '$JOB_SCRIPT' not found!"
    exit 1
fi

echo "📋 Processing job script: $JOB_SCRIPT"

# Extract the log file path from the qsub -o line
LOG_FILE=$(grep -E "^#\$ -o " "$JOB_SCRIPT" | sed 's/^#\$ -o //')

if [[ -n "$LOG_FILE" ]]; then
    # Expand tilde to home directory if present
    LOG_FILE_EXPANDED="${LOG_FILE/#\~/$HOME}"
    
    echo "🗂️  Found log output: $LOG_FILE"
    
    # Check if log file exists and remove it
    if [[ -f "$LOG_FILE_EXPANDED" ]]; then
        echo "🗑️  Removing existing log file: $LOG_FILE_EXPANDED"
        rm -f "$LOG_FILE_EXPANDED"
        
        if [[ $? -eq 0 ]]; then
            echo "✅ Successfully removed old log file"
        else
            echo "⚠️  Warning: Failed to remove log file"
        fi
    else
        echo "ℹ️  Log file does not exist yet (clean start)"
    fi
    
    # Create log directory if it doesn't exist
    LOG_DIR=$(dirname "$LOG_FILE_EXPANDED")
    if [[ ! -d "$LOG_DIR" ]]; then
        echo "📁 Creating log directory: $LOG_DIR"
        mkdir -p "$LOG_DIR"
    fi
else
    echo "⚠️  Warning: No log output line found in $JOB_SCRIPT"
fi

# Submit the job
echo "🚀 Submitting job..."
qsub "$JOB_SCRIPT"

if [[ $? -eq 0 ]]; then
    echo "✅ Job submitted successfully!"
    echo "📊 Check job status with: qstat"
    if [[ -n "$LOG_FILE" ]]; then
        echo "📄 Monitor log with: tail -f $LOG_FILE_EXPANDED"
    fi
else
    echo "❌ Failed to submit job!"
    exit 1
fi
