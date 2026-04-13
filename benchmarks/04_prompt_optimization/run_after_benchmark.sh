#!/bin/bash
# Wait for benchmark_usecase.py to finish, then run prompt experiment
echo "[$(date)] Waiting for benchmark_usecase.py (PID check)..."
while pgrep -f "benchmark_usecase.py" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] benchmark_usecase.py finished. Starting prompt experiment..."
cd /home/azureuser/workspace/h01
python3 -u awd_benchmark_02/prompt_experiment.py
echo "[$(date)] Prompt experiment finished."
