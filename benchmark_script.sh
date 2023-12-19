#!/bin/bash

# Define the output CSV file
output_file="benchmark_results.csv"

# Write the header to the CSV file
echo "input_size,time_cuda_kernel,time_total" > "$output_file"

# Array of input sizes
sizes=(16 32 64 256 1024 2048 4096 8192 16384)

# Iterate over each size
for size in "${sizes[@]}"; do
    # Perform 5 runs for each size
    for run in {1..5}; do
        # Run the command and capture the output
        output=$(./gaussian -s "$size" -q)

        # Extract the time measurements
        time_cuda_kernel=$(echo "$output" | grep "Time for CUDA kernels:" | awk '{print $5}')
        time_total=$(echo "$output" | grep "Time total" | awk '{print $6}')

        # Append the results to the CSV file
        echo "$size,$time_cuda_kernel,$time_total" >> "$output_file"
    done
done

echo "Benchmarking completed. Results saved in $output_file."

