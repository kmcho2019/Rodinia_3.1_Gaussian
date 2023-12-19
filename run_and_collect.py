import subprocess
import csv

# Define the list of configuration sizes
config_sizes = [16, 32, 64, 256, 1024, 2048, 4096, 8192, 16384]

# Create a CSV file for storing the collected data
with open('output_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['input_size', 'time_cuda_kernel', 'time_total'])

    # Loop through each configuration size
    for size in config_sizes:
        for _ in range(5):  # Run the script 5 times for each configuration
            # Run the provided script and capture the output
            result = subprocess.run(['python', 'triton_implementation.py', '-s', str(size), '-q'], stdout=subprocess.PIPE, text=True)

            # Extract relevant information from the script's output
            lines = result.stdout.split('\n')
            time_cuda_kernel = None
            time_total = None
            for line in lines:
                if 'Time for CUDA kernels' in line:
                    time_cuda_kernel = float(line.split(': ')[1].split(' sec')[0])
                elif 'Total time (including memory transfers)' in line:
                    time_total = float(line.split(': ')[1].split(' sec')[0])

            # Write the collected data to the CSV file
            csv_writer.writerow([size, time_cuda_kernel, time_total])

print("Data collection complete. Saved to output_data.csv.")

