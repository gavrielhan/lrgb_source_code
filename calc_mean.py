import json
import os
import numpy as np

# List of seed numbers
seeds = [18, 2025, 42, 5, 128]

# Path to the JSON files
file_pattern = "seed_{seed_num}_logs.json"

# Key to calculate the average and standard deviation for
key_to_average = "test_mae"

# Store the averages for the last 10 epochs for each seed
seed_averages = []

for seed in seeds:
    # Load the JSON file
    file_path = file_pattern.format(seed_num=seed)
    with open(file_path, 'r') as f:
        logs = json.load(f)

    # Get the last 10 epochs
    last_10_epochs = logs[-10:]

    # Extract the values for the specified key
    values = [entry[key_to_average] for entry in last_10_epochs]

    # Calculate the average for the last 10 epochs
    avg = np.mean(values)
    seed_averages.append(avg)

# Calculate the final mean and standard deviation
final_mean = np.mean(seed_averages)
final_std = np.std(seed_averages)

print(f"Final Mean ({key_to_average}): {final_mean}")
print(f"Final Standard Deviation ({key_to_average}): {final_std}")
