import json
import os
import numpy as np
os.chdir('/Users/gavrielhannuna/lrgb_source_code/training_logs')
# List of seed numbers
seeds = [2025, 42, 5, 123,18]

# Path to the JSON files
file_pattern = "seed_{seed_num}_logs.json"


# Key to calculate the statistics
val_key = "val_mae"  # Key for validation metric
test_key = "test_mae"  # Key for test metric
# Store the averages for the last 10 epochs for each seed
# Store the selected test_mae values for each seed
test_mae_values = []

for seed in seeds:
    # Load the JSON file
    file_path = file_pattern.format(seed_num=seed)
    with open(file_path, 'r') as f:
        logs = json.load(f)

    # Find the epoch with the lowest val_mae
    min_val_mae_entry = min(logs, key=lambda entry: entry[val_key])

    # Get the test_mae corresponding to the lowest val_mae
    test_mae_values.append(min_val_mae_entry[test_key])

# Calculate the final mean and standard deviation
final_mean = np.mean(test_mae_values)
final_std = np.std(test_mae_values)

# Calculate SD as a percentage of the mean
sd_percentage = (final_std / final_mean) * 100

print(f"Final Mean ({test_key}): {final_mean}")
print(f"Final Standard Deviation ({test_key}): {final_std}")
print(f"Standard Deviation as Percentage of Mean ({test_key}): {sd_percentage:.2f}%")