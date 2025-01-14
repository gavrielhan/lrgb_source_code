import json
import os
import numpy as np
os.chdir('/Users/gavrielhannuna/lrgb_source_code/training_logs/training_logs_sample')
# List of seed numbers
seeds = [2025, 42, 5, 123,18]
LRGB_d = 'Peptides-struct'
LRGB_sp = LRGB_d.split('-')[1]
# Path to the JSON files

file_pattern = "seed_{seed_num}_logs_{LRGB_sp}.json"



val_key = "val_metric"  # Key for validation metric
test_key = "test_metric"  # Key for test metric


# Store the averages for the last 10 epochs for each seed
# Store the selected test_mae values for each seed
test_values = []

for seed in seeds:
    # Load the JSON file
    file_path = file_pattern.format(seed_num=seed, LRGB_sp = LRGB_sp)
    with open(file_path, 'r') as f:
        logs = json.load(f)

    # Find the epoch with the lowest val_mae
    if LRGB_d == 'Peptides-struct':
        min_val_mae_entry = min(logs, key=lambda entry: entry[val_key])
        # Get the test_mae corresponding to the lowest val_mae
        test_values.append(min_val_mae_entry[test_key])
    else:
        # Find the epoch with the highest val_ap
        max_val_ap_entry = max(logs, key=lambda entry: entry[val_key])
        # Get the test_mae corresponding to the lowest val_mae
        test_values.append(max_val_ap_entry[test_key])



# Calculate the final mean and standard deviation
final_mean = np.mean(test_values)
final_std = np.std(test_values)

# Calculate SD as a percentage of the mean
sd_percentage = (final_std / final_mean) * 100

print(f"Final Mean ({test_key}): {final_mean}")
print(f"Final Standard Deviation ({test_key}): {final_std}")
print(f"Standard Deviation as Percentage of Mean ({test_key}): {sd_percentage:.2f}%")
