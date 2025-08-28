import json

input_file = 'output_data_test.json'
output_file = 'first_1000_test_samples.json'

# Load the original JSON file
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract the first 1000 samples
subset = data[:1000]

# Write to a new JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print(f"Saved first 1000 samples to {output_file}")
