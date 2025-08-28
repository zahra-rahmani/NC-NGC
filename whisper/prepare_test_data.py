import json

# Load your current test data
with open("output_data_test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

transformed_data = []

for item in data:
    candidates = item["input"]
    if len(candidates) == 0:
        continue  # skip empty ones

    input1 = candidates[0]  # best hypothesis (can be changed if needed)
    input2 = " | ".join(candidates[1:]) if len(candidates) > 1 else None
    output = item["output"]

    new_item = {
        "input1": input1,
        "input2": input2,
        "output": output
    }
    transformed_data.append(new_item)

# Save it to a new file
with open("transformed_test_data.json", "w", encoding="utf-8") as f:
    json.dump(transformed_data, f, ensure_ascii=False, indent=2)
