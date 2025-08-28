import torch

# Load your dataset
dataset = torch.load("data_large_one_of_four_eln.pt")

# Check number of examples
print("Total examples:", len(dataset))

# Pick the first example
example = dataset[0]

# Print hypotheses and label
print("Hypotheses:", example["hypotheses"])
print("Label:", example["label"])

# Check ELN tensor shape
print("ELN shape:", example["eln"].shape)
