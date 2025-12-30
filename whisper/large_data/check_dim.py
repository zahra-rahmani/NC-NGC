import torch

dataset = torch.load("data_large_one_of_four_eln.pt")

print("Total examples:", len(dataset))

example = dataset[0]

print("Hypotheses:", example["hypotheses"])
print("Label:", example["label"])

print("ELN shape:", example["eln"].shape)
