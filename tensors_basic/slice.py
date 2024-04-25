import torch


torch1 = torch.arange(10)
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(torch1)

# tensor(7)
print(f"Grab item from index: {torch1[7]}")

torch2 = torch1.reshape(5, 2)
print(f"Reshape: {torch2}")

print(f"Grab items from given column: {torch2[:, 0]}")
