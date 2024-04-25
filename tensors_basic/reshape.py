import torch


torch1 = torch.arange(10)
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"Original: {torch1}")

# Reshape

torch_reshape = torch1.reshape(2, 5)
# tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])
print(f"torch_reshape: {torch_reshape}")

# Reshape but don't know how many items exists
torch2 = torch.arange(10)

torch2_reshape = torch2.reshape(5, -1)  # or torch2_reshape = torch2.reshape(-1, 5)
print(f"torch2_reshape: {torch2_reshape}")
