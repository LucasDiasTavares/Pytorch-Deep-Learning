import torch
import numpy as np

'''
Add
Subtract
Multiply
Divide
Remainders
Exponents
Reassignment
'''

tensor_a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
tensor_b = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# Addition
tensor_a + tensor_b
print(f"Addition: {tensor_a + tensor_b}")
# Addition Function
print(f"Addition Function: {torch.add(tensor_a, tensor_b)}")
print(f"Addition Function Another Way: {tensor_a.add(tensor_b)}")

# Subtraction
print(f"Subtraction: {tensor_b - tensor_a}")
# Subtraction Function
print(f"Subtraction Function: {torch.add(tensor_a, tensor_b)}")

# Multiplication
print(f"Multiplication: {tensor_a * tensor_b}")
# Multiplication Function
print(f"Multiplication Function: {torch.mul(tensor_a, tensor_b)}")

# Division
print(f"Division: {tensor_b / tensor_a}")
# Division Function
print(f"Division Function: {torch.div(tensor_b, tensor_a)}")

# Remainder
print(f"Remainder: {tensor_b % tensor_a}")
# Remainder Function
print(f"Remainder Function: {torch.remainder(tensor_b, tensor_a)}")

# Exponents - power
# Docs https://pytorch.org/docs/stable/generated/torch.pow.html
print(f"Exponents Function: {torch.pow(tensor_a, tensor_b)}")















