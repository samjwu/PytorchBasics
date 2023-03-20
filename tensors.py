import torch
import numpy as np

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}\n")

data = [[1, 2],[3, 4]]
data_tensor = torch.tensor(data)
print(f"2x2 Array:\n {data}")
print(f"2x2 Torch Tensor:\n {data_tensor}\n")

np_array = np.array(data)
converted_tensor = torch.from_numpy(np_array)
converted_array = converted_tensor.numpy()
print(f"NumPy Array:\n {np_array}")
print(f"Torch Tensor from NumPy Array:\n {converted_tensor}")
print(f"NumPy Array from Torch Tensor:\n {converted_array}\n")

ones_tensor = torch.ones_like(data_tensor) # retains the properties of data_tensor
print(f"Ones Tensor:\n {ones_tensor}\n")
rand_tensor = torch.rand_like(data_tensor, dtype=torch.float) # overrides the datatype of data_tensor
print(f"Random Tensor:\n {rand_tensor}\n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor:\n {rand_tensor}")
print(f"Ones Tensor:\n {ones_tensor}")
print(f"Zeroes Tensor:\n {zeros_tensor}\n")

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n")

# move tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(f"{tensor}\n")

t1 = torch.cat([tensor, tensor], dim=1)
t2 = torch.stack([tensor, tensor], dim=1)
print(t1)
print(t2)
