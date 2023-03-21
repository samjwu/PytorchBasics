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
print(f"Ones Tensor:\n {ones_tensor}")
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
print(f"4x4 Ones Tensor:\n {tensor}")
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(f"4x4 Ones Tensor after applying tensor[:,1] = 0:\n {tensor}\n")

t1 = torch.cat([tensor, tensor], dim=1)
t2 = torch.stack([tensor, tensor], dim=1)
print(f"Concatenated tensor using cat:\n {t1}")
print(f"Concatenated tensor using stack:\n {t2}\n")

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(f"Matrix multiplication of tensor and its transpose using @:\n {y1}")
print(f"Matrix multiplication of tensor and its transpose using numpy.matmul:\n {y2}")
print(f"Matrix multiplication of tensor and its transpose using torch.matmul:\n {y3}\n")

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"Element-wise multiplication of tensor using *:\n {z1}")
print(f"Element-wise multiplication of tensor using numpy.mul:\n {z2}")
print(f"Element-wise multiplication of tensor using torch.mul:\n {z3}\n")

agg = tensor.sum()
agg_item = agg.item()
print(f"Aggregated tensor: {agg}")
print(f"Aggregated tensor type: {type(agg)}")
print(f"Aggregated tensor item: {agg_item}")
print(f"Aggregated tensor item type: {type(agg_item)}\n")

print(f"Tensor before adding 5 in-place:\n {tensor}")
tensor.add_(5)
print(f"Tensor after adding 5 in-place:\n {tensor}\n")
