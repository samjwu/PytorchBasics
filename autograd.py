import torch

x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad=True) # weight
b = torch.randn(3, requires_grad=True) # bias
z = torch.matmul(x, w)+b # layer
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(f"Gradient for weight = {w.grad}")
print(f"Gradient for bias = {b.grad}")

z = torch.matmul(x, w)+b
print(f"Gradient Tracking Enabled: {z.requires_grad}")

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(f"Gradient Tracking Enabled: {z.requires_grad}")

z = torch.matmul(x, w)+b
z_det = z.detach()
print(f"Gradient Tracking Enabled: {z_det.requires_grad}")

input_vector = torch.eye(4, 5, requires_grad=True)
output_vector = (input_vector+1).pow(2).t()
output_vector.backward(torch.ones_like(output_vector), retain_graph=True)
print(f"First call\n{input_vector.grad}")
output_vector.backward(torch.ones_like(output_vector), retain_graph=True)
print(f"\nSecond call\n{input_vector.grad}")
input_vector.grad.zero_()
output_vector.backward(torch.ones_like(output_vector), retain_graph=True)
print(f"\nCall after zeroing gradients\n{input_vector.grad}")
