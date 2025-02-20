import torch
import numpy as np


'''
Initialising Tensors
'''
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor1 = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, device=dev, requires_grad=False)
print(tensor1)

print(type(tensor1), tensor1.dtype, tensor1.shape, tensor1.device, tensor1.requires_grad)

tensor2 = torch.from_numpy(np.array([1,2,3]))
print(tensor2, type(tensor2))

'''
Random Initialisation
'''
t = torch.empty(size=(3, 3))
print(t)

t = torch.zeros(size=(3,3))
print(t)

t = torch.ones(size=(3,3))
print(t)

t = torch.eye(3)
print(t)

t = torch.rand(size=(3,3))
print(t)

t = torch.arange(start=0, end=5, step=1)
print(t)

t = torch.linspace(start=0, end=5, steps=10)
print(t)

t = torch.empty(size=(3,3)).normal_(mean=0, std=1)
print(t)

t = torch.diag(torch.eye(3))
print(t)

'''
Examples of Broadcasting
'''

x = torch.tensor([[1,2,3], [4,5,6]])
y = torch.tensor(5)

z = x + y
print(z)

x = torch.tensor([[1,2,3], [4,5,6]])
y = torch.tensor([1,2,3])
z = x + y
print(z)

x = torch.tensor([[1,2,3], [4,5,6]])
y = torch.tensor([[10], [20]])
z = x + y
print(z)

try:
    x = torch.tensor([[1,2,3], [4,5,6]])
    y = torch.tensor([10, 20])
    z = x + y
    print(z)
except Exception as e:
    print(e, "; broadcasting not possible")

x = torch.tensor([[1,2], [3,4]])
y = torch.tensor([10, 20])
z = x * y 
print(z)


'''
Other Useful features
'''
t = torch.tensor([[1,2,3], [4,5,6]])
x = torch.sum(t, dim=0)  # dim = 0 or axis = 0 means column wise (downside)
print(x)

values, _ = torch.max(t, dim=0)
print(values)

values, _ = torch.min(t, dim=1)
print(values)

z = torch.argmax(t, dim=0)
print(z)

z = torch.argmax(t, dim=1)
print(z)

'''
Reshaping Tensors
'''
x = torch.rand(size=(9,))
y = x.view(3,3)
print(y)

z = x.reshape(3, 3)
print(z)

x = torch.rand(size=(3, 3))
y = torch.rand(size=(3, 3))
z = torch.cat((x, y), dim=0)
print(z)

z = x.view(-1)  # unfolding
print(z)