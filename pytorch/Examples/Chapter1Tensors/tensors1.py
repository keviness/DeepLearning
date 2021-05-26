#---Chapter1: Tensors---
import torch
import numpy as np

x = torch.empty(5,3)
print(x)
print(x.size())

y = np.array([1,2,3])
y_tensor = torch.tensor(y)
print(y_tensor)