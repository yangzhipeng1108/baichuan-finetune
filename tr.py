import torch

a =  torch.tensor([[1,1,1,0,0,0],[1,1,1,1,0,0]])

for i in range(len(a)):
    print(i)
    print(a[1,:])
    a = torch.nonzero(a[i,:] == 0,as_tuple=True)
    print(a)

