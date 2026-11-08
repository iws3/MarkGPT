import torch
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
a = torch.tensor([1.0, 2.0, 3.0])        # on the CPU
b = a.to(dev)                              # copied to the GPU
print(a.device, b.device)
c = b * 10                                 # computed on the GPU
print(c, c.device)
print(c.cpu().numpy())                     # copy back to the CPU to use with NumPy