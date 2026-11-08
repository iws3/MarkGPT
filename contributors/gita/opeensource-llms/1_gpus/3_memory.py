import torch
x = torch.randn(1024, 1024, 64, device='cuda')     # 64M numbers
print(torch.cuda.memory_allocated() / 1e6, 'MB used')   # about 268 MB
x16 = x.half()
print(torch.cuda.memory_allocated() / 1e6, 'MB used')   # the half copy adds half as much
del x, x16
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1e6, 'MB used')   # back near zero