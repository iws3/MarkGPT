# from gguf import GGUFReader
# import glob, os
# path = glob.glob(os.path.expanduser('~/llama.cpp/models/*.gguf'))[0]

# r = GGUFReader(path)
# for key in list(r.fields)[:40]:
#     print(key)
# for t in r.tensors[:12]:
#     print(t.name, list(t.shape), t.tensor_type.name)

from gguf import GGUFReader
import glob, os
path = glob.glob(os.path.expanduser('~/llama.cpp/models/*.gguf'))[0]
r = GGUFReader(path)        
for key in list(r.fields)[:40]:
    print(key)
for t in r.tensors[:12]:
    print(t.name, list(t.shape), t.tensor_type.name)