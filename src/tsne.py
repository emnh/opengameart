import sys
sys.path += ['/mnt/d/dev/opengameart/bhtsne']
import bhtsne
import numpy as np
import json

data = []
lines = open('list.txt').readlines()
for line in lines:
    d = json.loads(line.rstrip())
    data.append(d["features"])
data = np.array(data)
print(data.shape)

embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])
outfd = open('embeddings.txt', 'w')
outfd.write(json.dumps(embedding_array.tolist()))
outfd.close()
#print(embedding_array)