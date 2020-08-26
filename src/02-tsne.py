#!/usr/bin/env python3
import sys
#sys.path += ['/mnt/d/dev/opengameart/bhtsne']
#import bhtsne
import numpy as np
import json
import os
from cuml import TSNE

print('reading data')
concatted = True
if not concatted:
    files = os.listdir('predict')
    data = np.zeros((len(files), 4096))
    for i, file in enumerate(files):
        fpath = os.path.join('predict', file)
        if os.stat(fpath).st_size == 16384:
            fd = open(fpath, 'rb')
            d = np.frombuffer(fd.read(), np.float32)
            fd.close()
            data[i] = d
        else:
            print("error on file: ", file)
    print(data.shape)
else:
    data = np.fromfile('predict.out', np.float32)
    data = data.reshape((-1, 4096))
    print(data.shape)
    #sys.exit()
#lines = open('opengameart-files/files-list.txt').readlines()
#for line in lines:
#    d = json.loads(line.rstrip())
#    data.append(d["features"])
#data = np.array(data)
#print(data.shape)

print('computing tsne embedding')
#embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1], verbose=True)
tsne = TSNE(n_components = 2)
embedding_array = tsne.fit_transform(data)
outfd = open('embeddings.txt', 'w')
outfd.write(json.dumps(embedding_array.tolist()))
outfd.close()
#print(embedding_array)
