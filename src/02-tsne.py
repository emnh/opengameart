#!/usr/bin/env python3
import sys
#sys.path += ['/mnt/d/dev/opengameart/bhtsne']
#import bhtsne
import numpy as np
import json
import os
from cuml import TSNE
from cuml import PCA
from cuml.manifold.umap import UMAP as cuUMAP
import random

print('reading data')
concatted = True
if not concatted:
    files = os.listdir('predict')
    files.sort()
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

#u = umap.UMAP()
#u.fit_transform(data)

#print(data.shape)

#data = data[0:10000]
indices = [x for x in range(len(data))]
#random.shuffle(indices)
data2 = data
#data2 = np.copy(data)
#for i1, i2 in zip(range(len(data)), indices):
#    data2[i1] = data[i2]

pcaFile = 'pca.np'
perComp = 100000
comps = len(data) // perComp + 1
pcaComps = 20
if 0:
    if not os.path.exists(pcaFile):
        data3 = np.zeros((len(data), pcaComps), np.float32)
        pca = PCA(n_components=pcaComps)
        for i in range(comps):
            data2 = data[i * perComp: (i + 1) * perComp]
            data2 = pca.fit_transform(data2)
            data3[i * perComp: (i + 1) * perComp, :] = data2
        data2 = data3
        del pca
        fd = open(pcaFile, 'wb')
        fd.write(data2.flatten().tobytes())
        fd.close()
    else:
        l = len(data)
        del data
        del data2
        data2 = np.fromfile(pcaFile, np.float32).reshape((l, pcaComps))

if 0:
    if not os.path.exists(pcaFile):
        pca = PCA(n_components=pcaComps)
        del data
        data2 = data2[0:perComp]
        print(data2.shape)
        #data2 = np.swapaxes(data2, 1, 0)
        data2 = pca.fit_transform(data2)
        fd = open(pcaFile, 'wb')
        fd.write(data2.flatten().tobytes())
        fd.close()
    else:
        l = len(data)
        del data
        del data2
        data2 = np.fromfile(pcaFile, np.float32).reshape((perComp, pcaComps))

#tsne = TSNE(n_components=2)
#embedding_array = bhtsne.run_bh_tsne(data2, initial_dims=data2.shape[1], verbose=True).tolist()
#data2 = data2[0:230000]
tsne = TSNE(n_components=2, perplexity=50, n_iter=5000, angle=0.8, learning_rate=10, n_neighbors=2)
embedding_array = tsne.fit_transform(data2).tolist()
#embedding_array = data2.tolist()

#umap = cuUMAP(n_neighbors = 10, local_connectivity=10, min_dist=1)
#embedding_array = umap.fit_transform(data2).tolist()
outdata = [x for x in embedding_array]
#for i1, i2 in enumerate(indices):
#    embedding_array[i2] = outdata[i1]

outfd = open('embeddings.txt', 'w')
outfd.write(json.dumps(embedding_array))
outfd.close()
#print(embedding_array)
