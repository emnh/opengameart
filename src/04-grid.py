#!/usr/bin/env python3

import numpy as np
from lapjv import lapjv
from PIL import Image
from tensorflow.python.keras.preprocessing import image
import json
import os
import math
from scipy.spatial.distance import cdist

# from https://github.com/prabodhhere/tsne-grid/blob/master/tsne_grid.py
def save_tsne_grid(img_collection, X_2d, out_res, out_dim):
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim), np.linspace(0, 1, out_dim))).reshape(-1, 2)
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    out = np.ones((out_dim*out_res, out_dim*out_res, 4))

    for pos, img in zip(grid_jv, img_collection[0:to_plot]):
        h_range = int(np.floor(pos[0]* (out_dim - 1) * out_res))
        w_range = int(np.floor(pos[1]* (out_dim - 1) * out_res))
        out[h_range:h_range + out_res, w_range:w_range + out_res]  = image.img_to_array(img) #[:,:,0:3]

    im = image.array_to_img(out)
    im.save(out_dir + out_name, quality=100)

def readXY():
    embeddings = json.loads(open('embeddings.txt').read())
    minx = 0.0
    maxx = 0.0
    miny = 0.0
    maxy = 0.0
    l = []
    for embedding in embeddings:
        x, y = embedding
        minx = min(minx, x)
        maxx = max(maxx, x)
        miny = min(miny, y)
        maxy = max(maxy, y)

    tw = (maxx - minx)
    th = (maxy - miny)

    l = np.zeros((to_plot, 2))
    for i, embedding in enumerate(embeddings[0:to_plot]):
        x, y = embedding
        x -= minx
        y -= miny
        top = x / tw
        left = y / th
        l[i, 0] = x
        l[i, 1] = y
        #l.append([x, y])

    return l

def readImages(out_res):
    #lines = open('opengameart-files/files-list.txt').readlines()
    files = os.listdir('predict')
    files.sort()
    #outfd = open('tsne.html', 'w')
    imgs = []
    for file in files[0:to_plot]:
        #d = json.loads(line.rstrip())
        #spath = d["path"]
        #path = spath.replace("/opengameart/files/", "/mnt/d/opengameart/files64/")
        path = os.path.join('/mnt/d/opengameart/sprites', file[:-len('.np')])
        try:
            imgs.append(Image.open(path).resize(size=(out_res, out_res)))
            error = False
        except:
            print("Error loading image: " + path)
            error = True
        if error:
            try:
                path = spath.replace("/opengameart/files/", "/mnt/d/opengameart/files/")
                imgs.append(Image.open(path).resize(size=(out_res, out_res)))
                error = False
            except:
                print("Error loading image: " + path)
        if error:
            imgs.append(Image.open(path).resize(size=(out_res, out_res)))
    for x in range(len(files), to_plot):
        path = 'blank.png'
        imgs.append(Image.open(path).resize(size=(out_res, out_res)))
    for i, img in enumerate(imgs):
        imgs[i] = img.convert('RGBA')
    print(len(imgs))
    return imgs

out_dir = './'
out_name = 'gridtsne.png'
out_res = 64
out_dim = math.ceil(math.sqrt(len(os.listdir('predict'))))
to_plot = np.square(out_dim)
img_collection = readImages(out_res)[0:to_plot]
X_2d = readXY()[0:to_plot]
#print(X_2d.shape)
save_tsne_grid(img_collection, X_2d, out_res, out_dim)
