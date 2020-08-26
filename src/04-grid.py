#!/usr/bin/env python3

import numpy as np
from PIL import Image
#from tensorflow.python.keras.preprocessing import image
import json
import os
import math
import random
import sys
import subprocess
import multiprocessing
from glob import glob
from scipy.spatial.distance import cdist
#from cuml.dask.neighbors import NearestNeighbors
#import cudf
#from cuml.neighbors import NearestNeighbors

Image.MAX_IMAGE_PIXELS = None

# from https://github.com/prabodhhere/tsne-grid/blob/master/tsne_grid.py
# scales to around 10k in 10 min
# TODO: needs updating
def save_tsne_grid(img_collection, X_2d, out_res, out_dim):
    from lapjv import lapjv
    toplot = out_dim * out_dim
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim), np.linspace(0, 1, out_dim))).reshape(-1, 2)
    cost_matrix = cdist(grid, X_2d[:toplot], "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    out = np.zeros((out_dim*out_res, out_dim*out_res, 4), np.uint8)

    for pos, i in zip(grid_jv, range(toplot)):
        x2, y2 = i % (img_collection.shape[0] // out_res), i // (img_collection.shape[1] // out_res)
        h_range2 = x2 * out_res
        w_range2 = y2 * out_res
        #print("hmmmmm", i, x2, y2, h_range2, w_range2)
        img = img_collection[h_range2:h_range2 + out_res, w_range2:w_range2 + out_res, :]

        h_range = int(np.floor(pos[0]* (out_dim - 1) * out_res))
        w_range = int(np.floor(pos[1]* (out_dim - 1) * out_res))
        out[h_range:h_range + out_res, w_range:w_range + out_res] = img

    im = Image.fromarray(out)
    im.save(out_dir + out_name, quality=100)

def readImage(args):
    path, out_res, processIndex = args
    f = lambda x: np.array(Image.open(x).resize(size=(out_res, out_res)).convert('RGBA'))
    try:
        img = Image.open(path)
        if img.width < out_res or img.height < out_res:
            #print("scaling", img)
            factor = max(2, out_res // min(img.width, img.height))
            outfname = 'out' + str(processIndex) + '.png'
            command = '../hqx/hqx -s ' + str(factor) + ' ' + path + ' ' + outfname
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            process.wait()
            img = f(outfname)
            os.remove(outfname)
        else:
            img = f(path)
        return img
    except:
        print("Error loading image: " + path)
    return f('../blank.png')


# approximate hacky solution, scales more (hopefully 200k)
def computeGrid2(img_collection, X_2d, out_res, out_dim):
    #xs = sorted(remaining, key=lambda x: x[0])
    #ys = sorted(remaining, key=lambda x: x[1])
    grid = np.zeros((out_dim, out_dim), np.bool)
    out = np.zeros((out_dim * out_res, out_dim * out_res, 4), dtype=np.uint8)

    circles = []
    #circlesSeen = {}
    for dx in range(out_dim):
        for dy in range(out_dim):
            cx = dx - out_dim // 2
            cy = dy - out_dim // 2
            theta = math.atan2(cy, cx)
            r = math.sqrt(cx * cx + cy * cy)
            circles.append((r, theta, cx, cy))
    circles.sort()

    xnormal = []
    xcircles = []
    maxr = 0
    for i, (dx, dy) in enumerate(X_2d):
        cx = dx - 0.5
        cy = dy - 0.5
        theta = math.atan2(cy, cx)
        r = math.sqrt(cx * cx + cy * cy)
        maxr = max(r, maxr)
        xcircles.append((r, theta, i, dx, dy))
        xnormal.append((i, dx, dy))
    xcircles.sort()
    xcircles = [(i, x, y) for _, _, i, x, y in xcircles]

    #random.shuffle(xnormal)
    #xnormal.sort(key=lambda x: (x[1], x[2]))
    #for r in range(out_dim):
    #    done = False
    #    for theta in range(360):
    #        x = int(r * math.cos(theta))
    #        y = int(r * math.sin(theta))
    #        if not (x, y) in circlesSeen:
    #            circlesSeen[(x, y)] = True
    #            circles.append((x, y))

    #squircle = []
    #for k, (i, x, y) in enumerate(xcircles):

    #def f(args):
    #    k, i, x, y = args

    for k, (i, x, y) in enumerate(xnormal):
        if k % 100 == 0:
            print("k", k, x, y, len(X_2d))
        #x, y = X_2d[i]
        #if k % 100 == 0:
        #    print("k", k, x, y, len(X_2d))

        #if random.random() > 0.5:
        #    continue

        if False:
            #u, v = (math.sqrt(x * x + y * y), math.atan2(y, x))
            u, v = x - 0.5, y - 0.5
            theta = math.atan2(v, u)
            r = math.sqrt(u * u + v * v)
            r = 0.5 * r / maxr
            u, v = min(r, 1) * math.cos(theta), min(r, 1) * math.sin(theta)

            # https://stackoverflow.com/questions/13211595/how-can-i-convert-coordinates-on-a-circle-to-coordinates-on-a-square
            x = 0.5 * math.sqrt(2 + u * u - v * v + 2 * u * math.sqrt(2)) - 0.5 * math.sqrt(2 + u * u - v * v - 2 * u * math.sqrt(2))
            y = 0.5 * math.sqrt(2 - u * u + v * v + 2 * v * math.sqrt(2)) - 0.5 * math.sqrt(2 - u * u + v * v - 2 * v * math.sqrt(2))
            x = x + 0.5
            y = y + 0.5
            if x < 0 or x > 1 or y < 0 or y > 1:
                print("squircle", u, v, x, y)
            x = max(x, 0.0)
            x = min(x, 1.0)
            y = max(y, 0.0)
            y = min(y, 1.0)

        #x = int(np.floor(x * out_dim)) % out_dim  # / out_dim
        #y = int(np.floor(y * out_dim)) % out_dim  # / out_dim

        od2 = out_dim - 1

        x = int(np.floor(x * od2))# / out_dim
        y = int(np.floor(y * od2))# / out_dim
        print("xy", x, y)
        if False:
            done = False
            for _, _, dx, dy in circles:
                if 0 <= x + dx < out_dim and 0 <= y + dy < out_dim and grid[x + dx, y + dy] == False:
                    x = x + dx
                    y = y + dy
                    done = True
                    break
            if not done:
                continue
            grid[x, y] = True

        # XXXXXXXXXXXXXXXXXXXXXXXX
        #x, y = i % (img_collection.shape[0] // out_res), i // (img_collection.shape[1] // out_res)

        x /= od2
        y /= od2

        h_range = int(np.floor(x * od2 * out_res))
        w_range = int(np.floor(y * od2 * out_res))

        #x2, y2 = i % out_dim, i // out_dim
        x2, y2 = i % (img_collection.shape[0] // out_res), i // (img_collection.shape[1] // out_res)
        #print(out_dim, img_collection.shape[0], img_collection.shape[1])
        h_range2 = x2 * out_res
        w_range2 = y2 * out_res
        img = img_collection[h_range2:h_range2 + out_res, w_range2:w_range2 + out_res, :]
        out[h_range:h_range + out_res, w_range:w_range + out_res, :] = img
    im = Image.fromarray(out)
    im.save(out_dir + out_name, quality=100)
    pass

# approximate hacky solution, scales more (hopefully 200k)
def computeGrid(img_collection, X_2d, out_res, out_dim):
    #grid = np.zeros(())
    out = np.ones((out_dim*out_res, out_dim*out_res, 4), dtype=np.uint8)
    nn = NearestNeighbors(n_neighbors=1)
    d = {}
    # TODO: assumes unique
    #for i, xy in enumerate(X_2d):
    #    d[tuple(xy)] = i

    def build(remaining):
        xs = sorted(remaining, key=lambda x: x[0])
        ys = sorted(remaining, key=lambda x: x[1])
        X = np.zeros((out_dim * out_dim, 2), np.float32)
        for x in range(out_dim):
            for y in range(out_dim):
                X[y * out_dim + x] = np.array([xs[int(x * len(xs) / out_dim)][0], ys[int(y * len(ys) / out_dim)][1]])
        #X = np.array(X)
        #print(X.shape)
        nn.fit(remaining)
        X_cudf = cudf.DataFrame(X)
        distances, indices = nn.kneighbors(X_cudf)
        return indices
    #remaining = X_2d
    indices = build(X_2d)

    seen = {}
    #for i, x in enumerate(xs):
        #for j, y in enumerate(ys):
    for x in range(out_dim):
        for y in range(out_dim):
            done = False
            e = 0
            while not done:
                #X = (xs[int(x * len(xs) / out_dim)], ys[int(y * len(ys) / out_dim)])
                #X_cudf = cudf.DataFrame(X)
                #distances, indices = nn.kneighbors(X_cudf)
                p = nearest = indices[(y * out_dim + x + e) % len(indices)]
                if p in seen:
                    e += 1
                    #print("rebuild")
                    #remaining = [X_2d[x] for i, x in enumerate(X_2d) if not i in seen]
                    #indices = build(remaining)
                else:
                    seen[p] = True
                    done = True
            #print("nearest", nearest)
            #p = d[tuple(nearest)]
            #grid[i][j] = img_collection[p]
            pos = (x, y)
            print("xyp", x, y, p)
            img = img_collection[p]
            h_range = x * out_res
            w_range = y * out_res
            #print("range", h_range, h_range + out_res, w_range, w_range + out_res)
            out[h_range:h_range + out_res, w_range:w_range + out_res, :] = readImage(img, out_res)

            #break
        #break

    #im = image.array_to_img(out)
    im = Image.fromarray(out)
    im.save(out_dir + out_name, quality=100)

def readXY():
    embeddings = json.loads(open('embeddings.txt').read())
    minx = 1.0e20
    maxx = 0.0
    miny = 1.0e20
    maxy = 0.0
    l = []

    xs = sorted(embeddings, key=lambda x: x[0])
    ys = sorted(embeddings, key=lambda x: x[1])
    minx, maxx = xs[3][0], xs[-3][0]
    miny, maxy = ys[3][1], ys[-3][1]
    #for x, y in embeddings:
    #    if x < minx or y < miny:
    #        print("xy", x, y)
    #    minx = min(minx, x)
    #    maxx = max(maxx, x)
    #    miny = min(miny, y)
    #    maxy = max(maxy, y)

    print("minmax", minx, maxx, miny, maxy)

    tw = (maxx - minx)
    th = (maxy - miny)

    l = np.zeros((to_plot, 2))
    for i, (x, y) in enumerate(embeddings[0:to_plot]):
        x -= minx
        y -= miny
        top = x / tw
        left = y / th
        top = min(1, max(0, top))
        left = min(1, max(0, left))
        l[i, 0] = top
        l[i, 1] = left
        #l.append([x, y])

    return l

def readImages(image_np_pattern, out_res):
    #lines = open('opengameart-files/files-list.txt').readlines()
    files = glob(image_np_pattern)
    files.sort()
    #outfd = open('tsne.html', 'w')
    imgs = []
    for i, file in enumerate(files[0:to_plot]):
        #d = json.loads(line.rstrip())
        #spath = d["path"]
        #path = spath.replace("/opengameart/files/", "/mnt/d/opengameart/files64/")
        path = os.path.join('/mnt/d/opengameart/sprites', file[:-len('.np')])
        imgs.append(path)
        #if i % 100 == 0:
        #    print("reading image", i, to_plot)
    path = '../blank.png'
    #blank = Image.open(path).resize(size=(out_res, out_res))
    for x in range(len(files), to_plot):
        imgs.append(path)
    #for i, img in enumerate(imgs):
    #    imgs[i] = img.convert('RGBA')
    print(len(imgs))
    return imgs

def prepareImages(img_collection, out_dim, out_res):
    dumpfile = 'imagedump' + str(out_res) + 'x' + str(out_res) + '.png'
    if not os.path.exists(dumpfile):
        out = np.zeros((out_dim * out_res, out_dim * out_res, 4), dtype=np.uint8)
        k = 0
        batch = []
        batchSize = 12
        for i, fpath in enumerate(img_collection):
            if i % 100 == 0:
                print("preparing images", i, len(img_collection))
            if i + 1 < len(img_collection) and len(batch) < batchSize:
                batch.append((fpath, out_res, len(batch)))
            else:
                with multiprocessing.Pool(12) as p:
                    images = p.map(readImage, batch)
                    for img in images:
                        #if np.equal(img[::3], 0).all():
                            # TODO: do something about empty images
                        #    print("empty", i, fpath)
                        x, y = k % out_dim, k // out_dim
                        h_range = x * out_res
                        w_range = y * out_res
                        out[h_range:h_range + out_res, w_range:w_range + out_res, :] = img
                        k += 1
                batch = []
        im = Image.fromarray(out)
        im.save(dumpfile, quality=100)
    else:
        out = np.array(Image.open(dumpfile).convert('RGBA'), np.uint8)
    # out = np.frombuffer(dumpfile).reshape((out_dim * out_res, out_dim * out_res, 4), dtype=np.uint8)
    return out

out_dir = './'
out_name = 'gridtsne14.png'
out_res = 32
image_np_pattern = '/mnt/d/opengameart/sprites/*.np'
out_dim = math.ceil(math.sqrt(len(glob(image_np_pattern))))
to_plot = np.square(out_dim)
img_collection = readImages(image_np_pattern, out_res)[0:to_plot]
images = prepareImages(img_collection, out_dim, out_res)
X_2d = readXY()[0:to_plot]
computeGrid2(images, X_2d, out_res, out_dim)
#out_dim = 32
#save_tsne_grid(images, X_2d, out_res, out_dim)

#print(X_2d.shape)
#save_tsne_grid(img_collection, X_2d, out_res, out_dim)
#computeGrid(img_collection, X_2d, out_res, out_dim)
