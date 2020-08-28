#!/usr/bin/env python3

import numpy as np
from PIL import Image
#from tensorflow.python.keras.preprocessing import image
import json
import os
import math
import random
import time
import sys
import subprocess
import multiprocessing
from glob import glob
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

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

def split(X, dim):
    xs = sorted(X, key=lambda x: x[0])
    ys = sorted(X, key=lambda x: x[1])
    xrange = xs[-1][0] - xs[0][0]
    yrange = ys[-1][1] - xs[0][1]
    #xmedian = xs[len(xs) // 2]
    #ymedian = ys[len(ys) // 2]
    if xrange > yrange:
        X = xs
    else:
        X = ys
    splitPoint = len(X) // 2
    return [X[0:splitPoint], X[splitPoint:len(X)]]

def save_tsne_grid2(img_collection, X_2d, out_res, out_dim):
    from lapjv import lapjv

    X = [(x, y, i) for i, (x, y) in enumerate(X_2d)]

    #bucketDim2 = 1024
    #bucketDim2 = 4096
    bucketDim2 = 16384
    iout = len(X) * 2
    nearestPower2 = 2 ** math.ceil(math.log2(len(X)))
    for i in range(nearestPower2 - len(X)):
        X.append((100, 100, iout))
    buckets = [X]
    while max([len(b) for b in buckets]) > bucketDim2:
        maxindex = 0
        maxb = 0
        for i, b in enumerate(buckets):
            if len(b) > maxb:
                maxb = len(b)
                maxindex = i
        maxbucket = buckets[maxindex]
        minbuckets = [b for i, b in enumerate(buckets) if i != maxindex]
        minbuckets.extend(split(maxbucket, random.randint(0, 1)))
        buckets = minbuckets
    print("buckets", [len(b) for b in buckets])

    bucket_dim = int(math.sqrt(bucketDim2))
    #bucketsX = out_dim // bucket_dim + 1
    #bucketsX = math.ceil(math.sqrt(len(X))) // bucket_dim + 1
    bucketsX = math.ceil(math.sqrt(len(buckets)))
    out_dim = (bucketsX + 1) * bucket_dim
    toplot = bucket_dim * bucket_dim
    assert toplot == bucketDim2
    out = np.zeros((out_dim * out_res, out_dim * out_res, 4), np.uint8)
    print('len(buckets)', len(buckets), 'bucketDim2', bucketDim2, 'bucket_dim', bucket_dim, 'bucketsX', bucketsX, 'out_dim', out_dim, 'toplot', toplot)

    for bi, bucket in enumerate(buckets):
        print("lapjv bucket", bi, len(buckets))
        X_2d = bucket
        baseX = (bi % (bucketsX - 0)) * bucket_dim * out_res
        baseY = (bi // (bucketsX - 0)) * bucket_dim * out_res
        indices = [i for x, y, i in bucket]
        bucket = [(x, y) for x, y, i in bucket]
        # zero pad
        #for i in range(bucketDim2 - len(bucket)):
        #    bucket.append([(0, 0)])
        npBucket = np.zeros((toplot, 2))
        for i, (x, y) in enumerate(bucket):
            npBucket[i][0] = x
            npBucket[i][1] = y

        grid = np.dstack(np.meshgrid(np.linspace(0, 1, bucket_dim), np.linspace(0, 1, bucket_dim))).reshape(-1, 2)
        cost_matrix = cdist(grid, npBucket, "sqeuclidean").astype(np.float32)
        cost_matrix = cost_matrix * (100000 / cost_matrix.max())
        row_asses, col_asses, _ = lapjv(cost_matrix)
        grid_jv = grid[col_asses]

        for pos, i in zip(grid_jv, range(toplot)):
            if i >= len(indices):
                #print("blank i", i)
                continue
            k = indices[i]

            x, y = pos
            x = int(np.floor(x * (bucket_dim - 1) * out_res)) + baseX
            y = int(np.floor(y * (bucket_dim - 1) * out_res)) + baseY
            # print("xy", baseX, baseY, x, y)
            h_range = x
            w_range = y

            if k == iout:
                #print("blank k", k, x, y)
                continue

            x2, y2 = k % (img_collection.shape[0] // out_res), k // (img_collection.shape[1] // out_res)
            h_range2 = x2 * out_res
            w_range2 = y2 * out_res
            #print("hmmmmm", i, x2, y2, h_range2, w_range2)
            img = img_collection[h_range2:h_range2 + out_res, w_range2:w_range2 + out_res, :]

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
            command = '../hqx/hqx -s ' + str(factor) + ' "' + path + '" ' + outfname
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

def blah():
    if False:
        # u, v = (math.sqrt(x * x + y * y), math.atan2(y, x))
        u, v = x - 0.5, y - 0.5
        theta = math.atan2(v, u)
        r = math.sqrt(u * u + v * v)
        r = 0.5 * r / maxr
        u, v = min(r, 1) * math.cos(theta), min(r, 1) * math.sin(theta)

        # https://stackoverflow.com/questions/13211595/how-can-i-convert-coordinates-on-a-circle-to-coordinates-on-a-square
        x = 0.5 * math.sqrt(2 + u * u - v * v + 2 * u * math.sqrt(2)) - 0.5 * math.sqrt(
            2 + u * u - v * v - 2 * u * math.sqrt(2))
        y = 0.5 * math.sqrt(2 - u * u + v * v + 2 * v * math.sqrt(2)) - 0.5 * math.sqrt(
            2 - u * u + v * v - 2 * v * math.sqrt(2))
        x = x + 0.5
        y = y + 0.5
        if x < 0 or x > 1 or y < 0 or y > 1:
            print("squircle", u, v, x, y)
        x = max(x, 0.0)
        x = min(x, 1.0)
        y = max(y, 0.0)
        y = min(y, 1.0)

def spreadXY(X_2d, threshold, speed):
    'spreads items until distance is greater than threshold'

    import cudf
    from cuml.neighbors import NearestNeighbors

    def kernel(x, y, outx, outy, threshold2):
        for i, (x2, y2) in enumerate(zip(x, y)):
            d = math.sqrt(x2 * x2 + y2 * y2)
            if 0 < d <= threshold2:
                outx[i] = x2 / d
                outy[i] = y2 / d
            else:
                outx[i] = 0
                outy[i] = 0
    print('spreadXY')
    length = len(X_2d)
    X = cudf.DataFrame()
    X['x'] = X_2d[0:length, 0]
    X['y'] = X_2d[0:length, 1]
    k = 8
    scale = 10000
    threshold *= scale
    speed *= scale
    X = X.mul(scale)
    #X = np.copy(X_2d[:length])
    for i in range(20):
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        #print(distances.shape)
        joins = []

        s = X.sum()
        print("iteration", i, "sum dist", s)

        newX = X
        for j in range(k):
            join = indices.drop([x for x in range(k) if x != j]) #.rename(mapper={j: 'x'}, columns=[j])
            join = join.merge(X, how='left', left_on=[j], right_index=True)
            join = join.drop(j)
            v = join.sub(X)
            v = v.apply_rows(kernel, incols=['x', 'y'], outcols=dict(outx=np.float32, outy=np.float32), kwargs=dict(threshold2=threshold))
            v = v.drop(['x', 'y'])
            v = v.rename(columns={'outx': 'x', 'outy': 'y'})
            newX = newX.sub(v.mul(speed))
            #newX = newX.add(1)
            #v = v.query('x * x + y * y <= ' + str(threshold * threshold))
        #print("newX")
        #print(newX)
        X = newX

        s = X.sum()
        print("iteration", i, "sum dist", s)
    X = X.truediv(scale)
    X = np.array(X.as_matrix())
    print(X.shape)
    return X

# approximate hacky solution, scales more (hopefully 200k)
def computeGrid2(imageColors, bigImage, X_2d, out_res, out_dim):
    #xs = sorted(remaining, key=lambda x: x[0])
    #ys = sorted(remaining, key=lambda x: x[1])
    grid = np.zeros((out_dim, out_dim), np.bool)
    out2 = np.zeros((out_dim * out_res, out_dim * out_res, 4), dtype=np.uint8)

    circles = []
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

    for k, (i, x, y) in enumerate(xcircles):
        #for k, i in enumerate(range(to_plot)):
        #x, y = X_2d[i]
        if k % 100 == 0:
            print("k", k, x, y, len(X_2d))

        od2 = out_dim

        x = int(np.floor(x * od2)) % od2
        y = int(np.floor(y * od2)) % od2

        if True:
            done = False
            j = 0
            for _, _, dx, dy in circles:
                if 0 <= x + dx < out_dim and 0 <= y + dy < out_dim and grid[x + dx, y + dy] == False:
                    x = x + dx
                    y = y + dy
                    done = True
                    break
                j += 1
                if j > 100:
                    break
            if not done:
                continue
            grid[x, y] = True

        #if grid[x, y]:
        #    continue
        #grid[x, y] = True

        h_range = int(np.floor(x * out_res))
        w_range = int(np.floor(y * out_res))

        x2, y2 = i % (bigImage.shape[0] // out_res), i // (bigImage.shape[1] // out_res)
        h_range2 = x2 * out_res
        w_range2 = y2 * out_res
        img = np.copy(bigImage[h_range2:h_range2 + out_res, w_range2:w_range2 + out_res, :])
        out2[h_range:h_range + out_res, w_range:w_range + out_res, :] = img
        #color = imageColors[i]
        #out2[h_range:h_range + out_res, w_range:w_range + out_res, 0] = color[0]
        #out2[h_range:h_range + out_res, w_range:w_range + out_res, 1] = color[1]
        #out2[h_range:h_range + out_res, w_range:w_range + out_res, 2] = color[2]
        #out2[h_range:h_range + out_res, w_range:w_range + out_res, 3] = color[3]


    im = Image.fromarray(out2)
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

    embeddings = [[x, y, i] for i, (x, y) in enumerate(embeddings)]

    cut = 100

    xs = sorted(embeddings, key=lambda x: x[0])
    #embeddings = xs
    zeroes = []
    for k in range(cut):
        i = xs[k][2]
        zeroes.append(i)
    ys = sorted(embeddings, key=lambda x: x[1])
    #embeddings = ys
    for k in range(cut):
        i = ys[k][2]
        zeroes.append(i)

    # XXX: made a bug that skipped every 13th image
    #print("warning: skipping every 13th to compensate for bug")
    #embeddings = [x for i, x in enumerate(embeddings) if i % 13 != 0]

    #for i in range(0, len(embeddings), 13):
    #    zeroes.append(i)

    xs = sorted(embeddings, key=lambda x: x[0])
    ys = sorted(embeddings, key=lambda x: x[1])

    minx, maxx = xs[cut][0], xs[-cut][0]
    miny, maxy = ys[cut][1], ys[-cut][1]
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
    for i, (x, y, _) in enumerate(embeddings[0:to_plot]):
        x -= minx
        y -= miny
        top = x / tw
        left = y / th
        top = min(1, max(0, top))
        left = min(1, max(0, left))
        #top += math.sqrt(abs(top - 0.5)) * math.copysign(1, top - 0.5) + 0.5
        #left += math.sqrt(abs(left - 0.5)) * math.copysign(1, left - 0.5) + 0.5
        l[i, 0] = top
        l[i, 1] = left
        #l.append([x, y])

    for i in zeroes:
        l[i][0] = 0
        l[i][1] = 0
        l[-i][0] = 0
        l[-i][1] = 0
    return l

def readImages(image_np_pattern, out_res, to_plot):
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
        start = time.time()
        with multiprocessing.Pool(batchSize) as p:
            for i, fpath in enumerate(img_collection):
                if i % 100 == 0:
                    end = time.time()
                    elapsed = end - start
                    peri = elapsed / max(1, i)
                    remaining = len(img_collection) - i
                    remTime = remaining * peri / 3600
                    print("preparing images", i, peri, 's per image', remTime, ' h left', len(img_collection), fpath)
                batch.append((fpath, out_res, len(batch)))
                if i + 1 >= len(img_collection) or len(batch) >= batchSize:
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

def getImageColors(img_collection):
    colors = np.zeros((len(img_collection), 4), np.uint8)
    def newColor():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        a = 255
        return np.array([r, g, b, a])
    oldBPath = ''
    color = newColor()
    for i, path in enumerate(img_collection):
        #l = len('.np') + len('_00000.png')
        l = len('_00000.png')
        bpath = path[:-l]
        if oldBPath != bpath:
            oldBPath = bpath
            color = newColor()
            #print(bpath)
        colors[i] = color
    return colors

def getFiles(imageDir):
    if not os.path.exists('filelist.txt'):
        paths = []
        for root, dirs, files in os.walk(imageDir):
            for file in files:
                if file.lower().endswith('.np'):
                    file2 = file[:-len('.np')]
                    paths.append(os.path.join(root, file2))
        fd = open('filelist.txt', 'w')
        fd.writelines([x + '\n' for x in paths])
        fd.close()
    else:
        fd = open('filelist.txt', 'r')
        paths = [x.rstrip() for x in fd.readlines()]
        fd.close()
    out_dim = math.ceil(math.sqrt(len(paths)))
    to_plot = np.square(out_dim)

    path = '../blank.png'
    # blank = Image.open(path).resize(size=(out_res, out_res))
    for x in range(len(paths), to_plot):
        paths.append(path)

    #print('WARNING: reenable sort for ^unpacked')
    paths.sort()

    return (out_dim, to_plot, paths)

if __name__ == '__main__':
    out_dir = './'
    out_name = 'gridtsne16.png'
    out_res = 32
    #out_res = 8
    #image_np_pattern = '/mnt/d/opengameart/sprites/*.np'

    imageDir = '/mnt/d/opengameart/unpacked'
    #imageDir = '/mnt/d/opengameart/sprites'
    out_dim, to_plot, pathlist = getFiles(imageDir)
    #out_dim = math.ceil(math.sqrt(len(glob(image_np_pattern))))
    #to_plot = np.square(out_dim)

    #img_collection = readImages(image_np_pattern, out_res, to_plot)[0:to_plot]

    X_2d = readXY()[0:to_plot]

    if 0:
        #X_2d = spreadXY(X_2d, 1 / out_dim, 0.1 / out_dim)
        X_2d = spreadXY(X_2d, 2 / out_dim, 0.25 / out_dim)
        plt.figure(figsize=(40, 40))
        #clr = [i * 255 // len(X_2d) for i in range(len(X_2d))]
        #clr = [i // out_dim for i in range(len(X_2d))]
        pad = lambda x: ('0' * (2 - len(x))) + x
        imageColors = getImageColors(pathlist)
        clr = [('#' + pad(hex(r)[2:]) + pad(hex(g)[2:]) + pad(hex(b)[2:])) for r, g, b, a in imageColors]
        plt.scatter(X_2d[:, 1], 1 - X_2d[:, 0], c=clr)
        plt.savefig('plot.png')
        print('plot done')

    if 1:
        images = prepareImages(pathlist, out_dim, out_res)
        save_tsne_grid2(images, X_2d, out_res, out_dim)
        #computeGrid2(imageColors, images, X_2d, out_res, out_dim)

#out_dim = 32
#save_tsne_grid(images, X_2d, out_res, out_dim)

#print(X_2d.shape)
#save_tsne_grid(img_collection, X_2d, out_res, out_dim)
#computeGrid(img_collection, X_2d, out_res, out_dim)
