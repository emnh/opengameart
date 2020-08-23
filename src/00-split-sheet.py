#!/usr/bin/env python3

from numba import jit
from PIL import Image
from PIL import ImageFilter
import numpy as np
from collections import deque
import random
from pprint import pprint
import sys

path = '/mnt/d/opengameart/files/ProjectUtumno_supplemental_0.png'
path = '/mnt/d/opengameart/unpacked/Atlas_0.zip/Atlas_0/terrain_atlas.png'
path = '/mnt/d/opengameart/files/Grasstop.png'


img = Image.open(path).convert('RGBA')
ar = np.array(img)

r = 0
g = 1
b = 2
a = 3

@jit(nopython=True)
def isTransparent(x):
    return x == 0

@jit(nopython=True)
def inRange(ar, x, y):
    return 0 <= x < ar.shape[0] and 0 <= y < ar.shape[1]

@jit(nopython=True)
def bfs(queue, ar, out, seen, nbs, sx, sy):
    #q = deque([(sx, sy)])
    queue[0] = [sx, sy]
    qlen = 1
    qi = 0

    cr = random.randint(0, 255)
    cg = random.randint(0, 255)
    cb = random.randint(0, 255)
    ca = 255
    color = [cr, cg, cb, ca]

    i = 0
    minx = sx
    miny = sy
    maxx = sx
    maxy = sy

    while 0 <= qi < qlen:
        x, y = queue[qi]
        qi += 1

        if seen[x, y] == 1:
            continue
        i += 1
        seen[x, y] = 1
        out[x, y] = color #[255, 0, 0, 255]

        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x)
        maxy = max(maxy, y)

        for dx, dy in nbs:
            nx = x + dx
            ny = y + dy
            if inRange(ar, nx, ny) and not isTransparent(ar[nx, ny, a]) and not seen[nx, ny] == 1:
                #c1 = out[x, y].astype(np.float32)
                #c2 = out[nx, ny].astype(np.float32)
                #d = np.linalg.norm(c2 - c1)
                #if d <= 128:
                queue[qlen] = [nx, ny]
                qlen += 1
                if qlen >= queue.shape[0]:
                    print("queue not big enough")
                    return
    return [minx, miny, maxx, maxy]
    #print(i)

@jit(nopython=True)
def splitSheet(ar):
    seen = np.zeros((ar.shape[0], ar.shape[1]), dtype=np.int8)
    out = np.copy(ar)
    nbs = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2)
    ]
    queue = np.zeros((ar.shape[0] * ar.shape[1] * len(nbs), 2), dtype=np.int16)
    d = {}
    for x in range(ar.shape[0]):
        for y in range(ar.shape[1]):
            if not isTransparent(ar[x, y, a]):
                bounds = bfs(queue, ar, out, seen, nbs, x, y)
                w, h = (bounds[2] - bounds[0], bounds[3] - bounds[1])
                for p in range(2, 8):
                    n = 2 ** p
                    # if it fits within tile
                    if n / 2 < w < n and n / 2 < h < n and bounds[0] // n == bounds[2] // n and bounds[3] // n == bounds[1] // n:
                        if n in d:
                            d[n] += 1
                        else:
                            d[n] = 1
                #if (w, h) in d:
                #    d[(w, h)] += 1
                #else:
                #    d[(w, h)] = 1
            else:
                ar[x, y] = [0, 0, 0, 0]
            #for nb in nbs:
    print("splitSheet")
    for k, v in sorted(d.items(), key=lambda x: x[1]):
        print(k, v)
    return out

def detectSize(ar):
    r = range(4, 65)
    l = [0 for x in r]
    for x in range(1, ar.shape[0]):
        #s = np.linalg.norm(ar[x] - ar[x - 1]) / ar.shape[1]
        s = 0
        for y in range(ar.shape[1]):
            if not(isTransparent(ar[x, y, 3]) or isTransparent(ar[x - 1, y, 3])):
                s += np.linalg.norm(ar[x, y] / 255 * ar[x, y, 3] / 255 - ar[x - 1, y] / 255 * ar[x - 1, y, 3] / 255) / (4 ** 0.5)
        s /= ar.shape[1]
        #print(x, s)
        for i, p in enumerate(r):
            n = p
            if x % n == 0:
                l[i] += s * n
    for i, p in enumerate(r):
        n = p
        print(n, l[i])

def detectSize2(ar):
    arf = np.copy(ar).astype(np.float32) / 255.0
    xcumulative = np.copy(arf)
    for x in range(1, ar.shape[0]):
        xcumulative[x] += xcumulative[x - 1]
    ycumulative = np.copy(xcumulative)
    for y in range(1, ar.shape[1]):
        ycumulative[:, y] += ycumulative[:, y - 1]
    #print("xcumulative")
    #print(xcumulative)
    #print("ycumulative")
    #print(ycumulative)

    #average = np.copy(ycumulative) * 0
    tries = [x for x in [4, 8, 16, 32, 48, 64, 96, 128] if x <= ar.shape[0]]
    average = np.zeros((len(tries), ar.shape[0], ar.shape[1], ar.shape[2]), np.float32)
    for x in range(1, ar.shape[0]):
        for y in range(1, ar.shape[1]):
            for i, n in enumerate(tries):
                if x % n == n - 1 and y % n == n - 1:
                    sx = (x + 1) // n - 1
                    sy = (y + 1) // n - 1
                    sxn = sx * n
                    syn = sy * n
                    average[i, sxn:(sxn + n), syn:(syn + n)] = ycumulative[x, y] - ycumulative[x - (n - 1), y - (n - 1)]
                    average[i, sxn:(sxn + n), syn:(syn + n)] /= n * n
                    average[i, sxn:(sxn + n), syn:(syn + n)] = \
                        np.abs(average[i, sxn:(sxn + n), syn:(syn + n)] - arf[sxn:(sxn + n), syn:(syn + n)])
            #for i, n in enumerate(tries):
            #    if x % n == 0:
            #        squares[i, x // n]

    #print("average dist")
    #print(average)

    xcumulative2 = np.copy(average)
    for x in range(1, average.shape[1]):
        xcumulative2[:, x] += xcumulative2[:, x - 1]
    ycumulative2 = np.copy(xcumulative2)
    for y in range(1, average.shape[2]):
        ycumulative2[:, :, y] += ycumulative2[:, :, y - 1]
    #print("xcumulative2")
    #print(xcumulative2)
    #print("ycumulative2")
    #print(ycumulative2)

    for i, n in enumerate(tries):
        tiles = ar.shape[0] // n * ar.shape[1] // n
        print(str(n) + ":", ycumulative2[i, -1, -1] / tiles, np.linalg.norm(ycumulative2[i, -1, -1] / tiles))

    #dist = np.abs(average - arf)
    #xcumulative = np.copy(dist)
    #for x in range(1, ar.shape[0]):
    #    xcumulative[x] += xcumulative[x - 1]
    #ycumulative = np.copy(xcumulative)
    #for y in range(1, ar.shape[1]):
    #    ycumulative[:, y] += ycumulative[:, y - 1]


#ar = ar[0:32, 0:32, :]
detectSize2(ar)
#out = splitSheet(ar)
#Image.fromarray(out).save('out.png')
#img.filter(ImageFilter.FIND_EDGES).save('edge.png')