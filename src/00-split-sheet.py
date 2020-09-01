#!/usr/bin/env python3

from numba import jit
from PIL import Image
from PIL import ImageFilter
import numpy as np
from collections import deque
import random
from pprint import pprint
import sys
import math
#import cv2 as cv
import os
from pathlib import Path

paths = [
    ('/mnt/d/opengameart/files/ProjectUtumno_supplemental_0.png', (32, 32)),
    ('/mnt/d/opengameart/files/grass-tiles-2-small.png', (32, 32)),
    ('/mnt/d/opengameart/files/StoneBlocks_byVellidragon.png', (32, 32)),
    ('/mnt/d/opengameart/files/terrain2_6.png', (64, 64)),
    ('/mnt/d/opengameart/unpacked/Atlas_0.zip/Atlas_0/terrain_atlas.png', (32, 32)),
    ('/mnt/d/opengameart/files/Grasstop.png', (16, 16)),
    ('/mnt/d/opengameart/files/%23011-Nekogare%20hey.png', (1, 1)),
    ('/mnt/d/opengameart/files/arcadArne_sheet_org_desat.png', (16, 16)),
    ('/mnt/d/opengameart/files/Green%20Iron.png', (16, 16))
]
path = paths[8][0]


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
    boundsList = []
    for x in range(ar.shape[0]):
        for y in range(ar.shape[1]):
            if not isTransparent(ar[x, y, a]):
                bounds = bfs(queue, ar, out, seen, nbs, x, y)
                boundsList.append(bounds)
                w, h = (bounds[2] - bounds[0], bounds[3] - bounds[1])
                #for p in range(2, 8):
                #    n = 2 ** p
                #    # if it fits within tile
                #    if n / 2 < w < n and n / 2 < h < n and bounds[0] // n == bounds[2] // n and bounds[3] // n == bounds[1] // n:
                #        if n in d:
                #            d[n] += 1
                #        else:
                #            d[n] = 1
                if w >= 4 and h >= 4:
                    if (w, h) in d:
                        d[(w, h)] += 1
                    else:
                        d[(w, h)] = 1
            else:
                ar[x, y] = [0, 0, 0, 0]
            #for nb in nbs:
    #print("splitSheet")
    ds = sorted(d.items(), key=lambda x: x[1])
    #for k, v in ds:
    #    print(k, v)
    return (out, ds, boundsList)
    #return out

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
    nbTileDist = np.zeros((len(tries), ar.shape[2]), np.float32)
    for x in range(1, ar.shape[0]):
        for y in range(1, ar.shape[1]):
            for i, n in enumerate(tries):
                if x % n == n - 1 and y % n == n - 1:
                    sx = (x + 1) // n - 1
                    sy = (y + 1) // n - 1
                    sxn = sx * n
                    syn = sy * n
                    average[i, sxn:(sxn + n), syn:(syn + n)] = ycumulative[x, y] - ycumulative[sxn, syn]
                    average[i, sxn:(sxn + n), syn:(syn + n)] /= n * n
                    average[i, sxn:(sxn + n), syn:(syn + n)] = \
                        np.abs(average[i, sxn:(sxn + n), syn:(syn + n)] - arf[sxn:(sxn + n), syn:(syn + n)])
                    if sx > 0 and sy > 0:
                        nba = ycumulative[x - n, y - n] - ycumulative[sxn - n, syn - n]
                        nbb = ycumulative[x, y] - ycumulative[sxn, syn]
                        nbTileDist[i] += np.abs((nba - nbb) / n * n)
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
        tileDiffs = (ar.shape[0] // n - 1) * (ar.shape[1] // n - 1)
        print(str(n) + ":", ycumulative2[i, -1, -1] / tiles, np.linalg.norm(ycumulative2[i, -1, -1] / tiles))
        if tileDiffs > 0:
            print(str(n) + "D:", np.linalg.norm(ycumulative2[i, -1, -1] / tiles) / (np.linalg.norm(nbTileDist[i]) / tileDiffs))
    print("")
    for i, n in enumerate(tries):
        tileDiffs = (ar.shape[0] // n - 1) * (ar.shape[1] // n - 1)
        if tileDiffs > 0:
            print(str(n) + ":", nbTileDist[i], np.linalg.norm(nbTileDist[i]) / tileDiffs)
    #dist = np.abs(average - arf)
    #xcumulative = np.copy(dist)
    #for x in range(1, ar.shape[0]):
    #    xcumulative[x] += xcumulative[x - 1]
    #ycumulative = np.copy(xcumulative)
    #for y in range(1, ar.shape[1]):
    #    ycumulative[:, y] += ycumulative[:, y - 1]

def detectSize3X(ar, fx):
    arf = np.copy(ar).astype(np.float32) / 255.0
    xs = np.zeros((ar.shape[0] + 1,))
    sq = math.sqrt
    sq = lambda x: x
    sm = np.sum
    for x in range(1, ar.shape[0]):
        #s1 = sm(np.abs(arf[x] * np.repeat(arf[x, :, 3][:, None], 4, axis=1) - arf[x - 1] * np.repeat(arf[x - 1, :, 3][:, None], 4, axis=1)))
        s1 = fx(arf, x)
        #print("xs1", x, s1)
        s1 = np.linalg.norm(s1)
        #print(str(x) + "xC:", s1)
        s2 = sm(np.repeat((1.0 - arf[x, :, 3] * arf[x - 1, :, 3])[:, None], 4, axis=1))
        s2 = np.linalg.norm(s2)
        #print(str(x) + "xA:", s2)
        #print(str(x) + "x:", s1) # + s2)
        xs[x] = s1 #+ s2
    print("xs1 avg", np.sum(xs) / xs.shape[0])

    normdiv = np.amax(xs) - np.amin(xs)
    if normdiv == 0.0:
        normdiv = 1.0
    xs = (xs - np.amin(xs)) / normdiv
    xs[ar.shape[0]] = 1
    xss = sorted(xs)
    #xdivider = xss[int(len(xss) * 0.5)]
    xstotal = np.sum(xs)
    res = []
    for x in range(4, ar.shape[0]+1):
        if ar.shape[0] % x != 0:
            continue
        vec = xs[0:ar.shape[0]+2:x]
        xdividx = max(0, len(xss) - vec.shape[0] - 1)
        #xdividx = max(0, len(xss) * 1 // 4 )
        xdivider = xss[xdividx]
        #s5 = np.sum(vec > xdivider) / vec.shape[0]
        #s6 = np.sum(xs <= xdivider) / max(1, xs.shape[0] - vec.shape[0] - 1)
        s5 = np.sum(vec) / vec.shape[0]
        s6 = s5
        #print(str(x), xs[x], xs[x] > xdivider)
        #print(str(x) + "x:", s4 - (xstotal - s5) / sq(max(1, xs.shape[0] - vec.shape[0] - 1)))
        #print(str(x) + "x:", s5, s5 - s6, s5, s6, xdividx, xdivider)
        res.append((x, s5))
    res.sort(key=lambda x: x[1])
    return res

def detectSize3Y(ar, fy):
    arf = np.copy(ar).astype(np.float32) / 255.0
    ys = np.zeros((ar.shape[1] + 1,))
    sq = math.sqrt
    sq = lambda x: x
    sm = np.sum
    for y in range(1, ar.shape[1]):
        #s1 = sm(np.abs(arf[:, y] * np.repeat(arf[:, y, 3][:, None], 4, axis=1) - arf[:, y - 1] * np.repeat(arf[:, y - 1, 3][:, None], 4, axis=1)))
        s1 = fy(arf, y)
        #print("ys1", y, s1)
        s1 = np.linalg.norm(s1)
        #print(str(y) + "yC:", s1)
        s2 = sm(np.repeat((1.0 - arf[:, y, 3] * arf[:, y - 1, 3])[:, None], 4, axis=1))
        s2 = np.linalg.norm(s2)
        #print(str(x) + "yA:", s2)
        #print(str(y) + "y:", s1) # + s2)
        ys[y] = s1 #+ s2
    print("ys1 avg", np.sum(ys) / ys.shape[0])

    normdiv = np.amax(ys) - np.amin(ys)
    if normdiv == 0.0:
        normdiv = 1.0
    ys = (ys - np.amin(ys)) / normdiv
    ys[ar.shape[1]] = 1
    yss = sorted(ys)
    #ydivider = yss[int(len(yss) * 0.5)]
    ystotal = np.sum(ys)
    res = []
    for y in range(4, ar.shape[1]+1):
        if ar.shape[1] % y != 0:
            continue
        vec = ys[0:ar.shape[1]+2:y]
        ydividx = max(0, len(yss) - vec.shape[0] - 1)
        #ydividx = max(0, len(yss) * 1 // 4)
        ydivider = yss[ydividx]
        #s5 = np.sum(vec > ydivider) / vec.shape[0]
        #s6 = np.sum(ys <= ydivider) / max(1, ys.shape[0] - vec.shape[0] - 1)
        s5 = np.sum(vec) / vec.shape[0]
        s6 = s5
        #print(str(y) + "y:", s4 - (ystotal - s5) / sq(max(1, ys.shape[0] - vec.shape[0] - 1)))
        #print(str(y), ys[y], ys[y] > ydivider)
        #print(str(y) + "y:", s5, s5 - s6, s5, s6, ydividx, ydivider)
        res.append((y, s5))
    res.sort(key=lambda x: x[1])
    return res

def getHorizontalAndVertical(src):
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                              cv.THRESH_BINARY, 15, -2)
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 32
    
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    
    # Apply morphology operations
    horizontal = cv.erode(horizontal, horizontalStructure)
    #horizontal = cv.erode(horizontal, horizontalStructure)
    #horizontal = cv.dilate(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 32
    
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    
    # Apply morphology operations
    vertical = cv.erode(vertical, verticalStructure)
    #vertical = cv.erode(vertical, verticalStructure)
    #vertical = cv.dilate(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)

    #cv.imwrite('horizontal.png', horizontal)
    #cv.imwrite('vertical.png', vertical)
    #cv.imwrite('horizontalAndVertical.png', horizontal + vertical)

    return (horizontal, vertical)


def getHV(img):
    #img = img.convert('RBGA')
    ar = np.array(img)
    pil_image = img.convert('RGB')
    if ar.shape[2] >= 4:
        pil_image *= np.repeat(ar[:, :, 3][:, :, None], 3, axis=2)
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    # src = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)
    src = open_cv_image

    # x = 1
    # y = 0
    # ksize = 13
    # dst1 = cv.Sobel(src, cv.CV_64F, x, y, ksize=ksize)
    # x = 0
    # y = 1
    # dst2 = cv.Sobel(src, cv.CV_64F, x, y, ksize=ksize)
    # dst1 = abs(dst1).astype(np.uint8) #cv.cvtColor(abs(dst1), cv.COLOR_BGR2GRAY)
    # dst2 = abs(dst2).astype(np.uint8) #cv.cvtColor(abs(dst2), cv.COLOR_BGR2GRAY)

    # cv.imwrite('sobelX.png', dst1)
    # cv.imwrite('sobelY.png', dst2)

    # dst = np.abs(dst2)
    # dstH, _ = getHorizontalAndVertical(dst2)
    # _, dstV = getHorizontalAndVertical(dst1)
    src = cv.GaussianBlur(src, (5, 5), 1)
    Image.fromarray(src[:, :, ::-1]).save('blur.png')
    dstH, dstV = getHorizontalAndVertical(src)
    return dstH, dstV

def tryToDetectSplitSize(img):
    ar = np.array(img)
    #dstH, dstV = dst1, dst2
    #dstH, dstV = (dstH + dstV, dstH + dstV)
    #ar[:, :, 0:3] = ar[:, :, 0:3] * dst[:, :, ::-1]
    #ar[:, :, 0:3] = dst[:, :, ::-1]
    img.save('out.png')
    #fx = lambda ar, x: np.sum(ar[x-2:x+2, :, 0])
    #fy = lambda ar, y: np.sum(ar[:, y-2:y+2, 0])

    dstH, dstV = getHV(img)

    # amount of white on line minus noise
    fx1 = lambda ar, x: np.sum(ar[x, :, 0]) # - np.sum(abs(ar[x, :-1, 0] - ar[x, 1:, 0]))
    fy1 = lambda ar, y: np.sum(ar[:, y, 0]) # - np.sum(abs(ar[:-1, y, 0] - ar[1:, y, 0]))
    # amount of noise on line
    fx2 = lambda ar, x: np.sum(abs(ar[x, :-1, 0] - ar[x, 1:, 0]))
    fy2 = lambda ar, y: np.sum(abs(ar[:-1, y, 0] - ar[1:, y, 0]))

    ar2 = np.copy(ar)

    #dstV, dstH = (dstH, dstV)

    for var, (dst2, f2), (dst, f), detectSize3 in zip(['x', 'y'], [(dstV, fx2), (dstH, fy2)], [(dstH, fx1), (dstV, fy1)], [detectSize3X, detectSize3Y]):
        if True:
            if len(dst.shape) < 3:
                ar[:, :, 0:3] = cv.cvtColor(dst, cv.COLOR_GRAY2RGB)
                ar2[:, :, 0:3] = cv.cvtColor(dst2, cv.COLOR_GRAY2RGB)
            else:
                ar[:, :, 0:3] = dst[:, :, ::-1]
                ar2[:, :, 0:3] = dst2[:, :, ::-1]
        Image.fromarray(ar).save('out2.png')
        res1 = detectSize3(ar, f)
        res2 = detectSize3(ar2, f2)
        d = {}
        for xy, val in res1[-10:]:
            print(var, xy, val)
            d[xy] = val
        print("")
        for xy, val in res2[-10:]:
            print(var, xy, val)
            if xy in d:
                d[xy] += val
            else:
                d[xy] = val
        print("")
        #for xy, val in sorted(d.items(), key=lambda x: x[1]):
        #    print(var, xy, val)
        print("")
    #out = splitSheet(ar)
    #Image.fromarray(out).save('out.png')
    #img.filter(ImageFilter.FIND_EDGES).save('edge.png')

def process(path):
    img = Image.open(path).convert('RGBA')
    ar = np.array(img)
    out, ds, boundsList = splitSheet(ar)
    s = sum(x[1] for x in ds)
    isSheet = s >= 2
    #print(ds)
    #outdir = '/mnt/i/opengameart/unpacked_sprites'
    dirname = os.path.dirname(path).replace('/unpacked/', '/unpacked_sprites/').replace('/files/', '/sprites/')
    Path(dirname).mkdir(parents=True, exist_ok=True)
    bname, ext = os.path.splitext(os.path.basename(path))
    #if not os.path.exists(outdir):
    #    os.mkdir(outdir)
    ar = np.array(img)
    if isSheet:
        k = 0
        for i, bounds in enumerate(boundsList):
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            if width >= 8 and height >= 8 and (width < img.width * 0.8 or height < img.height * 0.8):
                k += 1
                sprite = ar[bounds[0]:bounds[2], bounds[1]:bounds[3]]
                bname2 = bname + '_' + ((5 - len(str(k))) * '0') + str(k) + ext
                outname = os.path.join(dirname, bname2)
                if k == 1 or k % 100 == 0:
                    print("saving", outname)
                Image.fromarray(sprite).save(outname)
        #if k > 0:
        #    flist.write(path + '\n')
        # if ds[-1] >= s // 2:
        # uniformSplittable = ds[-1][1] >= s // 2
    # tryToDetectSplitSize(img)

def processDir():
    #fnames = os.listdir(path)
    savefile = 'allpaths.txt'
    fd = open(savefile, 'r')
    fpaths = [x[:-len('.np')] for x in fd.read().split('\0')[:-1] if x.lower().endswith('.png.np')]
    print("WARNING: skipping /files/ (and /sprites/, which should be skipped)")
    fpaths = [x for x in fpaths if not '/files/' in x and not '/sprites/' in x]

    fd.close()
    #for root, dirs, files in os.walk(path):
    #    for file in files:
    #        if file.lower().endswith('.png'):
    #            fpaths.append(os.path.join(root, file))

    if not os.path.exists('splitsheets.txt'):
        flistfd = open('splitsheets.txt', 'a')
        flistfd.close()
    flistfd = open('splitsheets.txt', 'r')
    flistdata = [x.rstrip() for x in flistfd.readlines()]
    flistfd.close()
    flistdict = {}
    for fl in flistdata:
        flistdict[fl] = True
    flist = open('splitsheets.txt', 'a')
    for i, fpath in enumerate(fpaths):
        if 0:
            if fname == 'smoke.png':
                # oom error
                continue
            if fname == 'trees_mega_pack_cc_by_3_0.png':
                # oom error
                continue
            if fname == 'Urban%20Character%20Pack%20large%20transparent_0.png':
                # oom error
                continue
            if fname == 'Urban%20Character%20Pack%20large.png':
                # oom error
                continue
            if fname == 'Urban%20Character%20Pack%20large_0.png':
                # oom error
                continue
        if '1bit_beauties' in fpath:
            continue
        if 'Hill_1.png' in fpath:
            continue
        if 'BirdPilot_0' in fpath:
            continue
        fname = os.path.basename(fpath)
        flow = fname.lower()
        if flow.endswith('png'):
            #fpath = os.path.join(path, fname)
            if fpath in flistdict:
                continue
            print("processing", i, 'out of', len(fpaths), 'at', fpath)
            try:
                process(fpath)
            except:
                print("error")
            flist.write(fpath + '\n')
    flist.close()

# TODO: delete unicolored "sprites"

if __name__ == "__main__":
    # ar = ar[0:32, 0:32, :]
    processDir()
